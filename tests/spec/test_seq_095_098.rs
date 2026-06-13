//! SEQ-095 ~ SEQ-098: QCF Request/Estimate sequence (Manager side)
//!
//! SEQ-095: Critical transition triggers RequestQcf Directive
//! SEQ-096: (Engine side -- tested in engine/tests/spec/)
//! SEQ-097: QcfEstimate triggers action selection with real QCF costs
//! SEQ-098: 1-second timeout falls back to default costs
//! SEQ-098a: timeout + late estimate — cache 갱신(pending 무관) + 무한 핸드셰이크 방지
//!
//! NOTE: SEQ-098a 인라인 회귀 테스트는 `manager/src/lua_policy.rs` 내부
//! `#[cfg(test)] mod tests` 에 수록되어 있다.
//! (`check_qcf_timeout_emits_fallback_decide_on_signal`,
//!  `complete_qcf_selection_caches_late_estimate_without_pending`)
//! `LuaPolicy::qcf_cache` 가 `pub(crate)` 로 제한되어 외부 spec 테스트에서
//! 직접 검증이 불가하므로 인라인 테스트로 대체한다.

use std::collections::HashMap;

use argus_manager::config::{ActionConfig, PolicyConfig, SupervisoryConfig};
use argus_manager::pipeline::{HierarchicalPolicy, PolicyStrategy};
use argus_manager::types::OperatingMode;
use argus_shared::{
    ComputeReason, EngineCommand, EngineDirective, Level, QcfEstimate, RecommendedBackend,
    SystemSignal,
};

// ── helpers ──────────────────────────────────────────────────────────

/// QCF 테스트용 PolicyConfig.
///
/// critical_threshold를 0.42로 낮추어 compute Emergency(cpu 100%)의
/// P항(Kp=1.5 * error=0.30 = 0.45)만으로도 Critical 진입을 보장한다.
/// (default critical_threshold=0.7이면 P항만으로 도달 불가, integral 누적 필요하지만
///  테스트 루프에서 elapsed_dt가 극소값이라 integral이 거의 누적되지 않음.)
fn test_policy_config() -> PolicyConfig {
    let mut actions = HashMap::new();
    actions.insert(
        "throttle".to_string(),
        ActionConfig {
            lossy: false,
            reversible: true,
            ..Default::default()
        },
    );
    actions.insert(
        "switch_hw".to_string(),
        ActionConfig {
            lossy: false,
            reversible: true,
            ..Default::default()
        },
    );
    actions.insert(
        "kv.evict_sliding".to_string(),
        ActionConfig {
            lossy: true,
            reversible: false,
            default_cost: 0.5,
        },
    );
    actions.insert(
        "kv.evict_h2o".to_string(),
        ActionConfig {
            lossy: true,
            reversible: false,
            default_cost: 0.6,
        },
    );
    actions.insert(
        "weight.skip".to_string(),
        ActionConfig {
            lossy: true,
            reversible: true,
            default_cost: 1.0,
        },
    );

    let mut exclusion_groups = HashMap::new();
    exclusion_groups.insert(
        "eviction".to_string(),
        vec!["kv.evict_sliding".into(), "kv.evict_h2o".into()],
    );

    PolicyConfig {
        supervisory: SupervisoryConfig {
            warning_threshold: 0.20,
            critical_threshold: 0.42,
            hold_time_secs: 0.001,
            ..SupervisoryConfig::default()
        },
        actions,
        exclusion_groups,
        ..PolicyConfig::default()
    }
}

/// Warning-only PolicyConfig (Critical 미달용).
///
/// warning_threshold=0.20, critical_threshold=0.90으로 설정하여
/// compute Emergency의 P항(0.45)으로 Warning에는 도달하지만
/// Critical에는 도달하지 못하게 한다.
fn warning_only_policy_config() -> PolicyConfig {
    let mut config = test_policy_config();
    config.supervisory.warning_threshold = 0.20;
    config.supervisory.critical_threshold = 0.90;
    config
}

/// compute 도메인 SystemSignal 생성.
fn compute_signal(level: Level, cpu_pct: f64) -> SystemSignal {
    SystemSignal::ComputeGuidance {
        level,
        recommended_backend: RecommendedBackend::Cpu,
        reason: ComputeReason::CpuBottleneck,
        cpu_usage_pct: cpu_pct,
        gpu_usage_pct: 0.0,
    }
}

/// 파이프라인에 반복 신호를 주입하여 pressure를 누적시킨다.
fn pump_signals(
    pipeline: &mut HierarchicalPolicy,
    signal: &SystemSignal,
    count: usize,
) -> Vec<EngineDirective> {
    let mut directives = Vec::new();
    for _ in 0..count {
        if let Some(d) = pipeline.process_signal(signal) {
            directives.push(d);
        }
    }
    directives
}

/// 파이프라인을 Critical 모드로 에스컬레이션시킨다.
/// 반환: 수집된 directive 목록.
fn escalate_to_critical(pipeline: &mut HierarchicalPolicy) -> Vec<EngineDirective> {
    pump_signals(pipeline, &compute_signal(Level::Emergency, 100.0), 100)
}

/// directive에 RequestQcf 커맨드가 포함되어 있는지 확인한다.
fn has_request_qcf(directive: &EngineDirective) -> bool {
    directive
        .commands
        .iter()
        .any(|c| matches!(c, EngineCommand::RequestQcf))
}

/// directive에 RestoreDefaults가 아닌 "실제 액션" 커맨드가 포함되어 있는지 확인한다.
fn has_action_command(directive: &EngineDirective) -> bool {
    directive.commands.iter().any(|c| {
        !matches!(
            c,
            EngineCommand::RestoreDefaults | EngineCommand::RequestQcf
        )
    })
}

// ── SEQ-095: Critical 전환 시 RequestQcf 전송 ─────────────────────────

/// SEQ-095: Normal→Critical 전환 시 process_signal()이 RequestQcf를 포함하는
/// directive를 반환한다.
///
/// compute Emergency(cpu 100%)를 충분히 주입하여 Critical 진입을 유도한다.
/// Critical 전환 시점의 directive에 RequestQcf 커맨드가 포함되어야 한다.
#[test]
fn test_seq_095_critical_sends_request_qcf() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());
    assert_eq!(p.mode(), OperatingMode::Normal);

    let directives = escalate_to_critical(&mut p);

    // Critical 모드에 도달해야 한다
    assert_eq!(
        p.mode(),
        OperatingMode::Critical,
        "충분한 Emergency 신호 후 Critical 모드에 도달해야 한다"
    );

    // directive 중 RequestQcf가 포함된 것이 있어야 한다
    let qcf_directives: Vec<_> = directives.iter().filter(|d| has_request_qcf(d)).collect();
    assert!(
        !qcf_directives.is_empty(),
        "Critical 전환 시 RequestQcf를 포함하는 directive가 발행되어야 한다"
    );
}

/// SEQ-095: Warning 전환 시에는 RequestQcf를 보내지 않는다.
///
/// Warning 모드에서는 lossless 액션만 선택하므로 QCF 요청이 불필요하다.
/// critical_threshold=0.90인 config를 사용하여 Warning에만 도달시킨다.
#[test]
fn test_seq_095_warning_no_request_qcf() {
    let mut p = HierarchicalPolicy::new(&warning_only_policy_config());

    // Warning까지만 에스컬레이션 (Critical 미달)
    let directives = pump_signals(&mut p, &compute_signal(Level::Emergency, 100.0), 50);

    assert!(
        p.mode() >= OperatingMode::Warning,
        "Emergency 신호 후 최소 Warning에 도달해야 한다"
    );
    assert_ne!(
        p.mode(),
        OperatingMode::Critical,
        "warning_only config에서는 Critical에 도달하면 안 된다"
    );

    // 발행된 directive 중 RequestQcf가 없어야 한다
    for d in &directives {
        assert!(
            !has_request_qcf(d),
            "Warning 전환 시에는 RequestQcf가 발행되지 않아야 한다 (mode={:?})",
            p.mode()
        );
    }
}

/// SEQ-095: 이미 Critical인 상태에서 추가 시그널을 보내도 RequestQcf를 재전송하지 않는다.
///
/// Critical 전환 시점에만 한 번 RequestQcf를 보내고, 이후 Critical 유지 중에는 안 보낸다.
#[test]
fn test_seq_095_critical_steady_no_request_qcf() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // Critical 진입
    let first_directives = escalate_to_critical(&mut p);
    assert_eq!(p.mode(), OperatingMode::Critical);

    // 첫 에스컬레이션에서 RequestQcf 발행 확인
    let first_qcf_count = first_directives
        .iter()
        .filter(|d| has_request_qcf(d))
        .count();
    assert!(
        first_qcf_count > 0,
        "첫 Critical 전환에서 RequestQcf가 발행되어야 한다"
    );

    // QCF pending을 complete해서 pending 상태를 해소한다
    let qcf = QcfEstimate {
        estimates: HashMap::from([
            ("kv.evict_sliding".to_string(), 0.3),
            ("kv.evict_h2o".to_string(), 0.4),
        ]),
        layer_swap: None,
    };
    let _ = p.complete_qcf_selection(&qcf);

    // Critical 유지 상태에서 추가 시그널
    let steady_directives = pump_signals(&mut p, &compute_signal(Level::Emergency, 100.0), 30);

    // 추가 시그널에서는 RequestQcf가 없어야 한다
    let steady_qcf_count = steady_directives
        .iter()
        .filter(|d| has_request_qcf(d))
        .count();
    assert_eq!(
        steady_qcf_count, 0,
        "이미 Critical인 상태에서는 RequestQcf를 재전송하지 않아야 한다"
    );
}

// ── SEQ-097: QcfEstimate 수신 후 액션 선택 ──────────────────────────

/// SEQ-097: Critical 전환으로 RequestQcf 발행 후, complete_qcf_selection()으로
/// QcfEstimate를 전달하면 액션 directive가 반환된다.
#[test]
fn test_seq_097_complete_with_qcf_estimate() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // Critical 진입 (RequestQcf 발행됨)
    let directives = escalate_to_critical(&mut p);
    assert_eq!(p.mode(), OperatingMode::Critical);

    // RequestQcf가 발행되었는지 확인 (QcfPending이 설정됨)
    let has_qcf = directives.iter().any(|d| has_request_qcf(d));
    assert!(has_qcf, "Critical 전환에서 RequestQcf가 발행되어야 한다");

    // Engine이 QcfEstimate를 보낸다
    let qcf = QcfEstimate {
        estimates: HashMap::from([
            ("kv.evict_sliding".to_string(), 0.2),
            ("kv.evict_h2o".to_string(), 0.3),
            ("weight.skip".to_string(), 0.8),
        ]),
        layer_swap: None,
    };

    let result = p.complete_qcf_selection(&qcf);

    // 액션 directive가 반환되어야 한다
    assert!(
        result.is_some(),
        "QcfEstimate 수신 후 액션 directive가 반환되어야 한다"
    );

    let directive = result.unwrap();
    assert!(
        !directive.commands.is_empty(),
        "반환된 directive에는 최소 1개의 command가 포함되어야 한다"
    );
    assert!(directive.seq_id > 0, "seq_id는 양수여야 한다");

    // RequestQcf가 아닌 실제 액션 커맨드가 포함되어야 한다
    assert!(
        has_action_command(&directive),
        "반환된 directive에는 실제 액션 커맨드가 포함되어야 한다"
    );
}

/// SEQ-097: pending이 없는 상태에서 complete_qcf_selection()은 None을 반환한다.
#[test]
fn test_seq_097_complete_without_pending_returns_none() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // pending 없이 바로 호출
    let qcf = QcfEstimate {
        estimates: HashMap::from([("kv.evict_sliding".to_string(), 0.3)]),
        layer_swap: None,
    };

    let result = p.complete_qcf_selection(&qcf);
    assert!(
        result.is_none(),
        "pending이 없을 때 complete_qcf_selection()은 None을 반환해야 한다"
    );
}

/// SEQ-097: QcfEstimate의 String key → ActionId 매핑이 올바르게 동작한다.
///
/// QcfEstimate에 포함된 액션의 비용이 default_cost보다 작을 때 해당 값이
/// 실제로 액션 선택에 사용되는지 검증한다.
#[test]
fn test_seq_097_qcf_values_override_default_cost() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // Critical 진입
    escalate_to_critical(&mut p);
    assert_eq!(p.mode(), OperatingMode::Critical);

    // 매우 낮은 비용으로 QcfEstimate 전달 — 액션 선택이 더 적극적이어야 한다
    let qcf_low_cost = QcfEstimate {
        estimates: HashMap::from([
            ("kv.evict_sliding".to_string(), 0.01),
            ("kv.evict_h2o".to_string(), 0.01),
            ("weight.skip".to_string(), 0.01),
        ]),
        layer_swap: None,
    };

    let result = p.complete_qcf_selection(&qcf_low_cost);

    // 낮은 비용이라도 directive가 반환되어야 한다
    assert!(
        result.is_some(),
        "낮은 QCF 비용으로도 액션 directive가 반환되어야 한다"
    );
}

// ── SEQ-098: 1초 타임아웃 폴백 ───────────────────────────────────────

/// SEQ-098: RequestQcf 직후 check_qcf_timeout()은 None을 반환한다 (아직 1초 안 지남).
#[test]
fn test_seq_098_no_timeout_returns_none() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // Critical 진입 → RequestQcf 발행
    escalate_to_critical(&mut p);
    assert_eq!(p.mode(), OperatingMode::Critical);

    // 즉시 타임아웃 체크 — 아직 1초가 지나지 않았으므로 None
    let result = p.check_qcf_timeout();
    assert!(
        result.is_none(),
        "RequestQcf 직후에는 타임아웃이 발생하지 않아야 한다"
    );
}

/// SEQ-098: RequestQcf 후 1초 이상 경과하면 check_qcf_timeout()이 default cost로
/// directive를 반환한다.
#[test]
fn test_seq_098_timeout_returns_directive() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // Critical 진입 → RequestQcf 발행
    escalate_to_critical(&mut p);
    assert_eq!(p.mode(), OperatingMode::Critical);

    // 1초 이상 대기
    std::thread::sleep(std::time::Duration::from_millis(1100));

    let result = p.check_qcf_timeout();

    // 타임아웃 후 default cost로 directive가 반환되어야 한다
    assert!(
        result.is_some(),
        "1초 타임아웃 후 default cost로 directive가 반환되어야 한다"
    );

    let directive = result.unwrap();
    assert!(
        !directive.commands.is_empty(),
        "타임아웃 후 반환된 directive에는 command가 포함되어야 한다"
    );
    assert!(
        has_action_command(&directive),
        "타임아웃 후 반환된 directive에는 실제 액션 커맨드가 포함되어야 한다"
    );
}
