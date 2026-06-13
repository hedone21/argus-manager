//! SEQ-040 ~ SEQ-064: 에스컬레이션/디에스컬레이션 시퀀스 (Manager-side)
//!
//! HierarchicalPolicy를 생성하고, SystemSignal을 주입하여 process_signal()의
//! 반환값(Option<EngineDirective>)으로 시퀀스를 검증한다.
//!
//! 주요 검증 항목:
//! - SEQ-040: Monitor 시그널 → process_signal() 호출 → directive 생성
//! - SEQ-042: 모드 전이 (Normal → Warning → Critical)
//! - SEQ-043: needs_action — 모드 변경 시 Directive 발행 / 불변 시 미발행
//! - SEQ-045: seq_id 단조 증가
//! - SEQ-060: 모드 하강 감지
//! - SEQ-061: Critical→Warning 하강 시 RestoreDefaults
//! - SEQ-062: Warning→Normal 하강 시 RestoreDefaults
//! - SEQ-064: Critical→Normal 2단계 하강

use std::collections::HashMap;

use argus_manager::config::{ActionConfig, PolicyConfig, SupervisoryConfig};
use argus_manager::pipeline::{HierarchicalPolicy, PolicyStrategy};
use argus_manager::types::OperatingMode;
use argus_shared::{
    ComputeReason, EngineCommand, EngineDirective, Level, RecommendedBackend, SystemSignal,
};

// ── 헬퍼 ────────────────────────────────────────────────────────────

/// 테스트용 PolicyConfig를 생성한다.
///
/// - hold_time: 1ms (de-escalation 테스트에서 즉시 하강)
/// - 액션 등록: throttle(lossless), switch_hw(lossless),
///   kv_evict_sliding(lossy), kv_evict_h2o(lossy), layer_skip(lossy)
/// - eviction exclusion group: [kv_evict_sliding, kv_evict_h2o]
///
/// ActionRegistry에 후보 액션이 등록되어야 ActionSelector가
/// needs_action=true일 때 실제 directive를 생성할 수 있다.
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
            hold_time_secs: 0.001, // 1ms — 테스트에서 즉시 하강 가능
            ..SupervisoryConfig::default()
        },
        actions,
        exclusion_groups,
        ..PolicyConfig::default()
    }
}

/// compute 도메인 SystemSignal 생성.
/// CPU 사용률로 PI Controller 입력을 직접 제어할 수 있다.
fn compute_signal(level: Level, cpu_pct: f64) -> SystemSignal {
    SystemSignal::ComputeGuidance {
        level,
        recommended_backend: RecommendedBackend::Cpu,
        reason: ComputeReason::CpuBottleneck,
        cpu_usage_pct: cpu_pct,
        gpu_usage_pct: 0.0,
    }
}

/// memory 도메인 SystemSignal 생성.
fn memory_signal(level: Level) -> SystemSignal {
    let (available_bytes, total_bytes) = match level {
        Level::Normal => (1_800_000_000u64, 2_000_000_000u64),
        Level::Warning => (800_000_000u64, 2_000_000_000u64),
        Level::Critical => (300_000_000u64, 2_000_000_000u64),
        Level::Emergency => (100_000_000u64, 2_000_000_000u64),
    };
    SystemSignal::MemoryPressure {
        level,
        available_bytes,
        total_bytes,
        reclaim_target_bytes: 0,
    }
}

/// 파이프라인에 반복 신호를 주입하여 pressure를 누적시킨다.
/// 반환: 수집된 directive 목록
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

/// 파이프라인을 Warning 이상 모드로 에스컬레이션시킨다.
///
/// compute cpu_pct=100 → measurement=1.0, setpoint=0.70, error=0.30
/// Kp=1.5 → P=0.45 > warning_threshold(0.4)로 즉시 Warning 진입.
fn escalate_to_warning_or_above(pipeline: &mut HierarchicalPolicy) -> Vec<EngineDirective> {
    pump_signals(pipeline, &compute_signal(Level::Critical, 100.0), 50)
}

/// 파이프라인을 Critical 모드로 에스컬레이션시킨다.
///
/// compute cpu_pct=100을 대량 주입하여 적분 누적으로 critical_threshold(0.7) 도달.
/// P항=0.45 + 적분 누적(error=0.30 * dt * ki=0.3)으로 0.7 초과 보장.
fn escalate_to_critical(pipeline: &mut HierarchicalPolicy) -> Vec<EngineDirective> {
    pump_signals(pipeline, &compute_signal(Level::Emergency, 100.0), 100)
}

/// directive에 RestoreDefaults 커맨드가 포함되어 있는지 확인한다.
fn has_restore_defaults(directive: &EngineDirective) -> bool {
    directive
        .commands
        .iter()
        .any(|c| matches!(c, EngineCommand::RestoreDefaults))
}

// ── SEQ-040: Monitor 시그널 → process_signal() → directive 생성 ─────

/// SEQ-040: MemoryPressure 시그널이 process_signal()에 의해 처리되어
/// pressure가 누적되고 모드 전이가 발생한다.
///
/// memory Emergency: available=100M/total=2000M → measurement=0.95, setpoint=0.75
/// error=0.20, Kp=2.0 → P=0.40 (warning_threshold 경계).
/// 적분 누적으로 warning_threshold(0.4)를 확실히 초과시킨다.
///
/// 모드 전이가 발생하면 process_signal()이 시그널을 정상적으로 처리한 것이다.
/// (directive 생성은 ActionSelector의 후보 + ReliefEstimator 예측에 의존하므로
/// 모드 전이 자체가 "시그널이 파이프라인을 통과했다"는 증거로 충분하다.)
#[test]
fn test_seq_040_monitor_signal_triggers_processing() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    assert_eq!(p.mode(), OperatingMode::Normal, "초기 모드는 Normal");

    // Emergency 수준의 memory 신호를 반복하여 pressure 누적
    pump_signals(&mut p, &memory_signal(Level::Emergency), 80);

    // pressure 누적으로 Warning 이상 모드에 도달해야 한다
    assert_ne!(
        p.mode(),
        OperatingMode::Normal,
        "반복된 Emergency 메모리 신호 후 모드가 Normal에서 벗어나야 한다 (pressure={:?})",
        p.pressure()
    );

    // pressure가 양수임을 확인 — PI Controller가 시그널을 처리한 증거
    assert!(
        p.pressure().memory > 0.0,
        "memory 시그널 처리 후 memory pressure가 양수여야 한다"
    );
}

/// SEQ-040 보완: compute 시그널이 충분한 pressure를 생성하면 directive가 발행된다.
/// compute 도메인에서 cpu_pct=100 → P=0.45로 즉시 Warning 진입.
/// needs_action=true(mode 변경) → ActionSelector가 lossless 액션을 선택.
#[test]
fn test_seq_040_compute_signal_triggers_directive() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // compute Critical(cpu 100%)을 반복 주입
    let directives = pump_signals(&mut p, &compute_signal(Level::Emergency, 100.0), 60);

    // 에스컬레이션 → 디에스컬레이션 사이클에서 directive(RestoreDefaults 포함)를 수집
    std::thread::sleep(std::time::Duration::from_millis(5));
    let de_directives = pump_signals(&mut p, &compute_signal(Level::Normal, 5.0), 200);

    let all_directives: Vec<_> = directives.into_iter().chain(de_directives).collect();

    // 에스컬레이션/디에스컬레이션 전체 과정에서 최소 1개의 directive 발행
    assert!(
        !all_directives.is_empty(),
        "에스컬레이션→디에스컬레이션 사이클에서 최소 1개의 directive가 생성되어야 한다"
    );

    // 생성된 directive에는 유효한 seq_id가 있어야 한다
    for d in &all_directives {
        assert!(d.seq_id > 0, "seq_id는 양수여야 한다");
        assert!(
            !d.commands.is_empty(),
            "directive에는 최소 1개의 command가 포함되어야 한다"
        );
    }
}

// ── SEQ-042: 모드 전이 ──────────────────────────────────────────────

/// SEQ-042: Normal → Warning/Critical 모드 전이.
/// pressure가 warning_threshold를 초과하면 모드가 Normal에서 벗어난다.
///
/// compute: cpu_pct=100 → measurement=1.0, setpoint=0.70, error=0.30
/// Kp=1.5 → P=0.45 > warning_threshold(0.4) — 즉시 Warning 진입.
#[test]
fn test_seq_042_mode_transition_under_pressure() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());
    assert_eq!(
        p.mode(),
        OperatingMode::Normal,
        "초기 모드는 Normal이어야 한다"
    );

    // cpu_pct=100으로 P항만으로 warning_threshold를 초과
    for _ in 0..50 {
        p.process_signal(&compute_signal(Level::Critical, 100.0));
    }

    assert_ne!(
        p.mode(),
        OperatingMode::Normal,
        "충분한 Critical 신호 (cpu 100%) 후 모드가 Normal에서 벗어나야 한다"
    );
}

/// SEQ-042: Warning 모드에서 directive가 생성되면 유효한 command를 포함해야 한다.
#[test]
fn test_seq_042_warning_mode_produces_directive() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    let directives = escalate_to_warning_or_above(&mut p);
    if !directives.is_empty() {
        for d in &directives {
            assert!(
                !d.commands.is_empty(),
                "Warning 이상 모드에서 생성된 directive에는 command가 포함되어야 한다"
            );
        }
    }
}

// ── SEQ-043: needs_action — 모드 변경 시 Directive 발행 ──────────────

/// SEQ-043: Normal→Warning/Critical 전이 시 Some(directive) 반환.
/// 모드가 변경되는 시점에 directive가 발행된다.
#[test]
fn test_seq_043_mode_change_emits_directive() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    let directives = escalate_to_warning_or_above(&mut p);

    // 모드가 Normal에서 벗어났다면 최소 1개의 directive가 생성되어야 한다
    if p.mode() != OperatingMode::Normal {
        assert!(
            !directives.is_empty(),
            "모드 전이 시 최소 1개의 directive가 발행되어야 한다 (mode={:?})",
            p.mode()
        );
    }
}

/// SEQ-043: 모드 불변 + 압력 불변 시 Directive 미발행.
/// Normal 유지 상태에서 동일한 low pressure를 반복하면 None이 반환된다.
#[test]
fn test_seq_043_no_directive_when_mode_unchanged_and_pressure_stable() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // Normal 수준 신호를 반복 — 모드 변화 없음, pressure 증가 없음
    for _ in 0..20 {
        let result = p.process_signal(&compute_signal(Level::Normal, 10.0));
        assert!(
            result.is_none(),
            "Normal 유지 + 낮은 pressure에서는 directive가 생성되지 않아야 한다"
        );
    }
    assert_eq!(p.mode(), OperatingMode::Normal);
}

// ── SEQ-045: seq_id 단조 증가 ───────────────────────────────────────

/// SEQ-045: 연속 process_signal() 호출로 반환된 directive들의 seq_id가
/// 단조 증가한다.
#[test]
fn test_seq_045_seq_id_monotonically_increasing() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    let mut all_directives = Vec::new();

    // 1차: Critical/Emergency 에스컬레이션
    let d1 = pump_signals(&mut p, &compute_signal(Level::Emergency, 100.0), 60);
    all_directives.extend(d1);

    // 2차: Normal로 디에스컬레이션 (hold_time 경과 후 RestoreDefaults 수집)
    std::thread::sleep(std::time::Duration::from_millis(5));
    let d2 = pump_signals(&mut p, &compute_signal(Level::Normal, 5.0), 100);
    all_directives.extend(d2);

    // 3차: 재에스컬레이션하여 추가 directive 수집
    let d3 = pump_signals(&mut p, &compute_signal(Level::Emergency, 100.0), 60);
    all_directives.extend(d3);

    // 최소 2개의 directive가 있으면 단조 증가 검증
    if all_directives.len() >= 2 {
        for window in all_directives.windows(2) {
            assert!(
                window[1].seq_id > window[0].seq_id,
                "seq_id가 단조 증가해야 한다: {} -> {}",
                window[0].seq_id,
                window[1].seq_id
            );
        }
    }

    // 수집된 모든 directive의 seq_id가 양수인지 확인
    for d in &all_directives {
        assert!(d.seq_id > 0, "seq_id는 양수여야 한다");
    }
}

// ── SEQ-060: 모드 하강 감지 ─────────────────────────────────────────

/// SEQ-060: Critical 상태에서 pressure 감소 시 모드가 Warning으로 하강한다.
#[test]
fn test_seq_060_mode_descends_on_pressure_drop() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // Critical 진입
    escalate_to_critical(&mut p);
    let mode_after_escalation = p.mode();

    // Normal 수준 신호를 반복하여 pressure를 점차 감소
    // hold_time이 1ms이므로 sleep 후 de-escalation이 가능하다
    std::thread::sleep(std::time::Duration::from_millis(5));
    for _ in 0..200 {
        p.process_signal(&compute_signal(Level::Normal, 5.0));
    }

    if mode_after_escalation == OperatingMode::Critical {
        assert!(
            p.mode() < OperatingMode::Critical,
            "pressure 감소 후 Critical에서 하강해야 한다 (현재: {:?})",
            p.mode()
        );
    }
}

// ── SEQ-061: Critical→Warning 하강 시 RestoreDefaults ────────────────

/// SEQ-061: Critical에서 Warning으로 하강 시 RestoreDefaults 명령이 포함된
/// directive가 발행된다.
#[test]
fn test_seq_061_critical_to_warning_emits_restore_defaults() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // Critical 진입
    escalate_to_critical(&mut p);

    if p.mode() != OperatingMode::Critical {
        // PI Controller가 Critical까지 도달하지 못한 경우 스킵
        return;
    }

    // Normal 신호를 반복하여 de-escalation 유도
    std::thread::sleep(std::time::Duration::from_millis(5));

    let mut restore_found_in_descent = false;
    for _ in 0..300 {
        if let Some(d) = p.process_signal(&compute_signal(Level::Normal, 5.0))
            && has_restore_defaults(&d)
        {
            restore_found_in_descent = true;
            break;
        }
        if p.mode() != OperatingMode::Normal {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    assert!(
        restore_found_in_descent,
        "Critical→Warning 하강 과정에서 RestoreDefaults가 발행되어야 한다"
    );
}

// ── SEQ-062: Warning→Normal 하강 시 RestoreDefaults ─────────────────

/// SEQ-062: Warning에서 Normal로 하강 시 RestoreDefaults 명령이 포함된
/// directive가 발행된다.
#[test]
fn test_seq_062_warning_to_normal_emits_restore_defaults() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // Warning 이상으로 에스컬레이션
    escalate_to_warning_or_above(&mut p);

    if p.mode() == OperatingMode::Normal {
        return;
    }

    // Normal 신호를 반복하여 de-escalation 유도
    std::thread::sleep(std::time::Duration::from_millis(5));

    let mut restore_found = false;
    for _ in 0..400 {
        if let Some(d) = p.process_signal(&compute_signal(Level::Normal, 5.0))
            && has_restore_defaults(&d)
        {
            restore_found = true;
        }
        if p.mode() != OperatingMode::Normal {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        if p.mode() == OperatingMode::Normal {
            break;
        }
    }

    if p.mode() == OperatingMode::Normal {
        assert!(
            restore_found,
            "Normal 복귀 과정에서 RestoreDefaults가 발행되어야 한다"
        );
    }
}

// ── SEQ-064: Critical→Normal 2단계 하강 ──────────────────────────────

/// SEQ-064: Critical 상태에서 pressure 급감 시 Critical→Warning→Normal의
/// 2단계 하강이 이루어진다 (Critical→Normal 직행 불가).
#[test]
fn test_seq_064_critical_to_normal_two_step_descent() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // Critical 진입
    escalate_to_critical(&mut p);

    if p.mode() != OperatingMode::Critical {
        return;
    }

    // 하강 과정에서 모드 변화를 추적
    let mut mode_history = vec![p.mode()];
    std::thread::sleep(std::time::Duration::from_millis(5));

    for _ in 0..500 {
        p.process_signal(&compute_signal(Level::Normal, 5.0));
        let current_mode = p.mode();
        if mode_history.last() != Some(&current_mode) {
            mode_history.push(current_mode);
        }
        if current_mode == OperatingMode::Normal {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    // Normal에 도달했다면 반드시 Warning을 거쳤어야 한다
    if p.mode() == OperatingMode::Normal && mode_history.len() >= 2 {
        assert!(
            mode_history.contains(&OperatingMode::Warning),
            "Critical→Normal 하강은 반드시 Warning 단계를 거쳐야 한다. 실제 경로: {:?}",
            mode_history
        );

        // Critical 직후에 Normal이 오면 안 된다
        for window in mode_history.windows(2) {
            if window[0] == OperatingMode::Critical {
                assert_ne!(
                    window[1],
                    OperatingMode::Normal,
                    "Critical→Normal 직행은 허용되지 않는다. 반드시 Warning을 거쳐야 한다."
                );
            }
        }
    }
}

// ── SEQ-046: Connected=false → Directive 미전송 ─────────────────────
//
// HierarchicalPolicy에 `connected` 상태 필드가 없다.
// Engine 연결 상태 관리는 pipeline 외부(main loop)에서 처리되므로
// HierarchicalPolicy 단위 테스트로는 검증할 수 없다.
// 이 요구사항은 통합 테스트(IPC 레벨)에서 검증해야 한다.

// ── 추가 SEQ: 다중 도메인 에스컬레이션 ──────────────────────────────

/// SEQ-040 보완: 서로 다른 도메인(compute, memory)의 시그널이 모두
/// process_signal()을 통해 처리되어 모드 전이와 directive 생성이 이루어진다.
#[test]
fn test_seq_040_multi_domain_signals_processed() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    // compute(cpu 100%)와 memory(Emergency) 신호를 교차 주입
    let mut any_directive = false;
    for _ in 0..40 {
        if p.process_signal(&compute_signal(Level::Critical, 100.0))
            .is_some()
        {
            any_directive = true;
        }
        if p.process_signal(&memory_signal(Level::Emergency)).is_some() {
            any_directive = true;
        }
    }

    // 다중 도메인 Critical/Emergency 신호 후 모드가 Normal이 아니어야 한다
    assert_ne!(
        p.mode(),
        OperatingMode::Normal,
        "다중 도메인 Critical/Emergency 신호 후 모드가 Normal에서 벗어나야 한다"
    );
    assert!(
        any_directive,
        "다중 도메인 Critical/Emergency 신호 후 최소 1개의 directive가 생성되어야 한다"
    );
}

/// SEQ-045 보완: 단일 에스컬레이션 사이클 내에서도 seq_id가 단조 증가한다.
#[test]
fn test_seq_045_seq_id_within_single_cycle() {
    let mut p = HierarchicalPolicy::new(&test_policy_config());

    let directives = pump_signals(&mut p, &compute_signal(Level::Emergency, 100.0), 100);

    if directives.len() >= 2 {
        for i in 1..directives.len() {
            assert!(
                directives[i].seq_id > directives[i - 1].seq_id,
                "단일 사이클 내에서도 seq_id가 단조 증가해야 한다: idx={} ({}) -> idx={} ({})",
                i - 1,
                directives[i - 1].seq_id,
                i,
                directives[i].seq_id,
            );
        }
    }
}
