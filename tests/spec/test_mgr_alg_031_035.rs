//! MGR-ALG-030 ~ MGR-ALG-035: ActionSelector Spec 테스트
//!
//! Cross-Domain Action Selector의 후보 필터링, 최적 조합 탐색,
//! exclusion group, latency budget, best-effort, 파라미터 보간을 검증한다.

use std::collections::HashMap;

use argus_manager::selector::ActionSelector;
use argus_manager::types::{ActionCommand, ActionId, OperatingMode, Operation, PressureVector};

use super::helpers::{MockEstimator, make_registry, no_state, rv};

fn pv(compute: f32, memory: f32, thermal: f32) -> PressureVector {
    PressureVector {
        compute,
        memory,
        thermal,
    }
}

fn command_ids(cmds: &[ActionCommand]) -> Vec<ActionId> {
    cmds.iter().map(|c| c.action).collect()
}

// ── MGR-ALG-030: pressure 없으면 빈 결과 ─────────────────────────────

/// MGR-ALG-030: pressure가 0이면 빈 결과를 반환한다.
#[test]
fn test_mgr_alg_030_no_action_when_no_pressure() {
    let registry = make_registry(
        &[
            ("switch_hw", false, true),
            ("kv.evict_sliding", true, false),
        ],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.6, 0.0, 0.4, -0.2));
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.8, 0.0, 0.1));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.0, 0.0, 0.0);
    let state = no_state();
    let qcf = HashMap::new();

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Critical,
        &state,
        &qcf,
        1.0,
        &[],
        &[],
    );

    assert!(cmds.is_empty(), "no action needed when pressure is zero");
}

// ── MGR-ALG-031 Rule1: Warning mode에서 lossy 제외 ──────────────────

/// MGR-ALG-031: Warning mode에서 lossy 후보가 필터링되어 lossless만 선택된다.
#[test]
fn test_mgr_alg_031_warning_mode_lossless_only() {
    let registry = make_registry(
        &[
            ("switch_hw", false, true),
            ("kv.evict_sliding", true, false),
        ],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, -0.1));
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.1));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.6, 0.0, 0.0);
    let state = no_state();
    let qcf = HashMap::new();

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Warning,
        &state,
        &qcf,
        1.0,
        &[],
        &[],
    );

    let ids = command_ids(&cmds);
    assert!(
        !ids.contains(&ActionId::KvEvictSliding),
        "lossy action must not appear in Warning mode"
    );
    assert!(
        ids.contains(&ActionId::SwitchHw),
        "lossless action should be selected"
    );
}

/// MGR-ALG-031: Warning mode이고 모든 액션이 lossy이면 빈 결과.
#[test]
fn test_mgr_alg_031_empty_candidates() {
    let registry = make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("weight.skip", true, true),
        ],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
    predictions.insert(ActionId::LayerSkip, rv(0.4, 0.0, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.5, 0.6, 0.0);
    let state = no_state();
    let qcf = HashMap::new();

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Warning,
        &state,
        &qcf,
        1.0,
        &[],
        &[],
    );

    assert!(
        cmds.is_empty(),
        "should return empty when all candidates are lossy in Warning mode"
    );
}

// ── MGR-ALG-031 Rule2: 활성 액션 제외 ──────────────────────────────

/// MGR-ALG-031: active 액션은 후보에서 제외된다.
#[test]
fn test_mgr_alg_031_active_action_excluded() {
    let registry = make_registry(
        &[("switch_hw", false, true), ("throttle", false, true)],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, 0.0));
    predictions.insert(ActionId::Throttle, rv(0.4, 0.0, 0.2, -0.3));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.5, 0.0, 0.0);
    let state = no_state();
    let qcf = HashMap::new();
    let active_actions = [ActionId::SwitchHw];

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Warning,
        &state,
        &qcf,
        1.0,
        &active_actions,
        &[],
    );

    let ids = command_ids(&cmds);
    assert!(
        !ids.contains(&ActionId::SwitchHw),
        "already active action should not be selected again"
    );
}

// ── MGR-ALG-031 Rule3: available_actions 필터링 ─────────────────────

/// MGR-ALG-031: available_actions가 비어있지 않으면 해당 목록에 없는 액션 제외.
#[test]
fn test_mgr_alg_031_available_actions_filters_candidates() {
    let registry = make_registry(
        &[
            ("switch_hw", false, true),
            ("throttle", false, true),
            ("kv.evict_h2o", true, false),
        ],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, 0.0));
    predictions.insert(ActionId::Throttle, rv(0.4, 0.0, 0.3, 0.0));
    predictions.insert(ActionId::KvEvictH2o, rv(0.0, 0.9, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.6, 0.0, 0.0);
    let state = no_state();
    let qcf = HashMap::new();
    let available = [ActionId::Throttle];

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Critical,
        &state,
        &qcf,
        1.0,
        &[],
        &available,
    );

    let ids = command_ids(&cmds);
    assert!(
        !ids.contains(&ActionId::SwitchHw),
        "switch_hw not in available_actions must be excluded"
    );
    assert!(
        ids.contains(&ActionId::Throttle),
        "throttle is in available_actions and should be selectable"
    );
}

/// MGR-ALG-031: available_actions가 비어있으면 필터링 안 함 (backward compat).
#[test]
fn test_mgr_alg_031_empty_available_actions_no_filtering() {
    let registry = make_registry(
        &[("switch_hw", false, true), ("throttle", false, true)],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, 0.0));
    predictions.insert(ActionId::Throttle, rv(0.4, 0.0, 0.3, 0.0));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.6, 0.0, 0.0);
    let state = no_state();
    let qcf = HashMap::new();

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Critical,
        &state,
        &qcf,
        1.0,
        &[],
        &[], // 비어있음 -> 필터링 없음
    );

    let ids = command_ids(&cmds);
    assert!(
        ids.contains(&ActionId::SwitchHw),
        "switch_hw should be selectable when available_actions is empty"
    );
}

// ── MGR-ALG-033: Critical mode 최적 조합 ────────────────────────────

/// MGR-ALG-033: Critical mode에서 cost가 가장 낮은 조합을 선택한다.
#[test]
fn test_mgr_alg_033_critical_mode_minimum_cost() {
    let registry = make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
        ],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
    predictions.insert(ActionId::KvEvictH2o, rv(0.0, 0.9, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.0, 0.7, 0.0);
    let state = no_state();
    let mut qcf = HashMap::new();
    qcf.insert(ActionId::KvEvictSliding, 0.5f32);
    qcf.insert(ActionId::KvEvictH2o, 2.0f32);

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Critical,
        &state,
        &qcf,
        1.0,
        &[],
        &[],
    );

    let ids = command_ids(&cmds);
    assert!(
        ids.contains(&ActionId::KvEvictSliding),
        "cheaper action should be selected"
    );
    assert!(
        !ids.contains(&ActionId::KvEvictH2o),
        "more expensive action should not be added unnecessarily"
    );
}

/// MGR-ALG-033: exclusion group으로 동일 그룹의 액션은 동시 선택 불가.
#[test]
fn test_mgr_alg_033_exclusion_group() {
    let registry = make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
        ],
        &[("eviction", &["kv.evict_sliding", "kv.evict_h2o"])],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.6, 0.0, 0.0));
    predictions.insert(ActionId::KvEvictH2o, rv(0.0, 0.6, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.0, 0.5, 0.0);
    let state = no_state();
    let mut qcf = HashMap::new();
    qcf.insert(ActionId::KvEvictSliding, 1.0f32);
    qcf.insert(ActionId::KvEvictH2o, 1.0f32);

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Critical,
        &state,
        &qcf,
        1.0,
        &[],
        &[],
    );

    let ids = command_ids(&cmds);
    let evict_count = ids
        .iter()
        .filter(|&&id| id == ActionId::KvEvictSliding || id == ActionId::KvEvictH2o)
        .count();
    assert_eq!(
        evict_count, 1,
        "only one eviction action should be selected due to exclusion group"
    );
}

/// MGR-ALG-033: latency budget 초과 시 조합 불가.
#[test]
fn test_mgr_alg_033_latency_budget_constraint() {
    let registry = make_registry(
        &[("throttle", false, true), ("kv_offload_disk", false, true)],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::Throttle, rv(0.5, 0.0, 0.3, -0.4));
    predictions.insert(ActionId::KvOffloadDisk, rv(0.0, 0.8, 0.0, -0.4));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.4, 0.7, 0.0);
    let state = no_state();
    let qcf = HashMap::new();

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Warning,
        &state,
        &qcf,
        0.6, // latency_budget = 0.6
        &[],
        &[],
    );

    let ids = command_ids(&cmds);
    let both_selected = ids.contains(&ActionId::Throttle) && ids.contains(&ActionId::KvOffloadDisk);
    assert!(
        !both_selected,
        "throttle+offload together exceeds latency budget"
    );
}

/// MGR-ALG-033: 모든 조합으로도 해소 불가 시 coverage 최대화 (best-effort).
#[test]
fn test_mgr_alg_033_best_effort_when_impossible() {
    let registry = make_registry(
        &[("switch_hw", false, true), ("throttle", false, true)],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.3, 0.0, 0.0, 0.0));
    predictions.insert(ActionId::Throttle, rv(0.2, 0.0, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.8, 0.0, 0.0);
    let state = no_state();
    let qcf = HashMap::new();

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Warning,
        &state,
        &qcf,
        1.0,
        &[],
        &[],
    );

    let ids = command_ids(&cmds);
    assert!(
        ids.contains(&ActionId::SwitchHw),
        "best-effort: switch_hw should be included"
    );
    assert!(
        ids.contains(&ActionId::Throttle),
        "best-effort: throttle should be included"
    );
}

/// MGR-ALG-033: cross-domain 해소 시 단일 액션만으로 충분하면 다른 액션 불필요.
#[test]
fn test_mgr_alg_033_cross_domain_single_action() {
    let registry = make_registry(
        &[
            ("switch_hw", false, true),
            ("throttle", false, true),
            ("kv.evict_sliding", true, false),
        ],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.7, 0.0, 0.5, -0.2));
    predictions.insert(ActionId::Throttle, rv(0.4, 0.0, 0.2, -0.5));
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.1));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.6, 0.0, 0.4);
    let state = no_state();
    let qcf = HashMap::new();

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Critical,
        &state,
        &qcf,
        1.0,
        &[],
        &[],
    );

    let ids = command_ids(&cmds);
    assert!(
        ids.contains(&ActionId::SwitchHw),
        "switch_hw should be selected for cross-domain relief"
    );
    assert!(
        !ids.contains(&ActionId::KvEvictSliding),
        "evict not needed when no memory pressure"
    );
}

// ── MGR-ALG-035: 파라미터 보간 ──────────────────────────────────────

/// MGR-ALG-035: pressure 크기에 따라 파라미터가 선형 보간된다.
#[test]
fn test_mgr_alg_035_parametrize_proportional() {
    let registry = make_registry(&[("kv.evict_sliding", true, false)], &[]);

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.0, 0.5, 0.0);
    let state = no_state();
    let mut qcf = HashMap::new();
    qcf.insert(ActionId::KvEvictSliding, 1.0f32);

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Critical,
        &state,
        &qcf,
        1.0,
        &[],
        &[],
    );

    assert_eq!(cmds.len(), 1);
    let cmd = &cmds[0];
    assert_eq!(cmd.action, ActionId::KvEvictSliding);

    if let Operation::Apply(params) = &cmd.operation {
        let keep_ratio = params.values.get("keep_ratio").copied().unwrap();
        // range = [0.3, 0.9], intensity=0.5 -> value = 0.9 - 0.5*(0.9-0.3) = 0.6
        let expected = 0.9 - 0.5 * (0.9 - 0.3);
        assert!(
            (keep_ratio - expected).abs() < 1e-5,
            "keep_ratio should be {expected}, got {keep_ratio}"
        );
    } else {
        panic!("expected Apply operation");
    }
}
