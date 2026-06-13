//! INV-041: 동일 배타 그룹 액션은 하나의 조합에 동시 미포함.
//! INV-042: 조합의 총 latency 악화 > latency_budget이면 배제.
//!
//! 검증 대상: `ActionSelector::select()` → `find_optimal()`.
//! 원본: spec/22-manager-algorithms.md ALG-033.

use std::collections::HashMap;

use argus_manager::selector::ActionSelector;
use argus_manager::types::{ActionId, OperatingMode};

use super::helpers::{MockEstimator, make_registry, no_state, pv, rv};

// ---------------------------------------------------------------------------
// INV-041: 동일 배타 그룹 액션은 하나의 조합에 동시 미포함
// ---------------------------------------------------------------------------

/// INV-041 기본: 같은 exclusion group의 두 액션이 동시에 선택되지 않아야 한다.
#[test]
fn inv041_exclusion_group_prevents_simultaneous_selection() {
    let registry = make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
        ],
        &[("eviction", &["kv.evict_sliding", "kv.evict_h2o"])],
    );

    let mut predictions = HashMap::new();
    // 둘 다 memory pressure를 충분히 해소
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
    predictions.insert(ActionId::KvEvictH2o, rv(0.0, 0.9, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.0, 0.8, 0.0);
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

    let ids: Vec<_> = cmds.iter().map(|c| c.action).collect();
    let both = ids.contains(&ActionId::KvEvictSliding) && ids.contains(&ActionId::KvEvictH2o);
    assert!(
        !both,
        "INV-041: Exclusive group actions must not appear together, got: {:?}",
        ids
    );
    // 정확히 하나만 선택되어야 한다
    let count = ids
        .iter()
        .filter(|&&id| id == ActionId::KvEvictSliding || id == ActionId::KvEvictH2o)
        .count();
    assert_eq!(
        count, 1,
        "INV-041: Exactly one from exclusion group should be selected"
    );
}

/// INV-041: 배타 그룹이 3개 이상의 멤버를 포함할 때도 최대 1개만 선택.
#[test]
fn inv041_exclusion_group_three_members() {
    let registry = make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
            ("kv.quant_dynamic", true, false),
        ],
        &[(
            "memory_ops",
            &["kv.evict_sliding", "kv.evict_h2o", "kv.quant_dynamic"],
        )],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.5, 0.0, 0.0));
    predictions.insert(ActionId::KvEvictH2o, rv(0.0, 0.5, 0.0, 0.0));
    predictions.insert(ActionId::KvQuantDynamic, rv(0.0, 0.5, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.0, 0.8, 0.0);
    let state = no_state();
    let mut qcf = HashMap::new();
    qcf.insert(ActionId::KvEvictSliding, 1.0f32);
    qcf.insert(ActionId::KvEvictH2o, 1.0f32);
    qcf.insert(ActionId::KvQuantDynamic, 1.0f32);

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

    let ids: Vec<_> = cmds.iter().map(|c| c.action).collect();
    let group_count = ids
        .iter()
        .filter(|&&id| {
            id == ActionId::KvEvictSliding
                || id == ActionId::KvEvictH2o
                || id == ActionId::KvQuantDynamic
        })
        .count();
    assert!(
        group_count <= 1,
        "INV-041: At most 1 from 3-member exclusion group, got {}",
        group_count
    );
}

/// INV-041: 서로 다른 배타 그룹의 액션은 동시 선택 가능.
#[test]
fn inv041_different_groups_can_coexist() {
    let registry = make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
            ("switch_hw", false, true),
            ("throttle", false, true),
        ],
        &[
            ("eviction", &["kv.evict_sliding", "kv.evict_h2o"]),
            ("hw_change", &["switch_hw", "throttle"]),
        ],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
    predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, 0.0));
    let estimator = MockEstimator::new(predictions);

    // compute + memory pressure
    let pressure = pv(0.6, 0.7, 0.0);
    let state = no_state();
    let mut qcf = HashMap::new();
    qcf.insert(ActionId::KvEvictSliding, 0.5);

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

    let ids: Vec<_> = cmds.iter().map(|c| c.action).collect();
    // 서로 다른 그룹이므로 둘 다 선택 가능
    let has_evict = ids.contains(&ActionId::KvEvictSliding);
    let has_hw = ids.contains(&ActionId::SwitchHw);
    assert!(
        has_evict && has_hw,
        "INV-041: Actions from different exclusion groups can coexist, got: {:?}",
        ids
    );
}

// ---------------------------------------------------------------------------
// INV-042: 조합의 총 latency 악화 > latency_budget이면 배제
// ---------------------------------------------------------------------------

/// INV-042 기본: 단일 액션의 latency 악화가 budget을 초과하면 배제.
#[test]
fn inv042_single_action_exceeds_latency_budget() {
    let registry = make_registry(&[("throttle", false, true)], &[]);

    let mut predictions = HashMap::new();
    // latency = -0.8 (악화), budget = 0.5 → -0.8 < -0.5이므로 배제
    predictions.insert(ActionId::Throttle, rv(0.5, 0.0, 0.3, -0.8));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.4, 0.0, 0.0);
    let state = no_state();
    let qcf = HashMap::new();

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Warning,
        &state,
        &qcf,
        0.5, // latency_budget = 0.5
        &[],
        &[],
    );

    let ids: Vec<_> = cmds.iter().map(|c| c.action).collect();
    assert!(
        !ids.contains(&ActionId::Throttle),
        "INV-042: Action with latency -0.8 exceeds budget 0.5, should be excluded"
    );
}

/// INV-042: 개별 액션은 budget 이내지만 조합의 총 latency가 초과하면 배제.
#[test]
fn inv042_combined_latency_exceeds_budget() {
    let registry = make_registry(
        &[("throttle", false, true), ("kv_offload_disk", false, true)],
        &[],
    );

    let mut predictions = HashMap::new();
    // 각각 -0.4 → 합산 -0.8
    predictions.insert(ActionId::Throttle, rv(0.5, 0.0, 0.3, -0.4));
    predictions.insert(ActionId::KvOffloadDisk, rv(0.0, 0.8, 0.0, -0.4));
    let estimator = MockEstimator::new(predictions);

    // 두 도메인 모두 pressure → 둘 다 필요한 상황
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
        0.6, // budget = 0.6, 합산 -0.8 > -0.6
        &[],
        &[],
    );

    let ids: Vec<_> = cmds.iter().map(|c| c.action).collect();
    let both = ids.contains(&ActionId::Throttle) && ids.contains(&ActionId::KvOffloadDisk);
    assert!(
        !both,
        "INV-042: Combined latency -0.8 exceeds budget 0.6, pair should be excluded"
    );
}

/// INV-042: latency_budget = 0.0이면 latency 악화가 있는 모든 조합이 배제.
#[test]
fn inv042_zero_budget_excludes_any_latency_degradation() {
    let registry = make_registry(
        &[("switch_hw", false, true), ("throttle", false, true)],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.6, 0.0, 0.3, -0.01));
    predictions.insert(ActionId::Throttle, rv(0.3, 0.0, 0.2, -0.01));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.5, 0.0, 0.0);
    let state = no_state();
    let qcf = HashMap::new();

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Warning,
        &state,
        &qcf,
        0.0, // zero budget
        &[],
        &[],
    );

    // latency_budget=0이므로 latency < 0인 액션은 모두 배제
    assert!(
        cmds.is_empty(),
        "INV-042: Zero latency budget should exclude actions with any latency degradation"
    );
}

/// INV-042 대조: latency가 budget 이내인 조합은 선택 가능.
#[test]
fn inv042_within_budget_allowed() {
    let registry = make_registry(
        &[("switch_hw", false, true), ("throttle", false, true)],
        &[],
    );

    let mut predictions = HashMap::new();
    // 각각 -0.2 → 합산 -0.4, budget = 0.5 → 이내
    predictions.insert(ActionId::SwitchHw, rv(0.6, 0.0, 0.3, -0.2));
    predictions.insert(ActionId::Throttle, rv(0.3, 0.0, 0.2, -0.2));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.5, 0.0, 0.0);
    let state = no_state();
    let qcf = HashMap::new();

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Warning,
        &state,
        &qcf,
        0.5, // budget = 0.5, 합산 = -0.4 이내
        &[],
        &[],
    );

    assert!(
        !cmds.is_empty(),
        "INV-042: Combination within latency budget should be selectable"
    );
}
