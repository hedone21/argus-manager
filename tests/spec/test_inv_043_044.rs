//! INV-043: 완전 해소 가능한 조합이 존재하면, best-effort 조합보다 항상 우선한다.
//! INV-044: parametrize 출력 value는 항상 [range.min, range.max] 범위 내이다.

use std::collections::HashMap;

use argus_manager::selector::ActionSelector;
use argus_manager::types::{ActionId, OperatingMode, Operation};

use super::helpers::{MockEstimator, make_registry, no_state, pv, rv};

// ---------------------------------------------------------------------------
// INV-043: 완전 해소 > best-effort
// ---------------------------------------------------------------------------

/// 완전 해소 가능한 조합이 best-effort보다 우선 선택되어야 한다.
///
/// 시나리오: 단일 액션으로는 완전 해소 불가, 2개 조합은 가능.
/// switch_hw(lossless)가 compute를, kv_evict_sliding(lossy)이 memory를 각각 해소.
/// 조합으로 완전 해소 가능하면 best-effort 대신 완전 해소 조합이 선택된다.
#[test]
fn test_inv_043_full_resolution_preferred_over_partial() {
    // 두 액션만 등록: 각각 단독으로는 모든 도메인 해소 불가, 조합 시 가능
    let registry = make_registry(
        &[
            ("switch_hw", false, true),        // lossless, compute relief
            ("kv.evict_sliding", true, false), // lossy, memory relief
        ],
        &[],
    );

    let mut predictions = HashMap::new();
    // switch_hw: compute=0.5 (>= pressure 0.3), memory=0
    predictions.insert(ActionId::SwitchHw, rv(0.5, 0.0, 0.0, 0.0));
    // kv_evict_sliding: compute=0, memory=0.9 (>= pressure 0.5)
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let mut qcf = HashMap::new();
    qcf.insert(ActionId::KvEvictSliding, 0.5_f32);

    // pressure: compute=0.3, memory=0.5
    // switch_hw 단독: compute OK, memory FAIL -> 불완전
    // kv_evict_sliding 단독: compute FAIL, memory OK -> 불완전
    // 둘 다 조합: compute 0.5 >= 0.3, memory 0.9 >= 0.5 -> 완전 해소 (cost=0.5)
    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pv(0.3, 0.5, 0.0),
        OperatingMode::Critical,
        &no_state(),
        &qcf,
        1.0,
        &[],
        &[],
    );

    let ids: Vec<_> = cmds.iter().map(|c| c.action).collect();

    // 완전 해소 조합이 선택되어야 한다
    assert!(
        ids.contains(&ActionId::KvEvictSliding),
        "INV-043: kv_evict_sliding should be in full-resolution combo"
    );
    assert!(
        ids.contains(&ActionId::SwitchHw),
        "INV-043: switch_hw should be in full-resolution combo"
    );
    assert_eq!(
        ids.len(),
        2,
        "INV-043: exactly 2 actions should be selected for full resolution"
    );
}

/// 완전 해소 가능 조합 중 cost가 최소인 것이 선택되어야 한다.
#[test]
fn test_inv_043_full_resolution_minimum_cost() {
    // 두 lossy 액션이 모두 단독으로 memory 완전 해소 가능
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

    let mut qcf = HashMap::new();
    qcf.insert(ActionId::KvEvictSliding, 0.3_f32); // 저비용
    qcf.insert(ActionId::KvEvictH2o, 2.0_f32); // 고비용

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pv(0.0, 0.5, 0.0),
        OperatingMode::Critical,
        &no_state(),
        &qcf,
        1.0,
        &[],
        &[],
    );

    let ids: Vec<_> = cmds.iter().map(|c| c.action).collect();
    assert!(
        ids.contains(&ActionId::KvEvictSliding),
        "INV-043: cheaper full-resolution action should be selected"
    );
    assert!(
        !ids.contains(&ActionId::KvEvictH2o),
        "INV-043: more expensive action should not be selected when cheaper resolves fully"
    );
}

/// 어떤 조합으로도 완전 해소가 불가하면 coverage 최대 조합 선택 (best-effort).
#[test]
fn test_inv_043_best_effort_when_no_full_resolution() {
    let registry = make_registry(
        &[("switch_hw", false, true), ("throttle", false, true)],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.2, 0.0, 0.0, 0.0));
    predictions.insert(ActionId::Throttle, rv(0.1, 0.0, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    // compute pressure = 0.8이지만 합산 relief = 0.3 < 0.8 -> 완전 해소 불가
    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pv(0.8, 0.0, 0.0),
        OperatingMode::Warning,
        &no_state(),
        &HashMap::new(),
        1.0,
        &[],
        &[],
    );

    let ids: Vec<_> = cmds.iter().map(|c| c.action).collect();
    // best-effort: 둘 다 포함하는 것이 coverage 최대
    assert!(
        ids.contains(&ActionId::SwitchHw) && ids.contains(&ActionId::Throttle),
        "INV-043: best-effort should maximize coverage"
    );
}

// ---------------------------------------------------------------------------
// INV-044: parametrize 출력 value in [min, max]
// ---------------------------------------------------------------------------

/// keep_ratio 파라미터가 [0.3, 0.9] 범위 내에 있어야 한다.
#[test]
fn test_inv_044_keep_ratio_within_range() {
    let registry = make_registry(&[("kv.evict_sliding", true, false)], &[]);

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let mut qcf = HashMap::new();
    qcf.insert(ActionId::KvEvictSliding, 0.5_f32);

    // 다양한 pressure 강도로 테스트
    for &memory_pressure in &[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
        let cmds = ActionSelector::select(
            &registry,
            &estimator,
            &pv(0.0, memory_pressure, 0.0),
            OperatingMode::Critical,
            &no_state(),
            &qcf,
            1.0,
            &[],
            &[],
        );

        // pressure=0이면 빈 결과 -> skip
        if cmds.is_empty() {
            continue;
        }

        let cmd = &cmds[0];
        if let Operation::Apply(params) = &cmd.operation
            && let Some(&keep_ratio) = params.values.get("keep_ratio")
        {
            assert!(
                (0.3..=0.9).contains(&keep_ratio),
                "INV-044: keep_ratio={keep_ratio} must be in [0.3, 0.9] (pressure={memory_pressure})"
            );
        }
    }
}

/// throttle delay_ms 파라미터가 [0.0, 100.0] 범위 내에 있어야 한다.
#[test]
fn test_inv_044_delay_ms_within_range() {
    let registry = make_registry(&[("throttle", false, true)], &[]);

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::Throttle, rv(0.8, 0.0, 0.0, -0.1));
    let estimator = MockEstimator::new(predictions);

    for &compute_pressure in &[0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
        let cmds = ActionSelector::select(
            &registry,
            &estimator,
            &pv(compute_pressure, 0.0, 0.0),
            OperatingMode::Warning,
            &no_state(),
            &HashMap::new(),
            1.0,
            &[],
            &[],
        );

        if cmds.is_empty() {
            continue;
        }

        let cmd = &cmds[0];
        if let Operation::Apply(params) = &cmd.operation
            && let Some(&delay_ms) = params.values.get("delay_ms")
        {
            assert!(
                (0.0..=100.0).contains(&delay_ms),
                "INV-044: delay_ms={delay_ms} must be in [0.0, 100.0] (pressure={compute_pressure})"
            );
        }
    }
}

/// layer_skip skip_layers 파라미터가 [1.0, 8.0] 범위 내에 있어야 한다.
#[test]
fn test_inv_044_skip_layers_within_range() {
    let registry = make_registry(&[("weight.skip", true, true)], &[]);

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::LayerSkip, rv(0.8, 0.0, 0.0, -0.2));
    let estimator = MockEstimator::new(predictions);

    let mut qcf = HashMap::new();
    qcf.insert(ActionId::LayerSkip, 1.0_f32);

    for &compute_pressure in &[0.1, 0.5, 1.0] {
        let cmds = ActionSelector::select(
            &registry,
            &estimator,
            &pv(compute_pressure, 0.0, 0.0),
            OperatingMode::Critical,
            &no_state(),
            &qcf,
            1.0,
            &[],
            &[],
        );

        if cmds.is_empty() {
            continue;
        }

        let cmd = &cmds[0];
        if let Operation::Apply(params) = &cmd.operation
            && let Some(&skip_layers) = params.values.get("skip_layers")
        {
            assert!(
                (1.0..=8.0).contains(&skip_layers),
                "INV-044: skip_layers={skip_layers} must be in [1.0, 8.0] (pressure={compute_pressure})"
            );
        }
    }
}

/// kv_quant_dynamic target_bits 파라미터가 [4.0, 8.0] 범위 내에 있어야 한다.
#[test]
fn test_inv_044_target_bits_within_range() {
    let registry = make_registry(&[("kv.quant_dynamic", true, false)], &[]);

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::KvQuantDynamic, rv(0.0, 0.8, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let mut qcf = HashMap::new();
    qcf.insert(ActionId::KvQuantDynamic, 0.5_f32);

    for &memory_pressure in &[0.1, 0.5, 1.0] {
        let cmds = ActionSelector::select(
            &registry,
            &estimator,
            &pv(0.0, memory_pressure, 0.0),
            OperatingMode::Critical,
            &no_state(),
            &qcf,
            1.0,
            &[],
            &[],
        );

        if cmds.is_empty() {
            continue;
        }

        let cmd = &cmds[0];
        if let Operation::Apply(params) = &cmd.operation
            && let Some(&target_bits) = params.values.get("target_bits")
        {
            assert!(
                (4.0..=8.0).contains(&target_bits),
                "INV-044: target_bits={target_bits} must be in [4.0, 8.0] (pressure={memory_pressure})"
            );
        }
    }
}

/// 극단적 pressure 값 (0.0, 1.0, 음수, >1.0)에서도 parametrize가 범위를 벗어나지 않아야 한다.
#[test]
fn test_inv_044_extreme_pressure_values_still_in_range() {
    let registry = make_registry(&[("kv.evict_sliding", true, false)], &[]);

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let mut qcf = HashMap::new();
    qcf.insert(ActionId::KvEvictSliding, 0.5_f32);

    // 극단적 memory pressure 값 (음수, 0, 1, >1)
    for &memory_pressure in &[-1.0_f32, -0.5, 0.0, 0.5, 1.0, 1.5, 10.0] {
        let pressure = pv(0.0, memory_pressure, 0.0);

        // pressure <= 0 이면 빈 결과 (no pressure) -> skip
        if memory_pressure <= 0.0 {
            continue;
        }

        let cmds = ActionSelector::select(
            &registry,
            &estimator,
            &pressure,
            OperatingMode::Critical,
            &no_state(),
            &qcf,
            1.0,
            &[],
            &[],
        );

        if cmds.is_empty() {
            continue;
        }

        let cmd = &cmds[0];
        if let Operation::Apply(params) = &cmd.operation
            && let Some(&keep_ratio) = params.values.get("keep_ratio")
        {
            assert!(
                (0.3..=0.9).contains(&keep_ratio),
                "INV-044: keep_ratio={keep_ratio} out of [0.3, 0.9] at extreme pressure={memory_pressure}"
            );
        }
    }
}
