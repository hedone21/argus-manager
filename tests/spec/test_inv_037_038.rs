//! INV-037: Warning 모드에서 Lossy 액션 선택 금지.
//! INV-038: 이미 활성 중인 액션은 재선택 금지.
//!
//! 검증 대상: `ActionSelector::select()` → `filter_candidates()`.
//! 원본: spec/22-manager-algorithms.md ALG-031.

use std::collections::HashMap;

use argus_manager::selector::ActionSelector;
use argus_manager::types::{ActionId, OperatingMode};

use super::helpers::{MockEstimator, make_registry, no_state, pv, rv};

// ---------------------------------------------------------------------------
// INV-037: Warning 모드에서 Lossy 액션 선택 금지
// ---------------------------------------------------------------------------

/// INV-037 기본: Warning 모드에서 Lossy 액션이 결과에 포함되지 않아야 한다.
#[test]
fn inv037_warning_mode_excludes_all_lossy_actions() {
    // switch_hw: lossless, kv_evict_sliding: lossy, layer_skip: lossy
    let registry = make_registry(
        &[
            ("switch_hw", false, true),
            ("kv.evict_sliding", true, false),
            ("weight.skip", true, true),
        ],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.5, 0.0, 0.3, 0.0));
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
    predictions.insert(ActionId::LayerSkip, rv(0.4, 0.0, 0.0, -0.1));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.5, 0.5, 0.0);
    let state = no_state();
    let mut qcf = HashMap::new();
    qcf.insert(ActionId::KvEvictSliding, 0.5);
    qcf.insert(ActionId::LayerSkip, 0.8);

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

    let ids: Vec<_> = cmds.iter().map(|c| c.action).collect();
    assert!(
        !ids.contains(&ActionId::KvEvictSliding),
        "INV-037: Lossy KvEvictSliding must not appear in Warning mode"
    );
    assert!(
        !ids.contains(&ActionId::LayerSkip),
        "INV-037: Lossy LayerSkip must not appear in Warning mode"
    );
}

/// INV-037 경계: 모든 액션이 Lossy이면 Warning 모드에서 빈 결과를 반환해야 한다.
#[test]
fn inv037_warning_mode_all_lossy_returns_empty() {
    let registry = make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
            ("weight.skip", true, true),
        ],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
    predictions.insert(ActionId::KvEvictH2o, rv(0.0, 0.8, 0.0, 0.0));
    predictions.insert(ActionId::LayerSkip, rv(0.4, 0.0, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.5, 0.7, 0.0);
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
        "INV-037: All lossy in Warning mode must yield empty result"
    );
}

/// INV-037 대조: Critical 모드에서는 Lossy 액션이 선택 가능해야 한다.
#[test]
fn inv037_critical_mode_allows_lossy() {
    let registry = make_registry(
        &[
            ("switch_hw", false, true),
            ("kv.evict_sliding", true, false),
        ],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.5, 0.0, 0.3, 0.0));
    predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.0, 0.7, 0.0);
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
    assert!(
        ids.contains(&ActionId::KvEvictSliding),
        "INV-037 contrast: Lossy should be selectable in Critical mode"
    );
}

// ---------------------------------------------------------------------------
// INV-038: 이미 활성 중인 액션은 재선택 금지
// ---------------------------------------------------------------------------

/// INV-038 기본: active_actions에 포함된 액션은 결과에서 제외되어야 한다.
#[test]
fn inv038_active_action_excluded_from_selection() {
    let registry = make_registry(
        &[("switch_hw", false, true), ("throttle", false, true)],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, 0.0));
    predictions.insert(ActionId::Throttle, rv(0.4, 0.0, 0.2, -0.2));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.6, 0.0, 0.0);
    let state = no_state();
    let qcf = HashMap::new();
    // SwitchHw가 이미 활성 중
    let active = [ActionId::SwitchHw];

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Warning,
        &state,
        &qcf,
        1.0,
        &active,
        &[],
    );

    let ids: Vec<_> = cmds.iter().map(|c| c.action).collect();
    assert!(
        !ids.contains(&ActionId::SwitchHw),
        "INV-038: Active action SwitchHw must not be re-selected"
    );
    // Throttle은 활성이 아니므로 선택 가능
    assert!(
        ids.contains(&ActionId::Throttle),
        "INV-038: Non-active Throttle should be selectable"
    );
}

/// INV-038: 모든 등록 액션이 활성 중이면 빈 결과를 반환해야 한다.
#[test]
fn inv038_all_active_returns_empty() {
    let registry = make_registry(
        &[("switch_hw", false, true), ("throttle", false, true)],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, 0.0));
    predictions.insert(ActionId::Throttle, rv(0.4, 0.0, 0.2, -0.2));
    let estimator = MockEstimator::new(predictions);

    let pressure = pv(0.6, 0.0, 0.0);
    let state = no_state();
    let qcf = HashMap::new();
    let active = [ActionId::SwitchHw, ActionId::Throttle];

    let cmds = ActionSelector::select(
        &registry,
        &estimator,
        &pressure,
        OperatingMode::Warning,
        &state,
        &qcf,
        1.0,
        &active,
        &[],
    );

    assert!(
        cmds.is_empty(),
        "INV-038: All active actions should yield empty result"
    );
}

/// INV-038: active_actions가 비어있으면 모든 후보가 선택 가능.
#[test]
fn inv038_empty_active_allows_all() {
    let registry = make_registry(
        &[("switch_hw", false, true), ("throttle", false, true)],
        &[],
    );

    let mut predictions = HashMap::new();
    predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, 0.0));
    predictions.insert(ActionId::Throttle, rv(0.4, 0.0, 0.2, -0.2));
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
        &[], // no active
        &[],
    );

    assert!(
        !cmds.is_empty(),
        "INV-038: Empty active_actions should allow selection"
    );
}
