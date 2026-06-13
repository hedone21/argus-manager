//! MGR-DAT-054: SwapWeights 메타데이터 — ActionRegistry + OnlineLinearEstimator 검증
//!
//! default_param_range, ActionRegistry 등록(lossy/reversible/cost),
//! lossy_actions 포함 여부, cold-start prior relief 일치.

use std::collections::HashMap;

use argus_manager::action_registry::ActionRegistry;
use argus_manager::config::{ActionConfig, PolicyConfig};
use argus_manager::relief::ReliefEstimator;
use argus_manager::relief::linear::OnlineLinearEstimator;
use argus_manager::types::{ActionId, ActionKind, FeatureVector};

// ── default_param_range ──

/// default_param_range(SwapWeights) → Some { param_name: "ratio", min: 0.0, max: 0.9 }.
#[test]
fn test_mgr_dat_054_default_param_range() {
    let mut actions = HashMap::new();
    actions.insert(
        "swap_weights".to_string(),
        ActionConfig {
            lossy: true,
            reversible: false,
            default_cost: 1.0,
        },
    );
    let config = PolicyConfig {
        actions,
        ..Default::default()
    };
    let registry = ActionRegistry::from_config(&config);

    let meta = registry
        .get(&ActionId::SwapWeights)
        .expect("SwapWeights must be registered");
    let range = meta
        .param_range
        .as_ref()
        .expect("SwapWeights must have param_range");
    assert_eq!(range.param_name, "ratio");
    assert!(
        (range.min - 0.0).abs() < f32::EPSILON,
        "min must be 0.0, got {}",
        range.min
    );
    assert!(
        (range.max - 0.9).abs() < f32::EPSILON,
        "max must be 0.9, got {}",
        range.max
    );
}

// ── ActionRegistry 등록 ──

/// [policy.actions.swap_weights] lossy=true, reversible=false, default_cost=1.0 등록.
#[test]
fn test_mgr_dat_054_registry_registration() {
    let mut actions = HashMap::new();
    actions.insert(
        "swap_weights".to_string(),
        ActionConfig {
            lossy: true,
            reversible: false,
            default_cost: 1.0,
        },
    );
    let config = PolicyConfig {
        actions,
        ..Default::default()
    };
    let registry = ActionRegistry::from_config(&config);

    let meta = registry
        .get(&ActionId::SwapWeights)
        .expect("SwapWeights must be in registry");
    assert_eq!(
        meta.kind,
        ActionKind::Lossy,
        "SwapWeights kind must be Lossy"
    );
    assert!(!meta.reversible, "SwapWeights must not be reversible");
    assert!(
        (meta.default_cost - 1.0).abs() < f32::EPSILON,
        "SwapWeights default_cost must be 1.0, got {}",
        meta.default_cost
    );
    assert!(
        meta.exclusion_group.is_none(),
        "SwapWeights must have no exclusion_group by default"
    );
}

/// lossy_actions()에 SwapWeights가 포함되어야 한다.
#[test]
fn test_mgr_dat_054_lossy_actions_contains_swap_weights() {
    let mut actions = HashMap::new();
    actions.insert(
        "swap_weights".to_string(),
        ActionConfig {
            lossy: true,
            reversible: false,
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
    let config = PolicyConfig {
        actions,
        ..Default::default()
    };
    let registry = ActionRegistry::from_config(&config);

    let lossy = registry.lossy_actions();
    assert!(
        lossy.contains(&ActionId::SwapWeights),
        "lossy_actions() must include SwapWeights"
    );
    assert!(
        !lossy.contains(&ActionId::SwitchHw),
        "SwitchHw must not be in lossy_actions()"
    );
}

// ── OnlineLinearEstimator cold-start prior ──

/// 관측 0건 cold-start 시 predict(SwapWeights) → prior (0.0, 0.5, 0.0, -0.2).
#[test]
fn test_mgr_dat_054_cold_start_prior_relief() {
    let estimator = OnlineLinearEstimator::default_config();
    let state = FeatureVector::zeros();

    let pred = estimator.predict(&ActionId::SwapWeights, &state);

    assert!(
        pred.compute.abs() < f32::EPSILON,
        "cold-start SwapWeights compute must be 0.0, got {}",
        pred.compute
    );
    assert!(
        (pred.memory - 0.5).abs() < f32::EPSILON,
        "cold-start SwapWeights memory must be 0.5, got {}",
        pred.memory
    );
    assert!(
        pred.thermal.abs() < f32::EPSILON,
        "cold-start SwapWeights thermal must be 0.0, got {}",
        pred.thermal
    );
    assert!(
        (pred.latency - (-0.2)).abs() < f32::EPSILON,
        "cold-start SwapWeights latency must be -0.2, got {}",
        pred.latency
    );
}
