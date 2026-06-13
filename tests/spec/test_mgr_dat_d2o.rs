//! MGR-DAT KvMergeD2o spec tests.
//!
//! ActionId::KvMergeD2o 등록, 배타 그룹, primary_domain,
//! serde round-trip, param_range, action_to_engine_command 검증.

use std::collections::HashMap;

use argus_manager::action_registry::ActionRegistry;
use argus_manager::config::{ActionConfig, PolicyConfig};
use argus_manager::types::{ActionId, ActionKind, Domain};

// ── ActionId::KvMergeD2o 기본 속성 ──

#[test]
fn test_kv_merge_d2o_from_str() {
    let id = ActionId::from_str("kv.merge_d2o");
    assert_eq!(id, Some(ActionId::KvMergeD2o));
}

#[test]
fn test_kv_merge_d2o_in_all() {
    let all = ActionId::all();
    assert!(
        all.contains(&ActionId::KvMergeD2o),
        "ActionId::all() must include KvMergeD2o"
    );
}

#[test]
fn test_kv_merge_d2o_primary_domain_is_memory() {
    assert_eq!(ActionId::KvMergeD2o.primary_domain(), Domain::Memory);
}

#[test]
fn test_kv_merge_d2o_serde_roundtrip() {
    let id = ActionId::KvMergeD2o;
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, r#""kv.merge_d2o""#);
    let back: ActionId = serde_json::from_str(&json).unwrap();
    assert_eq!(back, id);
}

// ── ActionRegistry: KvMergeD2o 등록 ──

#[test]
fn test_registry_kv_merge_d2o_registration() {
    let mut actions = HashMap::new();
    actions.insert(
        "kv.merge_d2o".to_string(),
        ActionConfig {
            lossy: true,
            reversible: false,
            ..Default::default()
        },
    );
    let config = PolicyConfig {
        actions,
        ..Default::default()
    };
    let registry = ActionRegistry::from_config(&config);

    let meta = registry
        .get(&ActionId::KvMergeD2o)
        .expect("KvMergeD2o must be registered");
    assert_eq!(meta.kind, ActionKind::Lossy);
    assert!(!meta.reversible);
}

#[test]
fn test_registry_kv_merge_d2o_param_range() {
    let mut actions = HashMap::new();
    actions.insert(
        "kv.merge_d2o".to_string(),
        ActionConfig {
            lossy: true,
            reversible: false,
            ..Default::default()
        },
    );
    let config = PolicyConfig {
        actions,
        ..Default::default()
    };
    let registry = ActionRegistry::from_config(&config);

    let meta = registry.get(&ActionId::KvMergeD2o).unwrap();
    let range = meta.param_range.as_ref().expect("should have param_range");
    assert_eq!(range.param_name, "keep_ratio");
    assert!((range.min - 0.3).abs() < f32::EPSILON);
    assert!((range.max - 0.9).abs() < f32::EPSILON);
}

// ── Exclusion group: eviction ──

#[test]
fn test_kv_merge_d2o_exclusion_group() {
    let mut actions = HashMap::new();
    for name in &[
        "kv.evict_sliding",
        "kv.evict_h2o",
        "kv.evict_streaming",
        "kv.merge_d2o",
    ] {
        actions.insert(
            name.to_string(),
            ActionConfig {
                lossy: true,
                reversible: false,
                ..Default::default()
            },
        );
    }
    let mut exclusion_groups = HashMap::new();
    exclusion_groups.insert(
        "eviction".to_string(),
        vec![
            "kv.evict_sliding".to_string(),
            "kv.evict_h2o".to_string(),
            "kv.evict_streaming".to_string(),
            "kv.merge_d2o".to_string(),
        ],
    );
    let config = PolicyConfig {
        actions,
        exclusion_groups,
        ..Default::default()
    };
    let registry = ActionRegistry::from_config(&config);

    assert!(registry.is_excluded(&ActionId::KvMergeD2o, &ActionId::KvEvictSliding));
    assert!(registry.is_excluded(&ActionId::KvMergeD2o, &ActionId::KvEvictH2o));
    assert!(registry.is_excluded(&ActionId::KvMergeD2o, &ActionId::KvEvictStreaming));
    assert!(registry.is_excluded(&ActionId::KvEvictH2o, &ActionId::KvMergeD2o));
}
