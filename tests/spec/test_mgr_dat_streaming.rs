//! MGR-DAT KvEvictStreaming spec tests.
//!
//! ActionId::KvEvictStreaming 등록, 배타 그룹, primary_domain,
//! serde round-trip 검증.

use std::collections::HashMap;

use argus_manager::action_registry::ActionRegistry;
use argus_manager::config::{ActionConfig, PolicyConfig};
use argus_manager::types::{ActionId, ActionKind, Domain};

// ── ActionId::KvEvictStreaming 기본 속성 ──

#[test]
fn test_kv_evict_streaming_from_str() {
    let id = ActionId::from_str("kv.evict_streaming");
    assert_eq!(id, Some(ActionId::KvEvictStreaming));
}

#[test]
fn test_kv_evict_streaming_in_all() {
    let all = ActionId::all();
    assert!(
        all.contains(&ActionId::KvEvictStreaming),
        "ActionId::all() must include KvEvictStreaming"
    );
}

#[test]
fn test_kv_evict_streaming_primary_domain_is_memory() {
    assert_eq!(ActionId::KvEvictStreaming.primary_domain(), Domain::Memory);
}

#[test]
fn test_kv_evict_streaming_serde_roundtrip() {
    let id = ActionId::KvEvictStreaming;
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, r#""kv.evict_streaming""#);
    let back: ActionId = serde_json::from_str(&json).unwrap();
    assert_eq!(back, id);
}

// ── ActionRegistry: KvEvictStreaming 등록 ──

#[test]
fn test_registry_kv_evict_streaming_registration() {
    let mut actions = HashMap::new();
    actions.insert(
        "kv.evict_streaming".to_string(),
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
        .get(&ActionId::KvEvictStreaming)
        .expect("KvEvictStreaming must be registered");
    assert_eq!(meta.kind, ActionKind::Lossy);
    assert!(!meta.reversible);
}

#[test]
fn test_registry_kv_evict_streaming_param_range() {
    let mut actions = HashMap::new();
    actions.insert(
        "kv.evict_streaming".to_string(),
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

    let meta = registry.get(&ActionId::KvEvictStreaming).unwrap();
    let range = meta.param_range.as_ref().expect("should have param_range");
    assert_eq!(range.param_name, "window_size");
}

// ── Exclusion group: eviction ──

#[test]
fn test_kv_evict_streaming_exclusion_group() {
    let mut actions = HashMap::new();
    for name in &["kv.evict_sliding", "kv.evict_h2o", "kv.evict_streaming"] {
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
        ],
    );
    let config = PolicyConfig {
        actions,
        exclusion_groups,
        ..Default::default()
    };
    let registry = ActionRegistry::from_config(&config);

    assert!(registry.is_excluded(&ActionId::KvEvictStreaming, &ActionId::KvEvictSliding));
    assert!(registry.is_excluded(&ActionId::KvEvictStreaming, &ActionId::KvEvictH2o));
    assert!(registry.is_excluded(&ActionId::KvEvictH2o, &ActionId::KvEvictStreaming));
}
