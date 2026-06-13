//! INV-016 확장: SwapWeights가 KvEvictSliding과 같은 배타 그룹에 속하지 않음.
//!
//! SwapWeights는 weight 스왑 액션이고 KvEvictSliding은 KV 캐시 eviction 액션이다.
//! 두 액션은 서로 다른 자원 도메인에서 동작하므로 배타 그룹 공유가 없어야 한다.

use std::collections::HashMap;

use argus_manager::action_registry::ActionRegistry;
use argus_manager::config::{ActionConfig, PolicyConfig};
use argus_manager::types::ActionId;

fn make_registry_with_swap_and_evict() -> ActionRegistry {
    let mut actions = HashMap::new();
    for (name, lossy, reversible) in &[
        ("swap_weights", true, false),
        ("kv.evict_sliding", true, false),
        ("kv.evict_h2o", true, false),
        ("throttle", false, true),
    ] {
        actions.insert(
            name.to_string(),
            ActionConfig {
                lossy: *lossy,
                reversible: *reversible,
                ..Default::default()
            },
        );
    }
    // SwapWeights는 어떤 배타 그룹에도 속하지 않음.
    // KvEvictSliding/H2o는 eviction 그룹.
    let mut exclusion_groups: HashMap<String, Vec<String>> = HashMap::new();
    exclusion_groups.insert(
        "eviction".to_string(),
        vec!["kv.evict_sliding".to_string(), "kv.evict_h2o".to_string()],
    );
    let config = PolicyConfig {
        actions,
        exclusion_groups,
        ..Default::default()
    };
    ActionRegistry::from_config(&config)
}

/// SwapWeights와 KvEvictSliding은 배타 그룹이 달라 is_excluded가 false여야 한다.
#[test]
fn test_inv_016_swap_weights_not_excluded_from_kv_evict_sliding() {
    let registry = make_registry_with_swap_and_evict();
    assert!(
        !registry.is_excluded(&ActionId::SwapWeights, &ActionId::KvEvictSliding),
        "INV-016: SwapWeights와 KvEvictSliding은 배타 그룹이 다르므로 excluded가 아니어야 한다"
    );
}

/// SwapWeights와 KvEvictH2o도 배타 그룹이 달라 is_excluded가 false여야 한다.
#[test]
fn test_inv_016_swap_weights_not_excluded_from_kv_evict_h2o() {
    let registry = make_registry_with_swap_and_evict();
    assert!(
        !registry.is_excluded(&ActionId::SwapWeights, &ActionId::KvEvictH2o),
        "INV-016: SwapWeights와 KvEvictH2o는 배타 그룹이 다르므로 excluded가 아니어야 한다"
    );
}

/// SwapWeights의 exclusion_group 메타필드가 None이어야 한다 (배타 그룹 미소속).
#[test]
fn test_inv_016_swap_weights_exclusion_group_is_none() {
    let registry = make_registry_with_swap_and_evict();
    let meta = registry
        .get(&ActionId::SwapWeights)
        .expect("SwapWeights must be registered");
    assert!(
        meta.exclusion_group.is_none(),
        "INV-016: SwapWeights는 어떤 배타 그룹에도 속하지 않아야 한다 (got {:?})",
        meta.exclusion_group
    );
}

/// SwapWeights와 Throttle도 배타 그룹 없음 — not excluded.
#[test]
fn test_inv_016_swap_weights_not_excluded_from_throttle() {
    let registry = make_registry_with_swap_and_evict();
    assert!(
        !registry.is_excluded(&ActionId::SwapWeights, &ActionId::Throttle),
        "INV-016: SwapWeights와 Throttle은 excluded가 아니어야 한다"
    );
}
