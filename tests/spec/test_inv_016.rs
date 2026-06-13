/// INV-016: 동일 배타 그룹 액션 동시 활성화 금지.
///
/// 원본: 01-architecture SYS-096
/// 검증: ActionRegistry의 is_excluded()가 같은 exclusion group 내 액션 쌍에 대해
///       true를 반환하고, ActionSelector가 조합 생성 시 배타 그룹을 강제하는지 확인.
use std::collections::HashMap;

use argus_manager::action_registry::ActionRegistry;
use argus_manager::config::{ActionConfig, PolicyConfig};
use argus_manager::types::ActionId;

use crate::helpers;

// ── ActionRegistry 수준: is_excluded ───────────────────────────

/// 같은 배타 그룹의 두 액션은 is_excluded == true여야 한다.
#[test]
fn test_inv_016_same_exclusion_group_is_excluded() {
    let registry = helpers::make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
            ("throttle", false, true),
        ],
        &[("eviction", &["kv.evict_sliding", "kv.evict_h2o"])],
    );

    // 같은 그룹 내 → excluded
    assert!(
        registry.is_excluded(&ActionId::KvEvictSliding, &ActionId::KvEvictH2o),
        "INV-016: kv_evict_sliding과 kv_evict_h2o는 같은 배타 그룹이므로 excluded"
    );
    assert!(
        registry.is_excluded(&ActionId::KvEvictH2o, &ActionId::KvEvictSliding),
        "INV-016: is_excluded는 대칭이어야 한다"
    );
}

/// 다른 그룹 또는 그룹 미소속 액션은 is_excluded == false여야 한다.
#[test]
fn test_inv_016_different_group_not_excluded() {
    let registry = helpers::make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
            ("throttle", false, true),
            ("switch_hw", false, true),
        ],
        &[("eviction", &["kv.evict_sliding", "kv.evict_h2o"])],
    );

    // 배타 그룹에 속하지 않은 액션 간 → not excluded
    assert!(
        !registry.is_excluded(&ActionId::Throttle, &ActionId::SwitchHw),
        "INV-016: throttle과 switch_hw는 배타 그룹이 아니다"
    );

    // 배타 그룹 소속 vs 미소속 → not excluded
    assert!(
        !registry.is_excluded(&ActionId::KvEvictSliding, &ActionId::Throttle),
        "INV-016: kv_evict_sliding과 throttle은 다른 그룹이다"
    );
}

/// 자기 자신과의 배타 검사: 동일 액션은 같은 그룹이므로 excluded여야 한다.
#[test]
fn test_inv_016_self_exclusion() {
    let registry = helpers::make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
        ],
        &[("eviction", &["kv.evict_sliding", "kv.evict_h2o"])],
    );

    // 자기 자신은 같은 그룹이므로 excluded
    assert!(
        registry.is_excluded(&ActionId::KvEvictSliding, &ActionId::KvEvictSliding),
        "INV-016: 동일 액션은 같은 그룹이므로 excluded"
    );
}

/// 복수 배타 그룹이 존재할 때 각 그룹 내에서만 배타가 적용되어야 한다.
#[test]
fn test_inv_016_multiple_exclusion_groups() {
    // 두 개의 배타 그룹을 설정
    let mut action_map = HashMap::new();
    for (name, lossy, reversible) in &[
        ("kv.evict_sliding", true, false),
        ("kv.evict_h2o", true, false),
        ("throttle", false, true),
        ("weight.skip", true, true),
    ] {
        action_map.insert(
            name.to_string(),
            ActionConfig {
                lossy: *lossy,
                reversible: *reversible,
                ..Default::default()
            },
        );
    }
    let mut group_map: HashMap<String, Vec<String>> = HashMap::new();
    group_map.insert(
        "eviction".into(),
        vec!["kv.evict_sliding".into(), "kv.evict_h2o".into()],
    );
    group_map.insert(
        "compute_reduction".into(),
        vec!["throttle".into(), "weight.skip".into()],
    );
    let config = PolicyConfig {
        actions: action_map,
        exclusion_groups: group_map,
        ..Default::default()
    };
    let registry = ActionRegistry::from_config(&config);

    // 같은 그룹 내 → excluded
    assert!(registry.is_excluded(&ActionId::KvEvictSliding, &ActionId::KvEvictH2o));
    assert!(registry.is_excluded(&ActionId::Throttle, &ActionId::LayerSkip));

    // 다른 그룹 간 → not excluded
    assert!(!registry.is_excluded(&ActionId::KvEvictSliding, &ActionId::Throttle));
    assert!(!registry.is_excluded(&ActionId::KvEvictH2o, &ActionId::LayerSkip));
    assert!(!registry.is_excluded(&ActionId::KvEvictSliding, &ActionId::LayerSkip));
    assert!(!registry.is_excluded(&ActionId::KvEvictH2o, &ActionId::Throttle));
}

/// 배타 그룹이 없는 설정에서는 어떤 액션 쌍도 excluded가 아니어야 한다.
#[test]
fn test_inv_016_no_exclusion_groups() {
    let registry = helpers::make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
            ("throttle", false, true),
        ],
        &[], // 배타 그룹 없음
    );

    assert!(
        !registry.is_excluded(&ActionId::KvEvictSliding, &ActionId::KvEvictH2o),
        "INV-016: 배타 그룹이 없으면 어떤 액션 쌍도 excluded가 아니다"
    );
    assert!(!registry.is_excluded(&ActionId::Throttle, &ActionId::KvEvictSliding));
}

/// 배타 그룹 멤버 수가 3개 이상일 때 모든 쌍이 excluded여야 한다.
#[test]
fn test_inv_016_exclusion_group_three_members() {
    let mut action_map = HashMap::new();
    for name in &["kv.evict_sliding", "kv.evict_h2o", "kv.quant_dynamic"] {
        action_map.insert(
            name.to_string(),
            ActionConfig {
                lossy: true,
                reversible: false,
                ..Default::default()
            },
        );
    }
    let mut group_map: HashMap<String, Vec<String>> = HashMap::new();
    group_map.insert(
        "cache_ops".into(),
        vec![
            "kv.evict_sliding".into(),
            "kv.evict_h2o".into(),
            "kv.quant_dynamic".into(),
        ],
    );
    let config = PolicyConfig {
        actions: action_map,
        exclusion_groups: group_map,
        ..Default::default()
    };
    let registry = ActionRegistry::from_config(&config);

    // 3개 액션의 모든 쌍 확인
    let members = [
        ActionId::KvEvictSliding,
        ActionId::KvEvictH2o,
        ActionId::KvQuantDynamic,
    ];
    for i in 0..members.len() {
        for j in (i + 1)..members.len() {
            assert!(
                registry.is_excluded(&members[i], &members[j]),
                "INV-016: {:?}와 {:?}는 같은 배타 그룹이므로 excluded",
                members[i],
                members[j]
            );
        }
    }
}

/// ActionRegistry의 exclusion_groups() 메서드가 설정한 그룹을 정확히 반영해야 한다.
#[test]
fn test_inv_016_exclusion_groups_accessor() {
    let registry = helpers::make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
        ],
        &[("eviction", &["kv.evict_sliding", "kv.evict_h2o"])],
    );

    let groups = registry.exclusion_groups();
    assert!(groups.contains_key("eviction"));
    let members = &groups["eviction"];
    assert!(members.contains(&ActionId::KvEvictSliding));
    assert!(members.contains(&ActionId::KvEvictH2o));
    assert_eq!(members.len(), 2);
}

/// ActionMeta의 exclusion_group 필드가 올바르게 설정되어야 한다.
#[test]
fn test_inv_016_action_meta_exclusion_group_field() {
    let registry = helpers::make_registry(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
            ("throttle", false, true),
        ],
        &[("eviction", &["kv.evict_sliding", "kv.evict_h2o"])],
    );

    let sliding_meta = registry.get(&ActionId::KvEvictSliding).unwrap();
    assert_eq!(
        sliding_meta.exclusion_group.as_deref(),
        Some("eviction"),
        "INV-016: kv_evict_sliding의 exclusion_group이 'eviction'이어야 한다"
    );

    let h2o_meta = registry.get(&ActionId::KvEvictH2o).unwrap();
    assert_eq!(
        h2o_meta.exclusion_group.as_deref(),
        Some("eviction"),
        "INV-016: kv_evict_h2o의 exclusion_group이 'eviction'이어야 한다"
    );

    let throttle_meta = registry.get(&ActionId::Throttle).unwrap();
    assert!(
        throttle_meta.exclusion_group.is_none(),
        "INV-016: throttle은 배타 그룹에 속하지 않아야 한다"
    );
}
