//! MGR-DAT-020 ~ MGR-DAT-056: Config + ActionRegistry Spec 테스트
//!
//! Config 기본값, PolicyConfig 파싱, ActionConfig default_cost,
//! ActionRegistry 생성/분류/배타 그룹/파라미터 범위를 검증한다.

use std::collections::HashMap;

use argus_manager::action_registry::ActionRegistry;
use argus_manager::config::{ActionConfig, Config, PolicyConfig};
use argus_manager::types::{ActionId, ActionKind};

// ── MGR-DAT-020/021: Config 기본값 ─────────────────────────────────

/// MGR-DAT-020: 기본 Config에서 poll_interval_ms = 1000.
#[test]
fn test_mgr_dat_020_default_config_poll_interval() {
    let config = Config::default();
    assert_eq!(config.manager.poll_interval_ms, 1000);
}

/// MGR-DAT-021: 기본 Config에서 Optional monitor들은 None.
#[test]
fn test_mgr_dat_021_default_config_all_monitors_enabled() {
    let config = Config::default();
    assert!(config.memory.is_none());
    assert!(config.external.is_none());
}

// ── MGR-DAT-030/031: PolicyConfig 파싱 ─────────────────────────────

/// MGR-DAT-030: PolicyConfig를 TOML에서 올바르게 파싱한다.
#[test]
fn test_mgr_dat_030_parse_policy_config() {
    let toml_str = r#"
[policy.pi_controller]
compute_kp = 1.5
compute_ki = 0.3

[policy.supervisory]
warning_threshold = 0.4
critical_threshold = 0.7

[policy.selector]
latency_budget = 0.5

[policy.actions.switch_hw]
lossy = false
reversible = true

[policy.actions."kv.evict_sliding"]
lossy = true
reversible = false

[policy.exclusion_groups]
eviction = ["kv.evict_sliding", "kv.evict_h2o"]
"#;
    let config: Config = toml::from_str(toml_str).unwrap();
    let policy = config.policy.unwrap();
    assert!((policy.pi_controller.compute_kp - 1.5).abs() < f32::EPSILON);
    assert!((policy.supervisory.warning_threshold - 0.4).abs() < f32::EPSILON);
    let switch = policy.actions.get("switch_hw").unwrap();
    assert!(!switch.lossy);
    assert!(switch.reversible);
    let kv_evict = policy.actions.get("kv.evict_sliding").unwrap();
    assert!(kv_evict.lossy);
    let eviction = policy.exclusion_groups.get("eviction").unwrap();
    assert_eq!(eviction.len(), 2);
}

/// MGR-DAT-031: PolicyConfig 기본값이 올바르게 설정된다.
#[test]
fn test_mgr_dat_031_policy_config_defaults() {
    let policy = PolicyConfig::default();
    assert!((policy.pi_controller.compute_kp - 1.5).abs() < f32::EPSILON);
    assert!((policy.pi_controller.memory_kp - 2.0).abs() < f32::EPSILON);
    assert!((policy.supervisory.warning_threshold - 0.4).abs() < f32::EPSILON);
    assert!((policy.supervisory.hold_time_secs - 4.0).abs() < f32::EPSILON);
    assert_eq!(policy.selector.algorithm, "exhaustive");
    assert!((policy.relief_model.forgetting_factor - 0.995).abs() < f32::EPSILON);
    assert_eq!(policy.relief_model.prior_weight, 5);
}

// ── MGR-DAT-035: ActionConfig default_cost ──────────────────────────

/// MGR-DAT-035: default_cost 필드 없이 파싱 시 1.0으로 폴백.
#[test]
fn test_mgr_dat_035_action_config_default_cost_fallback() {
    let toml_str = r#"
[policy.actions."kv.evict_sliding"]
lossy = true
reversible = false
"#;
    let config: Config = toml::from_str(toml_str).unwrap();
    let policy = config.policy.unwrap();
    let action = policy.actions.get("kv.evict_sliding").unwrap();
    assert!((action.default_cost - 1.0).abs() < f32::EPSILON);
}

/// MGR-DAT-035: 명시적 default_cost 값이 정확히 로드된다.
#[test]
fn test_mgr_dat_035_action_config_explicit_default_cost() {
    let toml_str = r#"
[policy.actions."kv.evict_sliding"]
lossy = true
reversible = false
default_cost = 0.5

[policy.actions."weight.skip"]
lossy = true
reversible = true
default_cost = 2.0
"#;
    let config: Config = toml::from_str(toml_str).unwrap();
    let policy = config.policy.unwrap();
    let evict = policy.actions.get("kv.evict_sliding").unwrap();
    assert!((evict.default_cost - 0.5).abs() < f32::EPSILON);
    let skip = policy.actions.get("weight.skip").unwrap();
    assert!((skip.default_cost - 2.0).abs() < f32::EPSILON);
}

/// MGR-DAT-035: ActionConfig::default()의 default_cost는 1.0.
#[test]
fn test_mgr_dat_035_action_config_default_impl() {
    let cfg = ActionConfig::default();
    assert!((cfg.default_cost - 1.0).abs() < f32::EPSILON);
}

// ── MGR-DAT-054/055: ActionRegistry 생성 및 분류 ───────────────────

fn make_policy_config(
    actions: &[(&str, bool, bool)],
    exclusion_groups: &[(&str, &[&str])],
) -> PolicyConfig {
    let mut action_map = HashMap::new();
    for (name, lossy, reversible) in actions {
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
    for (group, members) in exclusion_groups {
        group_map.insert(
            group.to_string(),
            members.iter().map(|s| s.to_string()).collect(),
        );
    }
    PolicyConfig {
        actions: action_map,
        exclusion_groups: group_map,
        ..Default::default()
    }
}

/// MGR-DAT-054: PolicyConfig로부터 ActionRegistry가 정상 생성된다.
#[test]
fn test_mgr_dat_054_from_config_basic() {
    let config = make_policy_config(
        &[
            ("switch_hw", false, true),
            ("kv.evict_sliding", true, false),
            ("throttle", false, true),
        ],
        &[],
    );
    let registry = ActionRegistry::from_config(&config);

    assert!(registry.get(&ActionId::SwitchHw).is_some());
    assert!(registry.get(&ActionId::KvEvictSliding).is_some());
    assert!(registry.get(&ActionId::Throttle).is_some());
    assert!(registry.get(&ActionId::LayerSkip).is_none());
}

/// MGR-DAT-054: config에 없는 action name은 무시된다.
#[test]
fn test_mgr_dat_054_unknown_action_ignored() {
    let config = make_policy_config(
        &[
            ("switch_hw", false, true),
            ("unknown_action_xyz", true, false),
        ],
        &[("bad_group", &["unknown_action_xyz", "also_unknown"])],
    );
    let registry = ActionRegistry::from_config(&config);

    assert!(registry.get(&ActionId::SwitchHw).is_some());
    let empty_group = registry.exclusion_groups().get("bad_group");
    assert!(
        empty_group.is_none() || empty_group.unwrap().is_empty(),
        "unknown action names should be filtered out"
    );
}

/// MGR-DAT-055: lossy 플래그 기반 Lossy/Lossless 분류가 올바르다.
#[test]
fn test_mgr_dat_055_lossy_lossless_classification() {
    let config = make_policy_config(
        &[
            ("switch_hw", false, true),
            ("throttle", false, true),
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
            ("kv.quant_dynamic", true, false),
            ("weight.skip", true, true),
        ],
        &[],
    );
    let registry = ActionRegistry::from_config(&config);

    // Lossless 확인
    let switch_meta = registry.get(&ActionId::SwitchHw).unwrap();
    assert_eq!(switch_meta.kind, ActionKind::Lossless);
    let throttle_meta = registry.get(&ActionId::Throttle).unwrap();
    assert_eq!(throttle_meta.kind, ActionKind::Lossless);

    // Lossy 확인
    let evict_meta = registry.get(&ActionId::KvEvictSliding).unwrap();
    assert_eq!(evict_meta.kind, ActionKind::Lossy);
    let skip_meta = registry.get(&ActionId::LayerSkip).unwrap();
    assert_eq!(skip_meta.kind, ActionKind::Lossy);

    // 분류 목록 검증
    let lossy = registry.lossy_actions();
    assert!(lossy.contains(&ActionId::KvEvictSliding));
    assert!(!lossy.contains(&ActionId::SwitchHw));

    let lossless = registry.lossless_actions();
    assert!(lossless.contains(&ActionId::SwitchHw));
    assert!(!lossless.contains(&ActionId::KvEvictSliding));
}

// ── MGR-DAT-056: 기본 파라미터 범위 ────────────────────────────────

/// MGR-DAT-056: 등록된 액션의 파라미터 범위가 기본값으로 설정된다.
#[test]
fn test_mgr_dat_056_default_param_ranges() {
    let config = make_policy_config(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.quant_dynamic", true, false),
            ("switch_hw", false, true),
        ],
        &[],
    );
    let registry = ActionRegistry::from_config(&config);

    let evict_range = registry
        .get(&ActionId::KvEvictSliding)
        .unwrap()
        .param_range
        .as_ref()
        .unwrap();
    assert_eq!(evict_range.param_name, "keep_ratio");
    assert!((evict_range.min - 0.3).abs() < f32::EPSILON);
    assert!((evict_range.max - 0.9).abs() < f32::EPSILON);

    let quant_range = registry
        .get(&ActionId::KvQuantDynamic)
        .unwrap()
        .param_range
        .as_ref()
        .unwrap();
    assert_eq!(quant_range.param_name, "target_bits");

    // switch_hw는 param_range 없음
    let switch_meta = registry.get(&ActionId::SwitchHw).unwrap();
    assert!(switch_meta.param_range.is_none());
}

// ── MGR-DAT-036: exclusion groups + is_excluded ────────────────────

/// MGR-DAT-036: exclusion_groups가 올바르게 파싱된다.
#[test]
fn test_mgr_dat_036_exclusion_groups() {
    let config = make_policy_config(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
            ("switch_hw", false, true),
        ],
        &[("eviction", &["kv.evict_sliding", "kv.evict_h2o"])],
    );
    let registry = ActionRegistry::from_config(&config);

    let groups = registry.exclusion_groups();
    assert!(groups.contains_key("eviction"));

    let eviction_members = &groups["eviction"];
    assert!(eviction_members.contains(&ActionId::KvEvictSliding));
    assert!(eviction_members.contains(&ActionId::KvEvictH2o));
    assert_eq!(eviction_members.len(), 2);
}

/// MGR-DAT-036: 같은 배타 그룹의 pair는 is_excluded가 true.
#[test]
fn test_mgr_dat_036_is_excluded() {
    let config = make_policy_config(
        &[
            ("kv.evict_sliding", true, false),
            ("kv.evict_h2o", true, false),
            ("switch_hw", false, true),
        ],
        &[("eviction", &["kv.evict_sliding", "kv.evict_h2o"])],
    );
    let registry = ActionRegistry::from_config(&config);

    assert!(registry.is_excluded(&ActionId::KvEvictSliding, &ActionId::KvEvictH2o));
    assert!(registry.is_excluded(&ActionId::KvEvictH2o, &ActionId::KvEvictSliding));
    assert!(!registry.is_excluded(&ActionId::KvEvictSliding, &ActionId::SwitchHw));
    assert!(!registry.is_excluded(&ActionId::SwitchHw, &ActionId::KvEvictH2o));
}
