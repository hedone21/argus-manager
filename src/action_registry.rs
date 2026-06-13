use std::collections::HashMap;

use crate::config::PolicyConfig;
use crate::types::{ActionId, ActionKind, ActionMeta, ParamRange};

/// 액션 메타데이터와 배타 그룹을 통합 관리하는 레지스트리.
pub struct ActionRegistry {
    actions: HashMap<ActionId, ActionMeta>,
    exclusion_groups: HashMap<String, Vec<ActionId>>,
}

impl ActionRegistry {
    /// PolicyConfig로부터 ActionRegistry를 구성한다.
    ///
    /// - lossy == false → Lossless, lossy == true → Lossy
    /// - exclusion_groups의 문자열을 ActionId로 변환 (알 수 없는 이름은 무시)
    pub fn from_config(config: &PolicyConfig) -> Self {
        let mut actions: HashMap<ActionId, ActionMeta> = HashMap::new();

        for (name, action_cfg) in &config.actions {
            let Some(id) = ActionId::from_str(name) else {
                continue;
            };
            let kind = if action_cfg.lossy {
                ActionKind::Lossy
            } else {
                ActionKind::Lossless
            };
            let meta = ActionMeta {
                id,
                kind,
                reversible: action_cfg.reversible,
                param_range: default_param_range(id),
                exclusion_group: None, // 이후 exclusion_groups 처리에서 채움
                default_cost: action_cfg.default_cost,
            };
            actions.insert(id, meta);
        }

        // exclusion_groups 처리: 문자열 → ActionId 변환
        let mut exclusion_groups: HashMap<String, Vec<ActionId>> = HashMap::new();
        for (group_name, members) in &config.exclusion_groups {
            let ids: Vec<ActionId> = members
                .iter()
                .filter_map(|s| ActionId::from_str(s))
                .collect();
            if !ids.is_empty() {
                // 각 액션 메타에 exclusion_group 설정
                for id in &ids {
                    if let Some(meta) = actions.get_mut(id) {
                        meta.exclusion_group = Some(group_name.clone());
                    }
                }
                exclusion_groups.insert(group_name.clone(), ids);
            }
        }

        Self {
            actions,
            exclusion_groups,
        }
    }

    /// 액션 메타데이터 조회.
    pub fn get(&self, action: &ActionId) -> Option<&ActionMeta> {
        self.actions.get(action)
    }

    /// 등록된 모든 액션 반복.
    pub fn all_actions(&self) -> impl Iterator<Item = &ActionMeta> {
        self.actions.values()
    }

    /// Lossy 액션 목록.
    pub fn lossy_actions(&self) -> Vec<ActionId> {
        self.actions
            .values()
            .filter(|m| m.kind == ActionKind::Lossy)
            .map(|m| m.id)
            .collect()
    }

    /// Lossless 액션 목록.
    pub fn lossless_actions(&self) -> Vec<ActionId> {
        self.actions
            .values()
            .filter(|m| m.kind == ActionKind::Lossless)
            .map(|m| m.id)
            .collect()
    }

    /// 배타 그룹 맵 반환.
    pub fn exclusion_groups(&self) -> &HashMap<String, Vec<ActionId>> {
        &self.exclusion_groups
    }

    /// 두 액션이 같은 배타 그룹에 속하는지 확인.
    pub fn is_excluded(&self, a: &ActionId, b: &ActionId) -> bool {
        for members in self.exclusion_groups.values() {
            let has_a = members.contains(a);
            let has_b = members.contains(b);
            if has_a && has_b {
                return true;
            }
        }
        false
    }

    /// 액션의 기본 QCF 비용(default_cost)을 반환한다.
    /// 등록되지 않은 액션은 1.0을 반환한다.
    pub fn default_cost(&self, action: &ActionId) -> f32 {
        self.actions
            .get(action)
            .map(|m| m.default_cost)
            .unwrap_or(1.0)
    }
}

/// 액션별 기본 파라미터 범위.
fn default_param_range(id: ActionId) -> Option<ParamRange> {
    match id {
        ActionId::KvEvictSliding | ActionId::KvEvictH2o | ActionId::KvMergeD2o => {
            Some(ParamRange {
                param_name: "keep_ratio".into(),
                min: 0.3,
                max: 0.9,
            })
        }
        ActionId::KvEvictStreaming => Some(ParamRange {
            param_name: "window_size".into(),
            min: 64.0,
            max: 2048.0,
        }),
        ActionId::KvQuantDynamic => Some(ParamRange {
            param_name: "target_bits".into(),
            min: 4.0,
            max: 8.0,
        }),
        ActionId::LayerSkip => Some(ParamRange {
            param_name: "skip_layers".into(),
            min: 1.0,
            max: 8.0,
        }),
        ActionId::Throttle => Some(ParamRange {
            param_name: "delay_ms".into(),
            min: 0.0,
            max: 100.0,
        }),
        ActionId::SwapWeights => Some(ParamRange {
            param_name: "ratio".into(),
            min: 0.0,
            max: 0.9,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ActionConfig, PolicyConfig};
    use std::collections::HashMap;

    /// 테스트용 기본 PolicyConfig 생성
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

    /// PolicyConfig로부터 ActionRegistry 생성이 정상적으로 동작해야 한다.
    #[test]
    fn test_from_config_basic() {
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
        // 등록되지 않은 액션은 None
        assert!(registry.get(&ActionId::LayerSkip).is_none());
    }

    /// lossy 플래그 기반 Lossy/Lossless 분류가 올바르게 동작해야 한다.
    #[test]
    fn test_lossy_lossless_classification() {
        let config = make_policy_config(
            &[
                ("switch_hw", false, true),        // Lossless
                ("throttle", false, true),         // Lossless
                ("kv.evict_sliding", true, false), // Lossy
                ("kv.evict_h2o", true, false),     // Lossy
                ("kv.quant_dynamic", true, false), // Lossy
                ("weight.skip", true, true),       // Lossy
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
        assert!(lossy.contains(&ActionId::KvEvictH2o));
        assert!(!lossy.contains(&ActionId::SwitchHw));

        let lossless = registry.lossless_actions();
        assert!(lossless.contains(&ActionId::SwitchHw));
        assert!(!lossless.contains(&ActionId::KvEvictSliding));
    }

    /// exclusion_groups가 올바르게 파싱되어 두 eviction 액션을 포함해야 한다.
    #[test]
    fn test_exclusion_groups() {
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
        assert!(
            groups.contains_key("eviction"),
            "eviction group should exist"
        );

        let eviction_members = &groups["eviction"];
        assert!(eviction_members.contains(&ActionId::KvEvictSliding));
        assert!(eviction_members.contains(&ActionId::KvEvictH2o));
        assert_eq!(eviction_members.len(), 2);
    }

    /// 같은 배타 그룹의 pair는 is_excluded가 true를 반환해야 한다.
    #[test]
    fn test_is_excluded() {
        let config = make_policy_config(
            &[
                ("kv.evict_sliding", true, false),
                ("kv.evict_h2o", true, false),
                ("switch_hw", false, true),
            ],
            &[("eviction", &["kv.evict_sliding", "kv.evict_h2o"])],
        );
        let registry = ActionRegistry::from_config(&config);

        // 같은 그룹 → excluded
        assert!(registry.is_excluded(&ActionId::KvEvictSliding, &ActionId::KvEvictH2o));
        assert!(registry.is_excluded(&ActionId::KvEvictH2o, &ActionId::KvEvictSliding));

        // 다른 그룹 → not excluded
        assert!(!registry.is_excluded(&ActionId::KvEvictSliding, &ActionId::SwitchHw));
        assert!(!registry.is_excluded(&ActionId::SwitchHw, &ActionId::KvEvictH2o));
    }

    /// config에 없는 action name은 무시되어야 한다.
    #[test]
    fn test_unknown_action_ignored() {
        let config = make_policy_config(
            &[
                ("switch_hw", false, true),
                ("unknown_action_xyz", true, false), // 알 수 없는 액션
            ],
            &[("bad_group", &["unknown_action_xyz", "also_unknown"])],
        );
        let registry = ActionRegistry::from_config(&config);

        // 알려진 액션만 등록됨
        assert!(registry.get(&ActionId::SwitchHw).is_some());
        // 알 수 없는 그룹은 비어있거나 존재하지 않음
        let empty_group = registry.exclusion_groups().get("bad_group");
        assert!(
            empty_group.is_none() || empty_group.unwrap().is_empty(),
            "unknown action names should be filtered out"
        );
    }

    /// 등록된 액션의 kind, reversible 값이 설정과 일치해야 한다.
    #[test]
    fn test_action_meta_fields() {
        let config = make_policy_config(
            &[
                ("kv.evict_sliding", true, false),
                ("weight.skip", true, true),
            ],
            &[],
        );
        let registry = ActionRegistry::from_config(&config);

        let evict_meta = registry.get(&ActionId::KvEvictSliding).unwrap();
        assert_eq!(evict_meta.kind, ActionKind::Lossy);
        assert!(!evict_meta.reversible);

        let skip_meta = registry.get(&ActionId::LayerSkip).unwrap();
        assert_eq!(skip_meta.kind, ActionKind::Lossy);
        assert!(skip_meta.reversible);
    }

    /// 파라미터 범위가 기본값으로 설정되어야 한다.
    #[test]
    fn test_default_param_ranges() {
        let config = make_policy_config(
            &[
                ("kv.evict_sliding", true, false),
                ("kv.quant_dynamic", true, false),
                ("switch_hw", false, true), // param_range 없음
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

    /// default_cost 기본값은 1.0이어야 한다.
    #[test]
    fn test_default_cost_fallback_is_one() {
        let config = make_policy_config(&[("kv.evict_sliding", true, false)], &[]);
        let registry = ActionRegistry::from_config(&config);
        // default_cost 미지정 → 1.0 반환
        assert!((registry.default_cost(&ActionId::KvEvictSliding) - 1.0).abs() < f32::EPSILON);
        // 등록되지 않은 액션 → 1.0 반환
        assert!((registry.default_cost(&ActionId::LayerSkip) - 1.0).abs() < f32::EPSILON);
    }

    /// 명시적으로 설정된 default_cost가 정확히 반환되어야 한다.
    #[test]
    fn test_default_cost_explicit_value() {
        let mut action_map = HashMap::new();
        action_map.insert(
            "kv.evict_sliding".to_string(),
            ActionConfig {
                lossy: true,
                reversible: false,
                default_cost: 0.4,
            },
        );
        action_map.insert(
            "weight.skip".to_string(),
            ActionConfig {
                lossy: true,
                reversible: true,
                default_cost: 1.8,
            },
        );
        let config = PolicyConfig {
            actions: action_map,
            ..Default::default()
        };
        let registry = ActionRegistry::from_config(&config);
        assert!((registry.default_cost(&ActionId::KvEvictSliding) - 0.4).abs() < f32::EPSILON);
        assert!((registry.default_cost(&ActionId::LayerSkip) - 1.8).abs() < f32::EPSILON);
    }
}
