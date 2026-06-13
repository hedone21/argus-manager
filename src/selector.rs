use std::collections::HashMap;

use crate::action_registry::ActionRegistry;
use crate::relief::ReliefEstimator;
use crate::types::{
    ActionCommand, ActionId, ActionKind, ActionParams, Domain, FeatureVector, OperatingMode,
    Operation, PressureVector, ReliefVector,
};

/// Cross-Domain Action Selector.
///
/// Stateless: 모든 상태는 `select()` 호출 인자로 전달된다.
/// 설계 §4 (docs/36_policy_design.md) 에 따라 exhaustive 조합 탐색으로 최적 액션 조합을 선택한다.
pub struct ActionSelector;

/// 조합 탐색 시 후보 정보
struct CandidateInfo {
    action: ActionId,
    relief: ReliefVector,
    cost: f32,
}

impl ActionSelector {
    /// 최적 액션 조합을 선택하여 `ActionCommand` 목록으로 반환한다.
    ///
    /// # 파라미터
    /// - `registry`: 액션 메타데이터 및 exclusion group
    /// - `estimator`: Relief 예측기
    /// - `pressure`: 현재 pressure 벡터
    /// - `mode`: 운영 모드 (Warning → lossless만, Critical → lossy 허용)
    /// - `engine_state`: 엔진 feature 벡터 (ReliefEstimator 예측에 사용)
    /// - `qcf_values`: 각 lossy 액션의 QCF 추정값 (on-demand)
    /// - `latency_budget`: 허용 가능한 latency 악화 상한 (양수)
    /// - `active_actions`: Engine이 현재 적용 중인 액션 목록 (EngineStatus에서 직접 전달)
    /// - `available_actions`: Engine이 실행 가능하다고 보고한 액션 목록 (비어있으면 필터링 안함)
    #[allow(clippy::too_many_arguments)]
    pub fn select(
        registry: &ActionRegistry,
        estimator: &dyn ReliefEstimator,
        pressure: &PressureVector,
        mode: OperatingMode,
        engine_state: &FeatureVector,
        qcf_values: &HashMap<ActionId, f32>,
        latency_budget: f32,
        active_actions: &[ActionId],
        available_actions: &[ActionId],
    ) -> Vec<ActionCommand> {
        // pressure가 전혀 없으면 빈 조합(cost=0)이 모든 제약을 충족하므로 즉시 반환
        if pressure.compute <= 0.0 && pressure.memory <= 0.0 && pressure.thermal <= 0.0 {
            return vec![];
        }

        let candidate_ids =
            Self::filter_candidates(registry, mode, active_actions, available_actions);

        if candidate_ids.is_empty() {
            return vec![];
        }

        // 각 후보에 대해 relief 예측 + cost 계산
        let candidates: Vec<CandidateInfo> = candidate_ids
            .iter()
            .map(|&action| {
                let relief = estimator.predict(&action, engine_state);
                let cost = Self::compute_cost(action, registry, qcf_values);
                CandidateInfo {
                    action,
                    relief,
                    cost,
                }
            })
            .collect();

        // 최적 조합 탐색
        let selected = Self::find_optimal(&candidates, pressure, registry, latency_budget);

        // 선택된 각 액션에 대해 파라미터 결정 후 ActionCommand 생성
        selected
            .into_iter()
            .map(|action| {
                let params = Self::parametrize(action, pressure, registry);
                ActionCommand {
                    action,
                    operation: Operation::Apply(params),
                }
            })
            .collect()
    }

    /// Phase 1: 후보 필터링
    ///
    /// - Warning mode: lossy 액션 제외
    /// - 이미 활성화된 액션 제외
    /// - available_actions가 비어있지 않으면 그 목록에 없는 액션 제외 (backward compat: 비어있으면 필터링 안함)
    fn filter_candidates(
        registry: &ActionRegistry,
        mode: OperatingMode,
        active_actions: &[ActionId],
        available_actions: &[ActionId],
    ) -> Vec<ActionId> {
        registry
            .all_actions()
            .filter(|meta| {
                // Warning mode에서는 Lossy 제외
                if mode == OperatingMode::Warning && meta.kind == ActionKind::Lossy {
                    return false;
                }
                // 이미 활성 중인 액션 제외
                if active_actions.contains(&meta.id) {
                    return false;
                }
                // Engine이 실행 가능하다고 보고한 것만 포함
                // available_actions가 비어있으면 필터링 안 함 (backward compat: Engine이 아직 보고 안 할 때)
                if !available_actions.is_empty() && !available_actions.contains(&meta.id) {
                    return false;
                }
                true
            })
            .map(|meta| meta.id)
            .collect()
    }

    /// 액션의 cost 계산.
    ///
    /// cost = qcf_value (lossy)
    /// Lossless이면 0.
    /// Lossy이지만 qcf_values에 없으면 INFINITY (사실상 선택 안됨).
    fn compute_cost(
        action: ActionId,
        registry: &ActionRegistry,
        qcf_values: &HashMap<ActionId, f32>,
    ) -> f32 {
        let Some(meta) = registry.get(&action) else {
            return f32::INFINITY;
        };
        if meta.kind == ActionKind::Lossless {
            return 0.0;
        }
        // QCF/OPR already normalizes quality cost across domains; no alpha needed
        qcf_values.get(&action).copied().unwrap_or(f32::INFINITY)
    }

    /// Phase 3: Exhaustive 조합 탐색으로 최적 액션 조합을 반환한다.
    ///
    /// 모든 압력을 해소하는 조합 중 total cost가 최소인 것을 선택한다.
    /// 어떤 조합도 완전 해소를 못하면 coverage 최대화 (best-effort).
    fn find_optimal(
        candidates: &[CandidateInfo],
        pressure: &PressureVector,
        registry: &ActionRegistry,
        latency_budget: f32,
    ) -> Vec<ActionId> {
        let n = candidates.len();
        if n == 0 {
            return vec![];
        }

        let total_masks = 1usize << n; // 2^n 조합

        let mut best_cost = f32::INFINITY;
        let mut best_mask: Option<usize> = None;

        // best-effort용: coverage가 최대인 조합
        let mut best_coverage = f32::NEG_INFINITY;
        let mut best_effort_mask: Option<usize> = None;

        for mask in 0..total_masks {
            // 선택된 액션 인덱스 수집
            let selected_indices: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();

            // exclusion group 검사: 같은 그룹의 액션이 2개 이상이면 skip
            if Self::has_exclusion_conflict(&selected_indices, candidates, registry) {
                continue;
            }

            // 합산 relief 계산
            let total_relief = selected_indices
                .iter()
                .fold(ReliefVector::zero(), |acc, &i| acc + candidates[i].relief);

            // latency budget 검사 (항상 적용 — 안전 제약)
            if total_relief.latency < -latency_budget {
                continue;
            }

            // pressure 제약 검사
            let satisfies = total_relief.compute >= pressure.compute
                && total_relief.memory >= pressure.memory
                && total_relief.thermal >= pressure.thermal;

            if satisfies {
                let total_cost: f32 = selected_indices.iter().map(|&i| candidates[i].cost).sum();
                if total_cost < best_cost {
                    best_cost = total_cost;
                    best_mask = Some(mask);
                }
            } else {
                // best-effort: coverage 최대화
                let coverage = f32::min(total_relief.compute, pressure.compute)
                    + f32::min(total_relief.memory, pressure.memory)
                    + f32::min(total_relief.thermal, pressure.thermal);
                if coverage > best_coverage {
                    best_coverage = coverage;
                    best_effort_mask = Some(mask);
                }
            }
        }

        // 완전 해소 가능한 조합이 있으면 그것을 사용, 없으면 best-effort
        let chosen_mask = best_mask.or(best_effort_mask);

        match chosen_mask {
            None => vec![],
            Some(mask) => (0..n)
                .filter(|&i| mask & (1 << i) != 0)
                .map(|i| candidates[i].action)
                .collect(),
        }
    }

    /// 선택된 인덱스 집합에 exclusion conflict가 있는지 확인한다.
    fn has_exclusion_conflict(
        indices: &[usize],
        candidates: &[CandidateInfo],
        registry: &ActionRegistry,
    ) -> bool {
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let a = candidates[indices[i]].action;
                let b = candidates[indices[j]].action;
                if registry.is_excluded(&a, &b) {
                    return true;
                }
            }
        }
        false
    }

    /// Phase 4: 파라미터 결정.
    ///
    /// primary_domain의 pressure intensity에 따라 param_range를 선형 보간한다.
    ///
    /// intensity 0.0 → range.max (보수적)
    /// intensity 1.0 → range.min (공격적)
    fn parametrize(
        action: ActionId,
        pressure: &PressureVector,
        registry: &ActionRegistry,
    ) -> ActionParams {
        let Some(meta) = registry.get(&action) else {
            return ActionParams::default();
        };
        let Some(range) = &meta.param_range else {
            return ActionParams::default();
        };

        let intensity = match action.primary_domain() {
            Domain::Compute => pressure.compute,
            Domain::Memory => pressure.memory,
            Domain::Thermal => pressure.thermal,
        };

        let intensity = intensity.clamp(0.0, 1.0);
        let value = range.max - intensity * (range.max - range.min);
        let value = value.clamp(range.min, range.max);

        let mut params = ActionParams::default();
        params.values.insert(range.param_name.clone(), value);
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ActionConfig, PolicyConfig};
    use crate::types::feature;
    use std::collections::HashMap;
    use std::io;
    use std::path::Path;

    // -------------------------------------------------------------------------
    // Mock ReliefEstimator
    // -------------------------------------------------------------------------

    struct MockEstimator {
        predictions: HashMap<ActionId, ReliefVector>,
    }

    impl MockEstimator {
        fn new(predictions: HashMap<ActionId, ReliefVector>) -> Self {
            Self { predictions }
        }
    }

    impl ReliefEstimator for MockEstimator {
        fn predict(&self, action: &ActionId, _state: &FeatureVector) -> ReliefVector {
            self.predictions.get(action).copied().unwrap_or_default()
        }

        fn observe(&mut self, _: &ActionId, _: &FeatureVector, _: &ReliefVector) {}

        fn save(&self, _: &Path) -> io::Result<()> {
            Ok(())
        }

        fn load(&mut self, _: &Path) -> io::Result<()> {
            Ok(())
        }

        fn observation_count(&self, _: &ActionId) -> u32 {
            0
        }
    }

    // -------------------------------------------------------------------------
    // 헬퍼
    // -------------------------------------------------------------------------

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

    fn make_registry(
        actions: &[(&str, bool, bool)],
        exclusion_groups: &[(&str, &[&str])],
    ) -> ActionRegistry {
        let config = make_policy_config(actions, exclusion_groups);
        ActionRegistry::from_config(&config)
    }

    fn rv(compute: f32, memory: f32, thermal: f32, latency: f32) -> ReliefVector {
        ReliefVector {
            compute,
            memory,
            thermal,
            latency,
        }
    }

    fn pv(compute: f32, memory: f32, thermal: f32) -> PressureVector {
        PressureVector {
            compute,
            memory,
            thermal,
        }
    }

    fn no_active_state() -> FeatureVector {
        FeatureVector::zeros()
    }

    fn command_ids(cmds: &[ActionCommand]) -> Vec<ActionId> {
        cmds.iter().map(|c| c.action).collect()
    }

    // -------------------------------------------------------------------------
    // 테스트
    // -------------------------------------------------------------------------

    /// Warning mode에서 lossy 후보가 필터링되어 lossless만 선택된다.
    #[test]
    fn test_warning_mode_lossless_only() {
        // switch_hw: lossless, compute+thermal 해소
        // kv_evict_sliding: lossy, memory 해소
        let registry = make_registry(
            &[
                ("switch_hw", false, true),
                ("kv.evict_sliding", true, false),
            ],
            &[],
        );

        let mut predictions = HashMap::new();
        predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, -0.1));
        predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.1));
        let estimator = MockEstimator::new(predictions);

        // compute pressure만 있음
        let pressure = pv(0.6, 0.0, 0.0);
        let state = no_active_state();
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

        let ids = command_ids(&cmds);
        // lossy인 kv_evict_sliding은 포함되면 안 됨
        assert!(
            !ids.contains(&ActionId::KvEvictSliding),
            "lossy action must not appear in Warning mode"
        );
        // switch_hw (lossless)는 포함됨
        assert!(
            ids.contains(&ActionId::SwitchHw),
            "lossless action should be selected"
        );
    }

    /// Critical mode에서 여러 lossy 중 cost가 가장 낮은 조합을 선택한다.
    #[test]
    fn test_critical_mode_minimum_cost() {
        // 두 evict 액션 모두 memory pressure 해소 가능하지만 different cost
        // kv_evict_sliding: qcf=0.5 → cost=0.5
        // kv_evict_h2o: qcf=2.0 → cost=2.0
        let registry = make_registry(
            &[
                ("kv.evict_sliding", true, false),
                ("kv.evict_h2o", true, false),
            ],
            &[], // exclusion group 없음 (이 테스트에서는 동시 선택 가능하게)
        );

        let mut predictions = HashMap::new();
        predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
        predictions.insert(ActionId::KvEvictH2o, rv(0.0, 0.9, 0.0, 0.0));
        let estimator = MockEstimator::new(predictions);

        let pressure = pv(0.0, 0.7, 0.0);
        let state = no_active_state();
        let mut qcf = HashMap::new();
        qcf.insert(ActionId::KvEvictSliding, 0.5f32);
        qcf.insert(ActionId::KvEvictH2o, 2.0f32);

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

        let ids = command_ids(&cmds);
        // sliding이 더 저렴하므로 선택되어야 함
        assert!(
            ids.contains(&ActionId::KvEvictSliding),
            "cheaper action should be selected"
        );
        // h2o 단독으로 충분한데 굳이 추가하면 cost 증가하므로 선택 안됨
        assert!(
            !ids.contains(&ActionId::KvEvictH2o),
            "more expensive action should not be added unnecessarily"
        );
    }

    /// switch_hw 하나로 compute+thermal 압력을 동시 해소하면 다른 액션 불필요.
    #[test]
    fn test_cross_domain_single_action() {
        let registry = make_registry(
            &[
                ("switch_hw", false, true),
                ("throttle", false, true),
                ("kv.evict_sliding", true, false),
            ],
            &[],
        );

        let mut predictions = HashMap::new();
        // switch_hw가 compute + thermal 동시 해소
        predictions.insert(ActionId::SwitchHw, rv(0.7, 0.0, 0.5, -0.2));
        // throttle은 compute만 해소
        predictions.insert(ActionId::Throttle, rv(0.4, 0.0, 0.2, -0.5));
        // evict는 memory만 해소
        predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.1));
        let estimator = MockEstimator::new(predictions);

        // compute + thermal pressure만 있음 (memory 없음)
        let pressure = pv(0.6, 0.0, 0.4);
        let state = no_active_state();
        let qcf = HashMap::new();

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

        let ids = command_ids(&cmds);
        assert!(
            ids.contains(&ActionId::SwitchHw),
            "switch_hw should be selected for cross-domain relief"
        );
        // memory pressure가 없으므로 evict 불필요
        assert!(
            !ids.contains(&ActionId::KvEvictSliding),
            "evict not needed when no memory pressure"
        );
    }

    /// kv_evict_sliding과 kv_evict_h2o는 exclusion group으로 동시 선택 불가.
    #[test]
    fn test_exclusion_group() {
        let registry = make_registry(
            &[
                ("kv.evict_sliding", true, false),
                ("kv.evict_h2o", true, false),
            ],
            &[("eviction", &["kv.evict_sliding", "kv.evict_h2o"])],
        );

        let mut predictions = HashMap::new();
        // 둘 다 memory pressure 해소
        predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.6, 0.0, 0.0));
        predictions.insert(ActionId::KvEvictH2o, rv(0.0, 0.6, 0.0, 0.0));
        let estimator = MockEstimator::new(predictions);

        let pressure = pv(0.0, 0.5, 0.0);
        let state = no_active_state();
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

        let ids = command_ids(&cmds);
        // 둘 중 하나만 선택되어야 함
        let evict_count = ids
            .iter()
            .filter(|&&id| id == ActionId::KvEvictSliding || id == ActionId::KvEvictH2o)
            .count();
        assert_eq!(
            evict_count, 1,
            "only one eviction action should be selected due to exclusion group"
        );
    }

    /// throttle+offload 조합이 latency_budget을 초과하면 하나만 선택된다.
    #[test]
    fn test_latency_budget_constraint() {
        // throttle: latency -0.4, offload: latency -0.4 → 합산 -0.8
        // latency_budget = 0.6 → -0.8 < -0.6 이므로 조합 불가
        let registry = make_registry(
            &[("throttle", false, true), ("kv_offload_disk", false, true)],
            &[],
        );

        let mut predictions = HashMap::new();
        predictions.insert(ActionId::Throttle, rv(0.5, 0.0, 0.3, -0.4));
        predictions.insert(ActionId::KvOffloadDisk, rv(0.0, 0.8, 0.0, -0.4));
        let estimator = MockEstimator::new(predictions);

        // compute + memory 압력
        let pressure = pv(0.4, 0.7, 0.0);
        let state = no_active_state();
        let qcf = HashMap::new();

        let cmds = ActionSelector::select(
            &registry,
            &estimator,
            &pressure,
            OperatingMode::Warning,
            &state,
            &qcf,
            0.6, // latency_budget = 0.6
            &[],
            &[],
        );

        let ids = command_ids(&cmds);
        // 두 개 동시 선택은 latency -0.8이므로 budget 초과 → 각각 단독만 가능
        let both_selected =
            ids.contains(&ActionId::Throttle) && ids.contains(&ActionId::KvOffloadDisk);
        assert!(
            !both_selected,
            "throttle+offload together exceeds latency budget"
        );
    }

    /// 모든 조합으로도 pressure를 완전히 해소 못하면 coverage 최대화 조합을 선택한다.
    #[test]
    fn test_best_effort_when_impossible() {
        // switch_hw: compute 0.3 (pressure 0.8 해소 불가)
        // throttle: compute 0.2
        // 둘 합쳐도 0.5 < 0.8
        let registry = make_registry(
            &[("switch_hw", false, true), ("throttle", false, true)],
            &[],
        );

        let mut predictions = HashMap::new();
        predictions.insert(ActionId::SwitchHw, rv(0.3, 0.0, 0.0, 0.0));
        predictions.insert(ActionId::Throttle, rv(0.2, 0.0, 0.0, 0.0));
        let estimator = MockEstimator::new(predictions);

        let pressure = pv(0.8, 0.0, 0.0);
        let state = no_active_state();
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

        // 완전 해소는 불가하지만, 최대 coverage를 위해 둘 다 선택해야 함
        let ids = command_ids(&cmds);
        assert!(
            ids.contains(&ActionId::SwitchHw),
            "best-effort: switch_hw should be included"
        );
        assert!(
            ids.contains(&ActionId::Throttle),
            "best-effort: throttle should be included"
        );
    }

    /// Warning mode이고 등록된 모든 액션이 lossy이면 빈 결과를 반환한다.
    #[test]
    fn test_empty_candidates() {
        let registry = make_registry(
            &[
                ("kv.evict_sliding", true, false),
                ("weight.skip", true, true),
            ],
            &[],
        );

        let mut predictions = HashMap::new();
        predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
        predictions.insert(ActionId::LayerSkip, rv(0.4, 0.0, 0.0, 0.0));
        let estimator = MockEstimator::new(predictions);

        let pressure = pv(0.5, 0.6, 0.0);
        let state = no_active_state();
        let qcf = HashMap::new();

        let cmds = ActionSelector::select(
            &registry,
            &estimator,
            &pressure,
            OperatingMode::Warning, // Warning → lossy 모두 제외 → 후보 없음
            &state,
            &qcf,
            1.0,
            &[],
            &[],
        );

        assert!(
            cmds.is_empty(),
            "should return empty when all candidates are lossy in Warning mode"
        );
    }

    /// pressure 크기에 따라 파라미터가 선형 보간된다.
    #[test]
    fn test_parametrize_proportional() {
        // kv_evict_sliding: keep_ratio range [0.3, 0.9]
        // memory pressure = 0.5 → value = 0.9 - 0.5*(0.9-0.3) = 0.9 - 0.3 = 0.6
        let registry = make_registry(&[("kv.evict_sliding", true, false)], &[]);

        let mut predictions = HashMap::new();
        predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.9, 0.0, 0.0));
        let estimator = MockEstimator::new(predictions);

        let pressure = pv(0.0, 0.5, 0.0);
        let state = no_active_state();
        let mut qcf = HashMap::new();
        qcf.insert(ActionId::KvEvictSliding, 1.0f32);

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

        assert_eq!(cmds.len(), 1);
        let cmd = &cmds[0];
        assert_eq!(cmd.action, ActionId::KvEvictSliding);

        if let Operation::Apply(params) = &cmd.operation {
            let keep_ratio = params.values.get("keep_ratio").copied().unwrap();
            let expected = 0.9 - 0.5 * (0.9 - 0.3); // = 0.6
            assert!(
                (keep_ratio - expected).abs() < 1e-5,
                "keep_ratio should be {expected}, got {keep_ratio}"
            );
        } else {
            panic!("expected Apply operation");
        }
    }

    /// pressure가 0이면 빈 결과를 반환한다.
    /// (빈 조합이 모든 제약을 충족하고 cost=0이 최소이므로 최적은 빈 조합)
    #[test]
    fn test_no_action_when_no_pressure() {
        let registry = make_registry(
            &[
                ("switch_hw", false, true),
                ("kv.evict_sliding", true, false),
            ],
            &[],
        );

        let mut predictions = HashMap::new();
        predictions.insert(ActionId::SwitchHw, rv(0.6, 0.0, 0.4, -0.2));
        predictions.insert(ActionId::KvEvictSliding, rv(0.0, 0.8, 0.0, 0.1));
        let estimator = MockEstimator::new(predictions);

        let pressure = pv(0.0, 0.0, 0.0); // pressure 없음
        let state = no_active_state();
        let qcf = HashMap::new();

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

        assert!(cmds.is_empty(), "no action needed when pressure is zero");
    }

    /// active 액션은 후보에서 제외된다.
    #[test]
    fn test_active_action_excluded() {
        let registry = make_registry(
            &[("switch_hw", false, true), ("throttle", false, true)],
            &[],
        );

        let mut predictions = HashMap::new();
        predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, 0.0));
        predictions.insert(ActionId::Throttle, rv(0.4, 0.0, 0.2, -0.3));
        let estimator = MockEstimator::new(predictions);

        // switch_hw가 이미 활성화된 상태 — FeatureVector는 ReliefEstimator 컨텍스트용,
        // 액션 필터링은 active_actions 파라미터로 직접 전달
        let mut state = FeatureVector::zeros();
        state.values[feature::ACTIVE_SWITCH_HW] = 1.0;

        let pressure = pv(0.5, 0.0, 0.0);
        let qcf = HashMap::new();
        // EngineStatus에서 직접 전달된 active_actions
        let active_actions = [ActionId::SwitchHw];

        let cmds = ActionSelector::select(
            &registry,
            &estimator,
            &pressure,
            OperatingMode::Warning,
            &state,
            &qcf,
            1.0,
            &active_actions,
            &[],
        );

        let ids = command_ids(&cmds);
        assert!(
            !ids.contains(&ActionId::SwitchHw),
            "already active action should not be selected again"
        );
    }

    /// available_actions가 비어있지 않으면 그 목록에 없는 액션은 후보에서 제외된다.
    #[test]
    fn test_available_actions_filters_candidates() {
        let registry = make_registry(
            &[
                ("switch_hw", false, true),
                ("throttle", false, true),
                ("kv.evict_h2o", true, false),
            ],
            &[],
        );

        let mut predictions = HashMap::new();
        predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, 0.0));
        predictions.insert(ActionId::Throttle, rv(0.4, 0.0, 0.3, 0.0));
        predictions.insert(ActionId::KvEvictH2o, rv(0.0, 0.9, 0.0, 0.0));
        let estimator = MockEstimator::new(predictions);

        let pressure = pv(0.6, 0.0, 0.0);
        let state = no_active_state();
        let qcf = HashMap::new();
        // Engine이 throttle만 실행 가능하다고 보고
        let available = [ActionId::Throttle];

        let cmds = ActionSelector::select(
            &registry,
            &estimator,
            &pressure,
            OperatingMode::Critical,
            &state,
            &qcf,
            1.0,
            &[],
            &available,
        );

        let ids = command_ids(&cmds);
        // switch_hw와 kv_evict_h2o는 available_actions에 없으므로 제외됨
        assert!(
            !ids.contains(&ActionId::SwitchHw),
            "switch_hw not in available_actions must be excluded"
        );
        assert!(
            !ids.contains(&ActionId::KvEvictH2o),
            "kv_evict_h2o not in available_actions must be excluded"
        );
        // throttle은 available_actions에 있으므로 선택될 수 있음
        assert!(
            ids.contains(&ActionId::Throttle),
            "throttle is in available_actions and should be selectable"
        );
    }

    /// available_actions가 비어있으면 모든 액션이 후보가 된다 (backward compat).
    #[test]
    fn test_empty_available_actions_no_filtering() {
        let registry = make_registry(
            &[("switch_hw", false, true), ("throttle", false, true)],
            &[],
        );

        let mut predictions = HashMap::new();
        predictions.insert(ActionId::SwitchHw, rv(0.8, 0.0, 0.5, 0.0));
        predictions.insert(ActionId::Throttle, rv(0.4, 0.0, 0.3, 0.0));
        let estimator = MockEstimator::new(predictions);

        let pressure = pv(0.6, 0.0, 0.0);
        let state = no_active_state();
        let qcf = HashMap::new();

        let cmds = ActionSelector::select(
            &registry,
            &estimator,
            &pressure,
            OperatingMode::Critical,
            &state,
            &qcf,
            1.0,
            &[],
            &[], // available_actions 비어있음 → 필터링 없음
        );

        // switch_hw가 더 많은 relief를 제공하므로 선택됨
        let ids = command_ids(&cmds);
        assert!(
            ids.contains(&ActionId::SwitchHw),
            "switch_hw should be selectable when available_actions is empty"
        );
    }
}
