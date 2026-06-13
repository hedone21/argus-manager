use std::collections::HashMap;
use std::io;
use std::path::Path;

use argus_manager::action_registry::ActionRegistry;
use argus_manager::config::{ActionConfig, PolicyConfig, SupervisoryConfig};
use argus_manager::relief::ReliefEstimator;
use argus_manager::types::{ActionId, FeatureVector, PressureVector, ReliefVector};

/// 테스트 전용 SupervisoryConfig. hold_time이 매우 짧아 테스트에서 즉시 하강 가능.
pub fn fast_supervisory_config() -> SupervisoryConfig {
    SupervisoryConfig {
        warning_threshold: 0.4,
        critical_threshold: 0.7,
        warning_release: 0.25,
        critical_release: 0.50,
        hold_time_secs: 0.001,
    }
}

/// PressureVector 생성 헬퍼
pub fn pv(compute: f32, memory: f32, thermal: f32) -> PressureVector {
    PressureVector {
        compute,
        memory,
        thermal,
    }
}

/// ReliefVector 생성 헬퍼
pub fn rv(compute: f32, memory: f32, thermal: f32, latency: f32) -> ReliefVector {
    ReliefVector {
        compute,
        memory,
        thermal,
        latency,
    }
}

/// FeatureVector::zeros() 헬퍼
pub fn no_state() -> FeatureVector {
    FeatureVector::zeros()
}

/// ActionRegistry 생성 헬퍼
pub fn make_registry(
    actions: &[(&str, bool, bool)],
    exclusion_groups: &[(&str, &[&str])],
) -> ActionRegistry {
    let config = make_policy_config(actions, exclusion_groups);
    ActionRegistry::from_config(&config)
}

pub fn make_policy_config(
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

/// Mock ReliefEstimator
pub struct MockEstimator {
    pub predictions: HashMap<ActionId, ReliefVector>,
}

impl MockEstimator {
    pub fn new(predictions: HashMap<ActionId, ReliefVector>) -> Self {
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
