use serde::Deserialize;

/// Top-level Manager configuration, loadable from TOML.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub manager: ManagerConfig,
    pub memory: Option<MemoryMonitorConfig>,
    pub thermal: Option<ThermalMonitorConfig>,
    pub compute: Option<ComputeMonitorConfig>,
    pub energy: Option<EnergyMonitorConfig>,
    pub external: Option<ExternalMonitorConfig>,
    /// Online adaptation settings for LuaPolicy.
    #[serde(default)]
    pub adaptation: AdaptationConfig,
    #[cfg(feature = "hierarchical")]
    pub policy: Option<PolicyConfig>,
}

impl Config {
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ManagerConfig {
    /// Default polling interval in milliseconds.
    pub poll_interval_ms: u64,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 1000,
        }
    }
}

/// Memory monitor configuration.
///
/// Thresholds are available memory percentage (descending: lower is worse).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MemoryMonitorConfig {
    pub enabled: bool,
    pub poll_interval_ms: Option<u64>,
    pub warning_pct: f64,
    pub critical_pct: f64,
    pub emergency_pct: f64,
    pub hysteresis_pct: f64,
}

impl Default for MemoryMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_ms: None,
            warning_pct: 40.0,
            critical_pct: 20.0,
            emergency_pct: 10.0,
            hysteresis_pct: 5.0,
        }
    }
}

/// Thermal monitor configuration.
///
/// Thresholds are in millidegrees Celsius (ascending: higher is worse).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ThermalMonitorConfig {
    pub enabled: bool,
    pub poll_interval_ms: Option<u64>,
    pub zone_types: Vec<String>,
    pub warning_mc: i32,
    pub critical_mc: i32,
    pub emergency_mc: i32,
    pub hysteresis_mc: i32,
}

impl Default for ThermalMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_ms: None,
            zone_types: Vec::new(),
            warning_mc: 60000,
            critical_mc: 75000,
            emergency_mc: 85000,
            hysteresis_mc: 5000,
        }
    }
}

/// Compute monitor configuration.
///
/// ComputeGuidance has no Emergency level (max: Critical).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ComputeMonitorConfig {
    pub enabled: bool,
    pub poll_interval_ms: Option<u64>,
    pub warning_pct: f64,
    pub critical_pct: f64,
    pub hysteresis_pct: f64,
    /// DEPRECATED: Use `gpu_backend = { kind = "custom_sysfs", path = "..." }` instead.
    /// 값이 있고 `gpu_backend`가 `Auto`면 내부적으로 `CustomSysfs`로 매핑된다.
    pub gpu_sysfs_path: Option<String>,
    /// GPU telemetry 백엔드 선택. `Auto`가 기본이며 실행 환경에서 자동 감지한다.
    pub gpu_backend: GpuBackend,
}

impl Default for ComputeMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_ms: None,
            warning_pct: 70.0,
            critical_pct: 90.0,
            hysteresis_pct: 5.0,
            gpu_sysfs_path: None,
            gpu_backend: GpuBackend::Auto,
        }
    }
}

/// GPU telemetry provider 선택지.
///
/// `Auto` — 런타임에 `/sys`를 스캔하여 Jetson → Adreno/Mali → Null 순 탐지.
/// `Null` — GPU 없음/미지원 (항상 None).
/// `Sysfs` — Adreno/Mali sysfs 후보 경로 폴백.
/// `CustomSysfs` — 명시된 sysfs 파일 하나만 util 소스로 사용.
/// `Jetson` — devfreq `cur_freq` 자동 탐지 + `tegrastats` 서브프로세스.
/// `JetsonExplicit` — 명시된 freq 파일 + tegrastats 바이너리.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GpuBackend {
    #[default]
    Auto,
    Null,
    Sysfs,
    CustomSysfs {
        path: String,
    },
    Jetson,
    JetsonExplicit {
        freq_path: String,
        #[serde(default)]
        tegrastats_bin: Option<String>,
    },
}

/// Energy monitor configuration.
///
/// Thresholds are battery percentage (descending: lower is worse).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EnergyMonitorConfig {
    pub enabled: bool,
    pub poll_interval_ms: Option<u64>,
    pub warning_pct: f64,
    pub critical_pct: f64,
    pub emergency_pct: f64,
    pub warning_power_budget_mw: u32,
    pub critical_power_budget_mw: u32,
    pub emergency_power_budget_mw: u32,
    pub ignore_when_charging: bool,
}

impl Default for EnergyMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_ms: None,
            warning_pct: 30.0,
            critical_pct: 15.0,
            emergency_pct: 5.0,
            warning_power_budget_mw: 3000,
            critical_power_budget_mw: 1500,
            emergency_power_budget_mw: 500,
            ignore_when_charging: true,
        }
    }
}

/// External monitor configuration for research/testing signal injection.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ExternalMonitorConfig {
    pub enabled: bool,
    /// Transport: "stdin" or "unix:<socket_path>".
    pub transport: String,
}

impl Default for ExternalMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            transport: "stdin".into(),
        }
    }
}

/// Configuration for LuaPolicy online adaptation (trigger, EWMA, relief defaults).
#[derive(Debug, Clone, Deserialize)]
pub struct AdaptationConfig {
    /// EWMA smoothing factor (default: 0.875 = 7/8, Jacobson TCP RTT).
    #[serde(default = "default_ewma_alpha")]
    pub ewma_alpha: f32,

    /// Path to save/load the learned relief table (empty = disabled).
    #[serde(default)]
    pub relief_table_path: String,

    /// Safe temperature baseline for thermal normalization (Celsius).
    #[serde(default = "default_temp_safe")]
    pub temp_safe_c: f32,

    /// Critical temperature ceiling for thermal normalization (Celsius).
    #[serde(default = "default_temp_critical")]
    pub temp_critical_c: f32,

    /// Trigger thresholds.
    #[serde(default)]
    pub trigger: TriggerConfig,

    /// Per-action default relief values [gpu, cpu, memory, thermal, latency, main_app_qos].
    #[serde(default)]
    pub default_relief: std::collections::HashMap<String, Vec<f32>>,

    /// DirectiveDeduplicator cooldown (seconds).
    /// cooldown이 경과하면 동일한 directive도 재방출하여 relief observation이 쌓이도록 한다.
    #[serde(default = "default_dedup_cooldown_secs")]
    pub dedup_cooldown_secs: f64,

    /// Relief table 자동 저장 인터벌 (초). 0이면 비활성화. 기본 300.0s.
    #[serde(default = "default_persist_interval_secs")]
    pub persist_interval_secs: f64,

    /// LinUCB UCB 탐색 가중치 (0 = 비활성, 기본 0.5).
    #[serde(default = "default_linucb_alpha")]
    pub linucb_alpha: f32,

    /// QCF quality penalty weight in DPP score (V_Q). 0 = 비활성. 기본 0.5.
    #[serde(default = "default_qcf_penalty_weight")]
    pub qcf_penalty_weight: f32,

    /// QCF cache TTL (초). 초과 시 stale로 판단해 재요청. 기본 5.0.
    #[serde(default = "default_qcf_stale_secs")]
    pub qcf_stale_secs: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TriggerConfig {
    #[serde(default = "default_tbt_enter")]
    pub tbt_enter: f64,
    #[serde(default = "default_tbt_exit")]
    pub tbt_exit: f64,
    #[serde(default = "default_tbt_warmup")]
    pub tbt_warmup_tokens: u32,
    #[serde(default = "default_mem_enter")]
    pub mem_enter: f64,
    #[serde(default = "default_mem_exit")]
    pub mem_exit: f64,
    #[serde(default = "default_temp_enter")]
    pub temp_enter: f64,
    #[serde(default = "default_temp_exit")]
    pub temp_exit: f64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.875,
            relief_table_path: String::new(),
            temp_safe_c: 35.0,
            temp_critical_c: 50.0,
            trigger: TriggerConfig::default(),
            default_relief: std::collections::HashMap::new(),
            dedup_cooldown_secs: 60.0,
            persist_interval_secs: 300.0,
            linucb_alpha: 0.5,
            qcf_penalty_weight: default_qcf_penalty_weight(),
            qcf_stale_secs: default_qcf_stale_secs(),
        }
    }
}

impl Default for TriggerConfig {
    fn default() -> Self {
        Self {
            tbt_enter: 0.30,
            tbt_exit: 0.10,
            tbt_warmup_tokens: 20,
            mem_enter: 0.80,
            mem_exit: 0.60,
            temp_enter: 0.70,
            temp_exit: 0.50,
        }
    }
}

fn default_ewma_alpha() -> f32 {
    0.875
}
fn default_temp_safe() -> f32 {
    35.0
}
fn default_temp_critical() -> f32 {
    50.0
}
fn default_tbt_enter() -> f64 {
    0.30
}
fn default_tbt_exit() -> f64 {
    0.10
}
fn default_tbt_warmup() -> u32 {
    20
}
fn default_mem_enter() -> f64 {
    0.80
}
fn default_mem_exit() -> f64 {
    0.60
}
fn default_temp_enter() -> f64 {
    0.70
}
fn default_temp_exit() -> f64 {
    0.50
}
fn default_dedup_cooldown_secs() -> f64 {
    60.0
}
fn default_persist_interval_secs() -> f64 {
    300.0
}
fn default_linucb_alpha() -> f32 {
    0.5
}
fn default_qcf_penalty_weight() -> f32 {
    0.5
}
fn default_qcf_stale_secs() -> f64 {
    5.0
}

#[cfg(feature = "hierarchical")]
mod hierarchical_config {
    use super::*;
    use crate::pi_controller::GainZone;
    use std::collections::HashMap;

    /// 계층형 정책 설정 (PI Controller + Supervisory + Selector + Relief Model)
    #[derive(Debug, Clone, Default, Deserialize)]
    #[serde(default)]
    pub struct PolicyConfig {
        pub pi_controller: PiControllerConfig,
        pub supervisory: SupervisoryConfig,
        pub selector: SelectorConfig,
        pub relief_model: ReliefModelConfig,
        pub actions: HashMap<String, ActionConfig>,
        pub exclusion_groups: HashMap<String, Vec<String>>,
    }

    /// PI Controller 설정
    #[derive(Debug, Clone, Deserialize)]
    #[serde(default)]
    pub struct PiControllerConfig {
        pub compute_kp: f32,
        pub compute_ki: f32,
        pub compute_setpoint: f32,
        pub memory_kp: f32,
        pub memory_ki: f32,
        pub memory_setpoint: f32,
        pub thermal_kp: f32,
        pub thermal_ki: f32,
        pub thermal_setpoint: f32,
        pub integral_clamp: f32,
        /// Memory 도메인의 gain scheduling 구간.
        /// 미설정 시 고정 memory_kp를 사용한다.
        #[serde(default)]
        pub memory_gain_zones: Vec<GainZone>,
    }

    impl Default for PiControllerConfig {
        fn default() -> Self {
            Self {
                compute_kp: 1.5,
                compute_ki: 0.3,
                compute_setpoint: 0.70,
                memory_kp: 2.0,
                memory_ki: 0.5,
                memory_setpoint: 0.75,
                thermal_kp: 1.0,
                thermal_ki: 0.2,
                thermal_setpoint: 0.80,
                integral_clamp: 2.0,
                memory_gain_zones: Vec::new(),
            }
        }
    }

    /// Supervisory 모드 전환 임계값 설정
    #[derive(Debug, Clone, Deserialize)]
    #[serde(default)]
    pub struct SupervisoryConfig {
        pub warning_threshold: f32,
        pub critical_threshold: f32,
        pub warning_release: f32,
        pub critical_release: f32,
        pub hold_time_secs: f32,
    }

    impl Default for SupervisoryConfig {
        fn default() -> Self {
            Self {
                warning_threshold: 0.4,
                critical_threshold: 0.7,
                warning_release: 0.25,
                critical_release: 0.50,
                hold_time_secs: 4.0,
            }
        }
    }

    /// Action Selector 설정
    #[derive(Debug, Clone, Deserialize)]
    #[serde(default)]
    pub struct SelectorConfig {
        pub latency_budget: f32,
        pub algorithm: String,
    }

    impl Default for SelectorConfig {
        fn default() -> Self {
            Self {
                latency_budget: 0.5,
                algorithm: "exhaustive".to_string(),
            }
        }
    }

    /// Relief Estimator 모델 설정
    #[derive(Debug, Clone, Deserialize)]
    #[serde(default)]
    pub struct ReliefModelConfig {
        pub forgetting_factor: f32,
        pub prior_weight: u32,
        pub storage_dir: String,
    }

    impl Default for ReliefModelConfig {
        fn default() -> Self {
            Self {
                forgetting_factor: 0.995,
                prior_weight: 5,
                storage_dir: "~/.argus/models".to_string(),
            }
        }
    }

    /// 액션별 메타데이터 설정
    #[derive(Debug, Clone, Deserialize)]
    #[serde(default)]
    pub struct ActionConfig {
        pub lossy: bool,
        pub reversible: bool,
        #[serde(default = "default_cost")]
        pub default_cost: f32,
    }

    impl Default for ActionConfig {
        fn default() -> Self {
            Self {
                lossy: false,
                reversible: false,
                default_cost: default_cost(),
            }
        }
    }

    fn default_cost() -> f32 {
        1.0
    }
}

#[cfg(feature = "hierarchical")]
pub use hierarchical_config::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_all_monitors_enabled() {
        let config = Config::default();
        assert_eq!(config.manager.poll_interval_ms, 1000);
        // Optional monitors are None by default
        assert!(config.memory.is_none());
        assert!(config.external.is_none());
    }

    #[test]
    fn parse_minimal_toml() {
        let toml_str = r#"
[manager]
poll_interval_ms = 500

[memory]
enabled = true
warning_pct = 35.0
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.manager.poll_interval_ms, 500);
        let mem = config.memory.unwrap();
        assert!(mem.enabled);
        assert_eq!(mem.warning_pct, 35.0);
        assert_eq!(mem.critical_pct, 20.0); // default
    }

    #[test]
    fn parse_external_config() {
        let toml_str = r#"
[external]
enabled = true
transport = "unix:/tmp/test.sock"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        let ext = config.external.unwrap();
        assert!(ext.enabled);
        assert_eq!(ext.transport, "unix:/tmp/test.sock");
    }

    #[test]
    fn parse_full_config() {
        let toml_str = r#"
[manager]
poll_interval_ms = 2000

[memory]
enabled = true
warning_pct = 40.0
critical_pct = 20.0
emergency_pct = 10.0
hysteresis_pct = 5.0

[thermal]
enabled = true
zone_types = ["x86_pkg_temp"]
warning_mc = 60000
critical_mc = 75000
emergency_mc = 85000
hysteresis_mc = 5000

[compute]
enabled = true
warning_pct = 70.0
critical_pct = 90.0

[energy]
enabled = false
ignore_when_charging = true

[external]
enabled = true
transport = "stdin"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.manager.poll_interval_ms, 2000);
        assert!(config.memory.unwrap().enabled);
        assert_eq!(
            config.thermal.unwrap().zone_types,
            vec!["x86_pkg_temp".to_string()]
        );
        assert!(!config.energy.unwrap().enabled);
        assert!(config.external.unwrap().enabled);
    }

    #[cfg(feature = "hierarchical")]
    mod hierarchical_tests {
        use super::super::*;

        #[test]
        fn parse_policy_config() {
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

        #[test]
        fn policy_config_defaults() {
            let policy = PolicyConfig::default();
            assert!((policy.pi_controller.compute_kp - 1.5).abs() < f32::EPSILON);
            assert!((policy.pi_controller.memory_kp - 2.0).abs() < f32::EPSILON);
            assert!((policy.supervisory.warning_threshold - 0.4).abs() < f32::EPSILON);
            assert!((policy.supervisory.hold_time_secs - 4.0).abs() < f32::EPSILON);
            assert_eq!(policy.selector.algorithm, "exhaustive");
            assert!((policy.relief_model.forgetting_factor - 0.995).abs() < f32::EPSILON);
            assert_eq!(policy.relief_model.prior_weight, 5);
        }

        #[test]
        fn config_policy_optional_none_by_default() {
            let config = Config::default();
            assert!(config.policy.is_none());
        }

        #[test]
        fn action_config_default_cost_fallback_is_one() {
            // default_cost 필드 없이 파싱 시 1.0으로 폴백되어야 한다
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

        #[test]
        fn action_config_explicit_default_cost_loaded() {
            // 명시적 default_cost 값이 정확히 로드되어야 한다
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

        #[test]
        fn action_config_default_impl_has_cost_one() {
            let cfg = ActionConfig::default();
            assert!((cfg.default_cost - 1.0).abs() < f32::EPSILON);
        }
    }
}
