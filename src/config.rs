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
    /// Safe temperature baseline for thermal normalization (Celsius).
    #[serde(default = "default_temp_safe")]
    pub temp_safe_c: f32,

    /// Critical temperature ceiling for thermal normalization (Celsius).
    #[serde(default = "default_temp_critical")]
    pub temp_critical_c: f32,

    /// Trigger thresholds.
    #[serde(default)]
    pub trigger: TriggerConfig,

    /// DirectiveDeduplicator cooldown (seconds).
    /// cooldown이 경과하면 동일한 directive도 재방출하여 relief observation이 쌓이도록 한다.
    #[serde(default = "default_dedup_cooldown_secs")]
    pub dedup_cooldown_secs: f64,
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
            temp_safe_c: 35.0,
            temp_critical_c: 50.0,
            trigger: TriggerConfig::default(),
            dedup_cooldown_secs: 60.0,
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
}
