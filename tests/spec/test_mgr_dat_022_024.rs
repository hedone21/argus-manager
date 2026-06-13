//! MGR-DAT-022 ~ MGR-DAT-024: Monitor Config 기본값 테스트
//!
//! 각 MonitorConfig 구조체의 필드 기본값이 스펙과 일치하는지 검증한다.
//! - MGR-DAT-022: MemoryMonitorConfig
//! - MGR-DAT-023: ThermalMonitorConfig
//! - MGR-DAT-024: ComputeMonitorConfig

use argus_manager::config::{ComputeMonitorConfig, MemoryMonitorConfig, ThermalMonitorConfig};

/// MGR-DAT-022: MemoryMonitorConfig 기본값
/// - enabled: true
/// - poll_interval_ms: None (상위 default 사용)
/// - warning_pct: 40.0 (Descending: 가용 메모리 비율)
/// - critical_pct: 20.0
/// - emergency_pct: 10.0
/// - hysteresis_pct: 5.0
#[test]
fn test_mgr_dat_022_memory_config_defaults() {
    let config = MemoryMonitorConfig::default();

    assert!(
        config.enabled,
        "Memory monitor should be enabled by default"
    );
    assert!(
        config.poll_interval_ms.is_none(),
        "poll_interval_ms should be None by default"
    );
    assert!(
        (config.warning_pct - 40.0).abs() < f64::EPSILON,
        "warning_pct should be 40.0, got {}",
        config.warning_pct
    );
    assert!(
        (config.critical_pct - 20.0).abs() < f64::EPSILON,
        "critical_pct should be 20.0, got {}",
        config.critical_pct
    );
    assert!(
        (config.emergency_pct - 10.0).abs() < f64::EPSILON,
        "emergency_pct should be 10.0, got {}",
        config.emergency_pct
    );
    assert!(
        (config.hysteresis_pct - 5.0).abs() < f64::EPSILON,
        "hysteresis_pct should be 5.0, got {}",
        config.hysteresis_pct
    );
}

/// MGR-DAT-023: ThermalMonitorConfig 기본값
/// - enabled: true
/// - warning_mc: 60000 (60C)
/// - critical_mc: 75000 (75C)
/// - emergency_mc: 85000 (85C)
/// - hysteresis_mc: 5000 (5C)
/// - zone_types: 빈 벡터 (모든 zone 모니터링)
#[test]
fn test_mgr_dat_023_thermal_config_defaults() {
    let config = ThermalMonitorConfig::default();

    assert!(
        config.enabled,
        "Thermal monitor should be enabled by default"
    );
    assert!(config.poll_interval_ms.is_none());
    assert_eq!(config.warning_mc, 60000, "warning_mc should be 60000");
    assert_eq!(config.critical_mc, 75000, "critical_mc should be 75000");
    assert_eq!(config.emergency_mc, 85000, "emergency_mc should be 85000");
    assert_eq!(config.hysteresis_mc, 5000, "hysteresis_mc should be 5000");
    assert!(
        config.zone_types.is_empty(),
        "zone_types should be empty by default"
    );
}

/// MGR-DAT-024: ComputeMonitorConfig 기본값
/// - enabled: true
/// - warning_pct: 70.0
/// - critical_pct: 90.0
/// - hysteresis_pct: 5.0
/// - Emergency level 없음 (ComputeGuidance는 최대 Critical)
#[test]
fn test_mgr_dat_024_compute_config_defaults() {
    let config = ComputeMonitorConfig::default();

    assert!(
        config.enabled,
        "Compute monitor should be enabled by default"
    );
    assert!(config.poll_interval_ms.is_none());
    assert!(
        (config.warning_pct - 70.0).abs() < f64::EPSILON,
        "warning_pct should be 70.0, got {}",
        config.warning_pct
    );
    assert!(
        (config.critical_pct - 90.0).abs() < f64::EPSILON,
        "critical_pct should be 90.0, got {}",
        config.critical_pct
    );
    assert!(
        (config.hysteresis_pct - 5.0).abs() < f64::EPSILON,
        "hysteresis_pct should be 5.0, got {}",
        config.hysteresis_pct
    );
}
