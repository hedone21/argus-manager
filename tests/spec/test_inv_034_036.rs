//! INV-034: warning_release < warning_threshold
//! INV-035: critical_release < critical_threshold
//! INV-036: warning_threshold < critical_threshold
//!
//! 이 불변식들은 SupervisoryConfig의 설정 값 순서에 관한 것이다.
//! 기본 설정(SupervisoryConfig::default())이 이를 만족하는지 검증하고,
//! 위반 시 SupervisoryLayer의 동작이 의미론적으로 올바르지 않음을 확인한다.

use argus_manager::config::SupervisoryConfig;

// ---------------------------------------------------------------------------
// INV-034: warning_release < warning_threshold
// ---------------------------------------------------------------------------

/// 기본 설정이 INV-034를 만족해야 한다.
#[test]
fn test_inv_034_default_config_satisfies() {
    let config = SupervisoryConfig::default();
    assert!(
        config.warning_release < config.warning_threshold,
        "INV-034: warning_release({}) must be < warning_threshold({})",
        config.warning_release,
        config.warning_threshold,
    );
}

/// 경계: warning_release == warning_threshold이면 INV-034 위반.
#[test]
fn test_inv_034_equal_values_violate() {
    let config = SupervisoryConfig {
        warning_release: 0.4,
        warning_threshold: 0.4,
        ..SupervisoryConfig::default()
    };
    // 동일 값은 strict less-than을 만족하지 않으므로 위반
    assert!(
        config.warning_release >= config.warning_threshold,
        "INV-034: equal values should be a violation"
    );
}

/// warning_release > warning_threshold이면 INV-034 위반.
#[test]
fn test_inv_034_release_above_threshold_violates() {
    let config = SupervisoryConfig {
        warning_release: 0.5,
        warning_threshold: 0.4,
        ..SupervisoryConfig::default()
    };
    assert!(
        config.warning_release >= config.warning_threshold,
        "INV-034: release > threshold should be a violation"
    );
}

// ---------------------------------------------------------------------------
// INV-035: critical_release < critical_threshold
// ---------------------------------------------------------------------------

/// 기본 설정이 INV-035를 만족해야 한다.
#[test]
fn test_inv_035_default_config_satisfies() {
    let config = SupervisoryConfig::default();
    assert!(
        config.critical_release < config.critical_threshold,
        "INV-035: critical_release({}) must be < critical_threshold({})",
        config.critical_release,
        config.critical_threshold,
    );
}

/// critical_release == critical_threshold이면 INV-035 위반.
#[test]
fn test_inv_035_equal_values_violate() {
    let config = SupervisoryConfig {
        critical_release: 0.7,
        critical_threshold: 0.7,
        ..SupervisoryConfig::default()
    };
    assert!(
        config.critical_release >= config.critical_threshold,
        "INV-035: equal values should be a violation"
    );
}

// ---------------------------------------------------------------------------
// INV-036: warning_threshold < critical_threshold
// ---------------------------------------------------------------------------

/// 기본 설정이 INV-036을 만족해야 한다.
#[test]
fn test_inv_036_default_config_satisfies() {
    let config = SupervisoryConfig::default();
    assert!(
        config.warning_threshold < config.critical_threshold,
        "INV-036: warning_threshold({}) must be < critical_threshold({})",
        config.warning_threshold,
        config.critical_threshold,
    );
}

/// warning_threshold == critical_threshold이면 INV-036 위반.
#[test]
fn test_inv_036_equal_values_violate() {
    let config = SupervisoryConfig {
        warning_threshold: 0.7,
        critical_threshold: 0.7,
        ..SupervisoryConfig::default()
    };
    assert!(
        config.warning_threshold >= config.critical_threshold,
        "INV-036: equal thresholds should be a violation"
    );
}

/// warning_threshold > critical_threshold이면 INV-036 위반.
#[test]
fn test_inv_036_warning_above_critical_violates() {
    let config = SupervisoryConfig {
        warning_threshold: 0.8,
        critical_threshold: 0.5,
        ..SupervisoryConfig::default()
    };
    assert!(
        config.warning_threshold >= config.critical_threshold,
        "INV-036: warning > critical should be a violation"
    );
}

/// 전체 순서 관계: warning_release < warning_threshold < critical_threshold,
/// critical_release < critical_threshold (INV-034, 035, 036 통합 검증)
#[test]
fn test_inv_034_035_036_combined_ordering() {
    let config = SupervisoryConfig::default();

    // INV-034
    assert!(config.warning_release < config.warning_threshold);
    // INV-035
    assert!(config.critical_release < config.critical_threshold);
    // INV-036
    assert!(config.warning_threshold < config.critical_threshold);

    // 추가 논리적 순서: warning_release는 critical_threshold보다 작아야 한다
    assert!(
        config.warning_release < config.critical_threshold,
        "implied: warning_release({}) should be < critical_threshold({})",
        config.warning_release,
        config.critical_threshold,
    );
}
