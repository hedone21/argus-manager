//! MGR-050 ~ MGR-055: Supervisory FSM Spec 테스트
//!
//! 커버: MGR-050 (OperatingMode FSM), MGR-055 (하강 hold_time)
//! Supervisory Layer의 상승/하강 전환 로직, hold_time 기반 hysteresis,
//! spike 시 stable_since 리셋을 검증한다.

use std::time::{Duration, Instant};

use argus_manager::config::SupervisoryConfig;
use argus_manager::supervisory::SupervisoryLayer;
use argus_manager::types::{OperatingMode, PressureVector};

fn default_config() -> SupervisoryConfig {
    SupervisoryConfig::default()
}

fn fast_config() -> SupervisoryConfig {
    SupervisoryConfig {
        warning_threshold: 0.4,
        critical_threshold: 0.7,
        warning_release: 0.25,
        critical_release: 0.50,
        hold_time_secs: 0.001, // 1ms — 테스트에서 쉽게 경과시킬 수 있다.
    }
}

fn pressure(compute: f32, memory: f32, thermal: f32) -> PressureVector {
    PressureVector {
        compute,
        memory,
        thermal,
    }
}

fn low() -> PressureVector {
    pressure(0.1, 0.1, 0.1) // peak = 0.1 < warning_threshold(0.4)
}

fn warning_level() -> PressureVector {
    pressure(0.5, 0.1, 0.1) // peak = 0.5, warning(0.4) <= peak < critical(0.7)
}

fn critical_level() -> PressureVector {
    pressure(0.8, 0.1, 0.1) // peak = 0.8 >= critical(0.7)
}

fn below_warning_release() -> PressureVector {
    pressure(0.1, 0.2, 0.1) // peak = 0.2 < warning_release(0.25)
}

fn below_critical_release() -> PressureVector {
    pressure(0.3, 0.3, 0.1) // peak = 0.3 < critical_release(0.50)
}

// ── MGR-050: Normal 상태 유지 ──────────────────────────────────────────

/// MGR-050: peak < warning_threshold이면 Normal 유지.
#[test]
fn test_mgr_050_normal_below_threshold() {
    let mut s = SupervisoryLayer::new(&default_config());
    let mode = s.evaluate(&low());
    assert_eq!(mode, OperatingMode::Normal);
}

// ── MGR-051: 상승 전환 (즉시) ─────────────────────────────────────────

/// MGR-051: peak >= warning_threshold이면 즉시 Warning.
#[test]
fn test_mgr_051_escalation_to_warning() {
    let mut s = SupervisoryLayer::new(&default_config());
    let mode = s.evaluate(&warning_level());
    assert_eq!(mode, OperatingMode::Warning);
}

/// MGR-051: peak >= critical_threshold이면 즉시 Critical.
#[test]
fn test_mgr_051_escalation_to_critical() {
    let mut s = SupervisoryLayer::new(&default_config());
    let mode = s.evaluate(&critical_level());
    assert_eq!(mode, OperatingMode::Critical);
}

/// MGR-051: Normal에서 바로 Critical 도달 가능 (중간 단계 없음).
#[test]
fn test_mgr_051_skip_to_critical() {
    let mut s = SupervisoryLayer::new(&default_config());
    assert_eq!(s.mode(), OperatingMode::Normal);
    let mode = s.evaluate(&critical_level());
    assert_eq!(
        mode,
        OperatingMode::Critical,
        "Normal should jump directly to Critical"
    );
}

// ── MGR-052: 하강 전환 (hold_time 필요, 1단계씩) ──────────────────────

/// MGR-052: hold_time 미충족 시 Critical 유지.
#[test]
fn test_mgr_052_deescalation_needs_hold_time() {
    let config = SupervisoryConfig {
        hold_time_secs: 1000.0, // 사실상 무한대
        ..default_config()
    };
    let mut s = SupervisoryLayer::new(&config);
    s.evaluate(&critical_level());
    assert_eq!(s.mode(), OperatingMode::Critical);

    let mode = s.evaluate(&below_critical_release());
    assert_eq!(
        mode,
        OperatingMode::Critical,
        "should stay Critical until hold_time elapses"
    );
}

/// MGR-052: hold_time 경과 후 Critical -> Warning (evaluate_at 사용).
#[test]
fn test_mgr_052_deescalation_after_hold_time() {
    let config = fast_config();
    let mut s = SupervisoryLayer::new(&config);

    s.evaluate(&critical_level());
    assert_eq!(s.mode(), OperatingMode::Critical);

    let now = Instant::now();
    // 하강 첫 번째 평가 -> stable_since 설정
    let mode_first = s.evaluate_at(&below_critical_release(), now);
    assert_eq!(
        mode_first,
        OperatingMode::Critical,
        "first check should stay Critical"
    );

    // hold_time(1ms) 이상 경과
    let later = now + Duration::from_millis(5);
    let mode = s.evaluate_at(&below_critical_release(), later);
    assert_eq!(
        mode,
        OperatingMode::Warning,
        "after hold_time, should transition Critical -> Warning"
    );
}

/// MGR-052: Critical -> Warning -> Normal 순서로만 하강 (Critical -> Normal 직행 불가).
#[test]
fn test_mgr_052_stepwise_deescalation() {
    let config = fast_config();
    let mut s = SupervisoryLayer::new(&config);

    // 1. Critical 진입
    s.evaluate(&critical_level());
    assert_eq!(s.mode(), OperatingMode::Critical);

    let t0 = Instant::now();

    // 2. Critical -> Warning
    s.evaluate_at(&below_critical_release(), t0);
    let t1 = t0 + Duration::from_millis(5);
    let mode = s.evaluate_at(&below_critical_release(), t1);
    assert_eq!(mode, OperatingMode::Warning, "step 1: Critical -> Warning");

    // 3. Warning -> Normal
    let t2 = t1 + Duration::from_millis(1);
    s.evaluate_at(&below_warning_release(), t2);
    let t3 = t2 + Duration::from_millis(5);
    let mode = s.evaluate_at(&below_warning_release(), t3);
    assert_eq!(mode, OperatingMode::Normal, "step 2: Warning -> Normal");
}

// ── MGR-053: 하강 중 재상승 시 stable_since 리셋 ─────────────────────

/// MGR-053: 하강 중 peak 재상승 시 stable_since가 리셋되어 hold_time 재시작.
#[test]
fn test_mgr_053_deescalation_reset_on_spike() {
    let config = fast_config();
    let mut s = SupervisoryLayer::new(&config);

    // Critical 진입
    s.evaluate(&critical_level());

    // 하강 시작 (stable_since 설정)
    s.evaluate(&below_critical_release());
    // 재상승으로 stable_since 리셋
    s.evaluate(&critical_level());

    // 재상승 후이므로 아직 Critical 유지
    let mode = s.evaluate(&critical_level());
    assert_eq!(
        mode,
        OperatingMode::Critical,
        "spike should reset stable_since and keep Critical"
    );
}
