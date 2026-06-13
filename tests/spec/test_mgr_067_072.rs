//! MGR-067 ~ MGR-072: ThresholdEvaluator Spec 테스트
//!
//! Ascending/Descending 방향의 레벨 상승/하강, hysteresis 기반 진동 방지,
//! 멀티 레벨 복귀, Emergency 비활성화를 검증한다.

use argus_manager::evaluator::{Direction, ThresholdEvaluator, Thresholds};
use argus_shared::Level;

fn ascending_thresholds() -> Thresholds {
    Thresholds {
        warning: 60.0,
        critical: 75.0,
        emergency: 85.0,
        hysteresis: 5.0,
    }
}

fn descending_thresholds() -> Thresholds {
    Thresholds {
        warning: 40.0,
        critical: 20.0,
        emergency: 10.0,
        hysteresis: 5.0,
    }
}

// ── MGR-067: Ascending 상승 경로 ────────────────────────────────────

/// MGR-067: Ascending 방향에서 Normal -> Warning -> Critical -> Emergency 순차 상승.
#[test]
fn test_mgr_067_ascending_escalation_path() {
    let mut e = ThresholdEvaluator::new(Direction::Ascending, ascending_thresholds());

    assert!(e.evaluate(50.0).is_none()); // Normal 유지
    assert_eq!(e.evaluate(65.0), Some(Level::Warning));
    assert_eq!(e.evaluate(76.0), Some(Level::Critical));
    assert_eq!(e.evaluate(90.0), Some(Level::Emergency));
}

// ── MGR-070: Ascending 레벨 스킵 ───────────────────────────────────

/// MGR-070: Ascending 방향에서 Normal -> Emergency 직행 가능.
#[test]
fn test_mgr_070_ascending_skip_to_emergency() {
    let mut e = ThresholdEvaluator::new(Direction::Ascending, ascending_thresholds());
    assert_eq!(e.evaluate(90.0), Some(Level::Emergency));
}

// ── MGR-071: Ascending hysteresis ───────────────────────────────────

/// MGR-071: Ascending 방향 hysteresis로 진동 방지.
/// Warning 상태에서 threshold - hysteresis 사이에 있으면 Warning 유지.
#[test]
fn test_mgr_071_ascending_hysteresis_prevents_oscillation() {
    let mut e = ThresholdEvaluator::new(Direction::Ascending, ascending_thresholds());

    e.evaluate(65.0); // -> Warning
    assert_eq!(e.level(), Level::Warning);

    // 58.0 >= (60.0 - 5.0 = 55.0) -> hysteresis zone -> Warning 유지
    assert!(e.evaluate(58.0).is_none());
    // 54.0 < 55.0 -> Normal로 복귀
    assert_eq!(e.evaluate(54.0), Some(Level::Normal));
}

/// MGR-071: Ascending Emergency에서 전 범위 복귀 (한 번에 Normal까지).
#[test]
fn test_mgr_071_ascending_multi_level_recovery() {
    let mut e = ThresholdEvaluator::new(Direction::Ascending, ascending_thresholds());

    e.evaluate(90.0); // -> Emergency
    assert_eq!(e.evaluate(40.0), Some(Level::Normal)); // 전 범위 복귀
}

/// MGR-071: Ascending hysteresis zone 내에서는 레벨 변경 없음.
#[test]
fn test_mgr_071_ascending_stay_in_hysteresis_zone() {
    let mut e = ThresholdEvaluator::new(Direction::Ascending, ascending_thresholds());

    e.evaluate(65.0); // -> Warning
    // 57.0 >= 55.0 -> hysteresis zone -> Warning 유지
    assert!(e.evaluate(57.0).is_none());
    assert_eq!(e.level(), Level::Warning);
}

// ── MGR-068: Descending 상승 경로 ──────────────────────────────────

/// MGR-068: Descending 방향에서 Normal -> Warning -> Critical -> Emergency 순차 상승.
#[test]
fn test_mgr_068_descending_escalation_path() {
    let mut e = ThresholdEvaluator::new(Direction::Descending, descending_thresholds());

    assert!(e.evaluate(50.0).is_none()); // Normal 유지
    assert_eq!(e.evaluate(35.0), Some(Level::Warning));
    assert_eq!(e.evaluate(15.0), Some(Level::Critical));
    assert_eq!(e.evaluate(5.0), Some(Level::Emergency));
}

// ── MGR-072: Descending 레벨 스킵 + hysteresis ─────────────────────

/// MGR-072: Descending 방향에서 Normal -> Emergency 직행 가능.
#[test]
fn test_mgr_072_descending_skip_to_emergency() {
    let mut e = ThresholdEvaluator::new(Direction::Descending, descending_thresholds());
    // 값 3.0 <= emergency(10.0) -> Emergency
    assert_eq!(e.evaluate(3.0), Some(Level::Emergency));
}

/// MGR-072: Descending 방향 hysteresis로 진동 방지.
#[test]
fn test_mgr_072_descending_hysteresis_prevents_oscillation() {
    let mut e = ThresholdEvaluator::new(Direction::Descending, descending_thresholds());

    e.evaluate(35.0); // -> Warning
    // 42.0 < (40.0 + 5.0 = 45.0) -> hysteresis zone -> Warning 유지
    assert!(e.evaluate(42.0).is_none());
    // 46.0 > 45.0 -> Normal 복귀
    assert_eq!(e.evaluate(46.0), Some(Level::Normal));
}

/// MGR-072: Descending Emergency에서 전 범위 복귀.
#[test]
fn test_mgr_072_descending_multi_level_recovery() {
    let mut e = ThresholdEvaluator::new(Direction::Descending, descending_thresholds());

    e.evaluate(5.0); // -> Emergency
    // 80.0 > (40.0 + 5.0 = 45.0) -> Normal
    assert_eq!(e.evaluate(80.0), Some(Level::Normal));
}

/// MGR-072: Descending 단계별 복귀.
#[test]
fn test_mgr_072_descending_step_recovery() {
    let mut e = ThresholdEvaluator::new(Direction::Descending, descending_thresholds());

    e.evaluate(5.0); // -> Emergency
    // 16.0 > (10.0 + 5.0 = 15.0) 이지만 < (20.0 + 5.0 = 25.0) -> Critical
    assert_eq!(e.evaluate(16.0), Some(Level::Critical));
    // 30.0 > 25.0 이지만 < 45.0 -> Warning
    assert_eq!(e.evaluate(30.0), Some(Level::Warning));
    // 50.0 > 45.0 -> Normal
    assert_eq!(e.evaluate(50.0), Some(Level::Normal));
}

// ── MGR-069: Emergency 레벨 비활성화 ────────────────────────────────

/// MGR-069: emergency를 f64::MAX로 설정하면 Emergency에 도달하지 않는다.
#[test]
fn test_mgr_069_no_emergency_level() {
    let mut e = ThresholdEvaluator::new(
        Direction::Ascending,
        Thresholds {
            warning: 70.0,
            critical: 90.0,
            emergency: f64::MAX,
            hysteresis: 5.0,
        },
    );

    assert_eq!(e.evaluate(99.0), Some(Level::Critical));
    assert_eq!(e.level(), Level::Critical); // Emergency 도달 불가
}
