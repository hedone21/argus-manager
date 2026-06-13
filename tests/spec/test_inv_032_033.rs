//! INV-032: 에스컬레이션은 즉시. Normal → Critical 직행 가능.
//! INV-033: 디에스컬레이션은 반드시 1단계씩. Critical → Normal 직행 불가.

use std::time::{Duration, Instant};

use argus_manager::config::SupervisoryConfig;
use argus_manager::supervisory::SupervisoryLayer;
use argus_manager::types::OperatingMode;

use super::helpers::{fast_supervisory_config, pv};

// ---------------------------------------------------------------------------
// INV-032: 에스컬레이션 즉시
// ---------------------------------------------------------------------------

/// Normal → Critical 직행이 가능해야 한다 (Warning 단계를 거치지 않음).
#[test]
fn test_inv_032_escalation_normal_to_critical_immediate() {
    let config = fast_supervisory_config();
    let mut layer = SupervisoryLayer::new(&config);
    assert_eq!(layer.mode(), OperatingMode::Normal);

    // critical_threshold(0.7) 이상의 peak → 즉시 Critical
    let mode = layer.evaluate(&pv(0.0, 0.0, 0.8));
    assert_eq!(
        mode,
        OperatingMode::Critical,
        "INV-032: Normal should jump directly to Critical when peak >= critical_threshold"
    );
}

/// Normal → Warning 전환도 즉시 이루어져야 한다.
#[test]
fn test_inv_032_escalation_normal_to_warning_immediate() {
    let config = fast_supervisory_config();
    let mut layer = SupervisoryLayer::new(&config);
    assert_eq!(layer.mode(), OperatingMode::Normal);

    // warning_threshold(0.4) <= peak < critical_threshold(0.7)
    let mode = layer.evaluate(&pv(0.5, 0.0, 0.0));
    assert_eq!(
        mode,
        OperatingMode::Warning,
        "INV-032: Normal should escalate to Warning immediately"
    );
}

/// Warning → Critical 전환도 즉시 이루어져야 한다.
#[test]
fn test_inv_032_escalation_warning_to_critical_immediate() {
    let config = fast_supervisory_config();
    let mut layer = SupervisoryLayer::new(&config);

    // 먼저 Warning 진입
    layer.evaluate(&pv(0.5, 0.0, 0.0));
    assert_eq!(layer.mode(), OperatingMode::Warning);

    // peak >= critical_threshold → 즉시 Critical
    let mode = layer.evaluate(&pv(0.8, 0.0, 0.0));
    assert_eq!(
        mode,
        OperatingMode::Critical,
        "INV-032: Warning should escalate to Critical immediately"
    );
}

/// 에스컬레이션에는 hold_time이 필요 없다 (긴 hold_time에서도 즉시 전환).
#[test]
fn test_inv_032_escalation_ignores_hold_time() {
    let config = SupervisoryConfig {
        hold_time_secs: 999.0, // 사실상 무한대
        ..fast_supervisory_config()
    };
    let mut layer = SupervisoryLayer::new(&config);

    // 즉시 Critical로 점프
    let mode = layer.evaluate(&pv(0.0, 0.8, 0.0));
    assert_eq!(
        mode,
        OperatingMode::Critical,
        "INV-032: escalation must be immediate regardless of hold_time"
    );
}

// ---------------------------------------------------------------------------
// INV-033: 디에스컬레이션 1단계씩
// ---------------------------------------------------------------------------

/// Critical → Normal 직행은 불가능해야 한다. 반드시 Warning을 거쳐야 한다.
#[test]
fn test_inv_033_deescalation_critical_to_normal_requires_warning_step() {
    let config = fast_supervisory_config();
    let mut layer = SupervisoryLayer::new(&config);
    let now = Instant::now();

    // Critical 진입
    layer.evaluate_at(&pv(0.8, 0.0, 0.0), now);
    assert_eq!(layer.mode(), OperatingMode::Critical);

    // 충분히 낮은 peak (warning_release 미만) + hold_time 경과
    let after_hold = now + Duration::from_millis(10);
    // 첫 번째 하강 평가 → stable_since 설정
    layer.evaluate_at(&pv(0.1, 0.1, 0.1), now + Duration::from_millis(1));
    // hold_time 경과 후 → Warning으로만 내려가야 함 (Normal 직행 불가)
    let mode = layer.evaluate_at(&pv(0.1, 0.1, 0.1), after_hold);
    assert_eq!(
        mode,
        OperatingMode::Warning,
        "INV-033: Critical must de-escalate to Warning, not directly to Normal"
    );

    // Warning에서 다시 hold_time 경과 후에야 Normal 도달
    let step2_start = after_hold + Duration::from_millis(1);
    layer.evaluate_at(&pv(0.1, 0.1, 0.1), step2_start);
    let step2_done = step2_start + Duration::from_millis(10);
    let mode2 = layer.evaluate_at(&pv(0.1, 0.1, 0.1), step2_done);
    assert_eq!(
        mode2,
        OperatingMode::Normal,
        "INV-033: after another hold_time in Warning, should reach Normal"
    );
}

/// Warning → Normal 하강에도 hold_time이 필요하다.
#[test]
fn test_inv_033_deescalation_warning_to_normal_needs_hold() {
    let config = SupervisoryConfig {
        hold_time_secs: 1000.0, // 매우 긴 hold_time
        ..fast_supervisory_config()
    };
    let mut layer = SupervisoryLayer::new(&config);

    // Warning 진입
    layer.evaluate(&pv(0.5, 0.0, 0.0));
    assert_eq!(layer.mode(), OperatingMode::Warning);

    // peak가 warning_release 미만이지만 hold_time 미충족
    let mode = layer.evaluate(&pv(0.1, 0.1, 0.1));
    assert_eq!(
        mode,
        OperatingMode::Warning,
        "INV-033: de-escalation requires hold_time to elapse"
    );
}

/// 하강 중 peak 재상승이 발생하면 stable_since가 리셋되어야 한다.
#[test]
fn test_inv_033_deescalation_resets_on_spike() {
    let config = fast_supervisory_config();
    let mut layer = SupervisoryLayer::new(&config);
    let now = Instant::now();

    // Critical 진입
    layer.evaluate_at(&pv(0.8, 0.0, 0.0), now);

    // 하강 시작
    layer.evaluate_at(&pv(0.1, 0.1, 0.1), now + Duration::from_millis(1));

    // 재상승 → stable_since 리셋
    layer.evaluate_at(&pv(0.8, 0.0, 0.0), now + Duration::from_millis(5));

    // hold_time이 이전 하강 시작 기준으로 경과했더라도 리셋으로 인해 Critical 유지
    let mode = layer.evaluate_at(&pv(0.1, 0.1, 0.1), now + Duration::from_millis(10));
    // 이 시점에서 stable_since가 방금 설정됨 → 아직 hold_time 미경과 → Critical 유지
    assert_eq!(
        mode,
        OperatingMode::Critical,
        "INV-033: spike during de-escalation should reset hold timer"
    );
}

/// 하강 과정의 전체 순서: Critical → Warning → Normal
#[test]
fn test_inv_033_full_stepwise_deescalation() {
    let config = fast_supervisory_config();
    let mut layer = SupervisoryLayer::new(&config);
    let now = Instant::now();

    // 1. Critical 진입
    layer.evaluate_at(&pv(0.9, 0.0, 0.0), now);
    assert_eq!(layer.mode(), OperatingMode::Critical);

    let low = pv(0.1, 0.1, 0.1); // peak = 0.1 < warning_release(0.25)

    // 2. Critical → Warning
    layer.evaluate_at(&low, now + Duration::from_millis(1));
    let mode1 = layer.evaluate_at(&low, now + Duration::from_millis(20));
    assert_eq!(mode1, OperatingMode::Warning, "step 1: Critical -> Warning");

    // 3. Warning → Normal
    layer.evaluate_at(&low, now + Duration::from_millis(21));
    let mode2 = layer.evaluate_at(&low, now + Duration::from_millis(40));
    assert_eq!(mode2, OperatingMode::Normal, "step 2: Warning -> Normal");
}
