//! INV-030: can_act=false 일 때 integral 값은 변하지 않는다.
//! INV-031: integral in [0, integral_clamp] (항상 유지)

use argus_manager::pi_controller::PiController;

const EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// INV-030: can_act=false → integral 동결
// ---------------------------------------------------------------------------

/// can_act=false 상태에서 update를 여러 번 호출해도 출력(=integral 기반)이 변하지 않아야 한다.
#[test]
fn test_inv_030_can_act_false_freezes_integral() {
    // kp=0으로 비례항 제거 → 출력 = ki * integral
    let ki = 1.0_f32;
    let mut ctrl = PiController::new(0.0, ki, 0.5, 10.0);

    // integral을 어느 정도 쌓는다 (measurement=0.8, error=0.3)
    ctrl.update(0.8, 1.0); // integral += 0.3 * 1.0 = 0.3
    ctrl.update(0.8, 1.0); // integral += 0.3 * 1.0 = 0.6

    // 현재 integral 기반 출력 확인 (dt=0 → integral 추가 없음)
    let baseline = ctrl.update(0.8, 0.0);
    assert!(baseline > 0.0, "integral should be positive");

    // can_act = false로 전환
    ctrl.set_can_act(false);

    // 여러 번 update해도 출력 불변
    for _ in 0..10 {
        let out = ctrl.update(0.8, 1.0); // dt=1.0이어도 integral 동결
        assert!(
            (out - baseline).abs() < EPS,
            "INV-030: integral must not change when can_act=false. expected={baseline}, got={out}"
        );
    }
}

/// can_act=false → true로 재전환 시 다시 누적이 시작되어야 한다.
#[test]
fn test_inv_030_can_act_restored_resumes_accumulation() {
    let ki = 1.0_f32;
    let mut ctrl = PiController::new(0.0, ki, 0.5, 10.0);

    ctrl.update(0.8, 1.0);
    let before_freeze = ctrl.update(0.8, 0.0);

    ctrl.set_can_act(false);
    ctrl.update(0.8, 1.0); // 동결 → 변화 없음

    ctrl.set_can_act(true);
    let after_resume = ctrl.update(0.8, 1.0); // 다시 누적

    assert!(
        after_resume > before_freeze,
        "after re-enabling can_act, integral should grow: before={before_freeze}, after={after_resume}"
    );
}

/// can_act=false에서 measurement < setpoint 인 경우에도 integral이 감소하지 않아야 한다.
#[test]
fn test_inv_030_can_act_false_no_decrease_below_setpoint() {
    let ki = 1.0_f32;
    let mut ctrl = PiController::new(0.0, ki, 0.5, 10.0);

    // integral 쌓기
    ctrl.update(0.9, 2.0);
    let baseline = ctrl.update(0.9, 0.0);

    ctrl.set_can_act(false);

    // measurement < setpoint → error = 0 이므로 integral 변화 없어야 함
    let out = ctrl.update(0.1, 5.0);
    assert!(
        (out - baseline).abs() < EPS,
        "INV-030: integral must stay frozen even with low measurement. expected={baseline}, got={out}"
    );
}

// ---------------------------------------------------------------------------
// INV-031: integral in [0, integral_clamp]
// ---------------------------------------------------------------------------

/// integral은 절대 integral_clamp을 초과해서는 안 된다.
#[test]
fn test_inv_031_integral_clamped_to_upper_bound() {
    let clamp = 1.0_f32;
    let ki = 1.0_f32;
    let mut ctrl = PiController::new(0.0, ki, 0.0, clamp);

    // 큰 error * 긴 dt로 강제 과누적
    for _ in 0..100 {
        ctrl.update(1.0, 1.0);
    }
    // output = ki * integral_clamped → 최대 ki * clamp
    let out = ctrl.update(1.0, 0.0);
    let max_possible = (ki * clamp).clamp(0.0, 1.0);
    assert!(
        (out - max_possible).abs() < EPS,
        "INV-031: integral should be clamped to {clamp}. output={out}, expected_max={max_possible}"
    );
}

/// integral은 절대 0 미만이 되어서는 안 된다.
/// (measurement < setpoint 이면 error = 0이므로 integral은 감소 없음)
#[test]
fn test_inv_031_integral_never_negative() {
    let ki = 1.0_f32;
    let mut ctrl = PiController::new(0.0, ki, 0.5, 10.0);

    // measurement < setpoint만 반복 → error = 0 → integral 누적 없음
    for _ in 0..50 {
        let out = ctrl.update(0.0, 1.0);
        assert!(
            out >= 0.0,
            "INV-031: output (= ki * integral) must never be negative, got {out}"
        );
    }
}

/// 다양한 clamp 값에서 integral이 올바르게 제한되는지 검증.
#[test]
fn test_inv_031_various_clamp_values() {
    for &clamp in &[0.1, 0.5, 1.0, 5.0, 100.0] {
        let ki = 1.0_f32;
        let mut ctrl = PiController::new(0.0, ki, 0.0, clamp);

        for _ in 0..100 {
            ctrl.update(1.0, 1.0);
        }
        // output = ki * integral → ki * min(accumulated, clamp) → ki * clamp
        let out = ctrl.update(1.0, 0.0);
        // output 자체도 [0, 1] 범위로 clamp되므로 ki * integral_clamp와 1.0 중 작은 것
        let expected = (ki * clamp).clamp(0.0, 1.0);
        assert!(
            (out - expected).abs() < EPS,
            "INV-031: clamp={clamp} — expected output={expected}, got={out}"
        );
    }
}

/// integral은 단 한 번의 update에서도 범위를 벗어나지 않아야 한다 (edge case: 매우 큰 dt).
#[test]
fn test_inv_031_single_large_dt_still_clamped() {
    let clamp = 2.0_f32;
    let ki = 1.0_f32;
    let mut ctrl = PiController::new(0.0, ki, 0.0, clamp);

    // error=1.0 * dt=1000.0 → integral=1000 without clamp
    ctrl.update(1.0, 1000.0);
    let out = ctrl.update(1.0, 0.0);
    // integral은 clamp 이하, output은 [0,1]
    assert!(
        out <= 1.0,
        "INV-031: output must be <= 1.0 (clamped), got {out}"
    );
    // ki=1.0, clamp=2.0 → ki*clamp=2.0 → clamped to 1.0
    assert!(
        (out - 1.0).abs() < EPS,
        "INV-031: expected 1.0 (ki*clamp=2.0 clamped), got {out}"
    );
}
