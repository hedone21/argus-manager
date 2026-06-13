//! MGR-ALG-010 ~ MGR-ALG-014: PI Controller Spec 테스트
//!
//! PI Controller의 비례항, 적분항, anti-windup, gain scheduling을
//! public API를 통해 검증한다.

use argus_manager::pi_controller::{GainZone, PiController};

const EPS: f32 = 1e-6;

// ── MGR-ALG-010: PI 기본 동작 ──────────────────────────────────────────

/// MGR-ALG-010: measurement < setpoint이면 출력은 항상 0.
#[test]
fn test_mgr_alg_010_pi_zero_error() {
    let mut ctrl = PiController::new(1.5, 0.3, 0.7, 2.0);
    let out = ctrl.update(0.5, 0.1);
    assert!(out.abs() < EPS, "expected 0.0, got {out}");
}

/// MGR-ALG-010: ki=0, measurement > setpoint이면 output = kp * error (clamped).
#[test]
fn test_mgr_alg_010_pi_proportional_only() {
    let kp = 1.5_f32;
    let setpoint = 0.7_f32;
    let measurement = 0.9_f32;
    let mut ctrl = PiController::new(kp, 0.0, setpoint, 2.0);
    let out = ctrl.update(measurement, 0.1);
    let expected = (kp * (measurement - setpoint)).clamp(0.0, 1.0);
    assert!(
        (out - expected).abs() < EPS,
        "expected {expected}, got {out}"
    );
}

/// MGR-ALG-010: 출력은 항상 [0.0, 1.0] 범위에 clamp된다.
#[test]
fn test_mgr_alg_010_pi_output_clamped() {
    let mut ctrl = PiController::new(10.0, 5.0, 0.0, 100.0);
    let out = ctrl.update(1.0, 10.0);
    assert!(
        (0.0..=1.0).contains(&out),
        "output {out} should be in [0.0, 1.0]"
    );
}

/// MGR-ALG-010: 같은 error로 여러 번 update하면 적분 누적으로 출력이 점진 증가.
#[test]
fn test_mgr_alg_010_pi_integral_accumulation() {
    let mut ctrl = PiController::new(0.0, 0.5, 0.7, 10.0);
    let mut last = 0.0_f32;
    for _ in 0..5 {
        let out = ctrl.update(0.9, 0.1);
        assert!(
            out >= last,
            "output should be non-decreasing, got {out} after {last}"
        );
        last = out;
    }
    assert!(
        last > 0.0,
        "accumulated integral should yield positive output"
    );
}

/// MGR-ALG-010: 같은 error라도 짧은 dt vs 긴 dt에서 긴 dt 쪽이 더 높은 누적 출력.
#[test]
fn test_mgr_alg_010_pi_spike_vs_sustained() {
    let kp = 0.0_f32;
    let ki = 1.0_f32;
    let setpoint = 0.5_f32;
    let measurement = 0.8_f32;

    let mut ctrl_short = PiController::new(kp, ki, setpoint, 100.0);
    let mut ctrl_long = PiController::new(kp, ki, setpoint, 100.0);

    for _ in 0..5 {
        ctrl_short.update(measurement, 0.01); // 총 dt = 0.05
        ctrl_long.update(measurement, 1.0); // 총 dt = 5.0
    }

    let out_short = ctrl_short.update(measurement, 0.0);
    let out_long = ctrl_long.update(measurement, 0.0);
    assert!(
        out_long > out_short,
        "longer dt should yield higher output: short={out_short}, long={out_long}"
    );
}

// ── MGR-ALG-011: Gain Scheduling ─────────────────────────────────────

/// MGR-ALG-011: gain_zones가 비어있으면 기존 고정 Kp와 동일하게 동작.
#[test]
fn test_mgr_alg_011_gain_zones_empty_uses_base_kp() {
    let kp = 2.0_f32;
    let setpoint = 0.7_f32;
    let measurement = 0.9_f32;

    let mut ctrl_base = PiController::new(kp, 0.0, setpoint, 2.0);
    let mut ctrl_zones = PiController::new(kp, 0.0, setpoint, 2.0).with_gain_zones(vec![]);

    let out_base = ctrl_base.update(measurement, 0.1);
    let out_zones = ctrl_zones.update(measurement, 0.1);
    assert!(
        (out_base - out_zones).abs() < EPS,
        "empty gain_zones should behave identically: base={out_base}, zones={out_zones}"
    );
}

/// MGR-ALG-011: measurement가 zone의 above 미만이면 base Kp 사용.
#[test]
fn test_mgr_alg_011_gain_zones_low() {
    let base_kp = 1.0_f32;
    let setpoint = 0.5_f32;
    let zones = vec![GainZone {
        above: 0.8,
        kp: 5.0,
    }];
    let mut ctrl = PiController::new(base_kp, 0.0, setpoint, 2.0).with_gain_zones(zones);
    let out = ctrl.update(0.6, 0.0);
    let expected = (base_kp * (0.6 - setpoint)).clamp(0.0, 1.0);
    assert!(
        (out - expected).abs() < EPS,
        "below zone threshold should use base_kp: expected={expected}, got={out}"
    );
}

/// MGR-ALG-011: measurement가 mid zone에 있으면 해당 zone의 Kp 적용.
#[test]
fn test_mgr_alg_011_gain_zones_mid() {
    let base_kp = 1.0_f32;
    let setpoint = 0.5_f32;
    let zones = vec![
        GainZone {
            above: 0.8,
            kp: 3.0,
        },
        GainZone {
            above: 0.9,
            kp: 8.0,
        },
    ];
    let mut ctrl = PiController::new(base_kp, 0.0, setpoint, 2.0).with_gain_zones(zones);
    let measurement = 0.82_f32;
    let out = ctrl.update(measurement, 0.0);
    let expected = (3.0_f32 * (measurement - setpoint)).clamp(0.0, 1.0);
    assert!(
        (out - expected).abs() < EPS,
        "mid zone should use kp=3.0: expected={expected}, got={out}"
    );
}

/// MGR-ALG-011: measurement가 가장 높은 zone에 있으면 최고 Kp 적용.
#[test]
fn test_mgr_alg_011_gain_zones_high() {
    let base_kp = 1.0_f32;
    let setpoint = 0.5_f32;
    let zones = vec![
        GainZone {
            above: 0.8,
            kp: 3.0,
        },
        GainZone {
            above: 0.9,
            kp: 8.0,
        },
    ];
    let mut ctrl = PiController::new(base_kp, 0.0, setpoint, 2.0).with_gain_zones(zones);
    let measurement = 0.95_f32;
    let out = ctrl.update(measurement, 0.0);
    let expected = (8.0_f32 * (measurement - setpoint)).clamp(0.0, 1.0);
    assert!(
        (out - expected).abs() < EPS,
        "high zone should use kp=8.0: expected={expected}, got={out}"
    );
}

// ── MGR-ALG-012: Anti-Windup ─────────────────────────────────────────

/// MGR-ALG-012: can_act=false이면 integral이 동결된다.
#[test]
fn test_mgr_alg_012_pi_anti_windup_can_act_false() {
    let mut ctrl = PiController::new(0.0, 1.0, 0.7, 10.0);
    // 적분을 어느 정도 쌓는다
    ctrl.update(0.9, 0.1);
    ctrl.update(0.9, 0.1);
    let integral_before = ctrl.update(0.9, 0.0); // dt=0 → 추가 누적 없음

    // can_act=false → 적분 동결
    ctrl.set_can_act(false);
    let out_frozen = ctrl.update(0.9, 0.1);
    let out_frozen2 = ctrl.update(0.9, 0.1);
    assert!(
        (out_frozen - out_frozen2).abs() < EPS,
        "integral should be frozen when can_act=false"
    );
    assert!(
        (integral_before - out_frozen).abs() < EPS,
        "frozen output {out_frozen} should match pre-freeze {integral_before}"
    );
}

/// MGR-ALG-012: integral이 clamp 이상 올라가지 않아야 한다.
#[test]
fn test_mgr_alg_012_pi_anti_windup_clamp() {
    let clamp = 1.0_f32;
    let ki = 1.0_f32;
    let mut ctrl = PiController::new(0.0, ki, 0.0, clamp);
    for _ in 0..100 {
        ctrl.update(1.0, 1.0);
    }
    let out = ctrl.update(1.0, 0.0);
    let max_possible = (ki * clamp).clamp(0.0, 1.0);
    assert!(
        (out - max_possible).abs() < EPS,
        "expected {max_possible}, got {out}"
    );
}

/// MGR-ALG-012: reset_integral 호출 후 적분이 0이 되어야 한다.
#[test]
fn test_mgr_alg_012_pi_reset_integral() {
    let mut ctrl = PiController::new(0.0, 1.0, 0.5, 10.0);
    ctrl.update(0.9, 1.0);
    ctrl.update(0.9, 1.0);
    ctrl.reset_integral();
    let out = ctrl.update(0.9, 0.0);
    assert!(
        out.abs() < EPS,
        "after reset, output should be 0.0, got {out}"
    );
}
