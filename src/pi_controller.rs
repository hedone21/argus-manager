use serde::Deserialize;

/// Gain scheduling의 구간 정의.
/// measurement가 `above` 이상일 때 `kp`를 적용한다.
#[derive(Debug, Clone, Deserialize)]
pub struct GainZone {
    pub above: f32,
    pub kp: f32,
}

/// 단일 도메인용 PI 컨트롤러.
///
/// 원시 측정값(0.0~1.0)을 연속적인 pressure intensity(0.0~1.0)로 변환한다.
/// - 비례항(P): 현재 setpoint 초과분에 비례
/// - 적분항(I): 초과 상태가 누적된 시간에 비례
/// - 단방향: measurement < setpoint 이면 출력 0 (완화 방향은 담당 안 함)
pub struct PiController {
    kp: f32,
    ki: f32,
    setpoint: f32,
    integral: f32,
    integral_clamp: f32,
    /// false 이면 적분을 동결한다 (anti-windup: 해소 가능한 액션이 없을 때).
    can_act: bool,
    /// Gain scheduling 구간 목록. 비어있으면 고정 Kp를 사용한다.
    gain_zones: Vec<GainZone>,
}

impl PiController {
    pub fn new(kp: f32, ki: f32, setpoint: f32, integral_clamp: f32) -> Self {
        Self {
            kp,
            ki,
            setpoint,
            integral: 0.0,
            integral_clamp,
            can_act: true,
            gain_zones: Vec::new(),
        }
    }

    /// gain_zones를 설정하는 builder 메서드.
    pub fn with_gain_zones(mut self, zones: Vec<GainZone>) -> Self {
        self.gain_zones = zones;
        self
    }

    /// measurement에 따라 적용할 Kp를 결정한다.
    /// gain_zones가 비어있으면 고정 Kp를 반환한다.
    fn effective_kp(&self, measurement: f32) -> f32 {
        for zone in self.gain_zones.iter().rev() {
            if measurement >= zone.above {
                return zone.kp;
            }
        }
        self.kp
    }

    /// pressure intensity를 계산한다.
    ///
    /// `dt`: 마지막 호출 이후 경과 시간(초). 적분항 누적에 사용된다.
    pub fn update(&mut self, measurement: f32, dt: f32) -> f32 {
        let error = (measurement - self.setpoint).max(0.0);

        if self.can_act {
            self.integral = (self.integral + error * dt).clamp(0.0, self.integral_clamp);
        }

        let kp = self.effective_kp(measurement);
        (kp * error + self.ki * self.integral).clamp(0.0, 1.0)
    }

    /// 해소 가능한 액션 존재 여부를 설정한다.
    ///
    /// false 이면 적분 누적이 동결되어 windup을 방지한다.
    pub fn set_can_act(&mut self, can_act: bool) {
        self.can_act = can_act;
    }

    /// 적분항을 0으로 리셋한다.
    pub fn reset_integral(&mut self) {
        self.integral = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    /// measurement < setpoint 이면 출력은 항상 0.
    #[test]
    fn test_pi_zero_error() {
        let mut ctrl = PiController::new(1.5, 0.3, 0.7, 2.0);
        let out = ctrl.update(0.5, 0.1);
        assert!(out.abs() < EPS, "expected 0.0, got {out}");
    }

    /// ki=0, measurement > setpoint → output = kp * error (clamp 적용).
    #[test]
    fn test_pi_proportional_only() {
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

    /// 같은 error로 여러 번 update → 적분 누적으로 출력이 점진 증가.
    #[test]
    fn test_pi_integral_accumulation() {
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

    /// can_act=false 이면 integral이 동결되어야 한다.
    #[test]
    fn test_pi_anti_windup_can_act_false() {
        let mut ctrl = PiController::new(0.0, 1.0, 0.7, 10.0);
        // 먼저 적분을 어느 정도 쌓는다.
        ctrl.update(0.9, 0.1);
        ctrl.update(0.9, 0.1);
        let integral_before = {
            // update 한 번 더 해서 현재 출력을 기록한다.
            let out = ctrl.update(0.9, 0.0);
            out // ki=1.0, kp=0 → output = ki * integral
        };

        // can_act = false 로 전환 후 여러 번 호출해도 출력 변화 없어야 한다.
        ctrl.set_can_act(false);
        let out_frozen = ctrl.update(0.9, 0.1);
        let out_frozen2 = ctrl.update(0.9, 0.1);
        assert!(
            (out_frozen - out_frozen2).abs() < EPS,
            "integral should be frozen when can_act=false"
        );
        // 동결 전의 마지막 출력과 동일해야 한다.
        assert!(
            (integral_before - out_frozen).abs() < EPS,
            "frozen output {out_frozen} should match pre-freeze {integral_before}"
        );
    }

    /// integral이 clamp 이상 올라가지 않아야 한다.
    #[test]
    fn test_pi_anti_windup_clamp() {
        let clamp = 1.0_f32;
        let ki = 1.0_f32;
        let mut ctrl = PiController::new(0.0, ki, 0.0, clamp);
        // 큰 error * 긴 dt 로 강제 누적
        for _ in 0..100 {
            ctrl.update(1.0, 1.0);
        }
        // output = ki * integral_clamped → ki * clamp
        let out = ctrl.update(1.0, 0.0);
        let max_possible = (ki * clamp).clamp(0.0, 1.0);
        assert!(
            (out - max_possible).abs() < EPS,
            "expected {max_possible}, got {out}"
        );
    }

    /// 결과가 항상 [0.0, 1.0] 범위 안에 있어야 한다.
    #[test]
    fn test_pi_output_clamped() {
        let mut ctrl = PiController::new(10.0, 5.0, 0.0, 100.0);
        let out = ctrl.update(1.0, 10.0);
        assert!(
            (0.0..=1.0).contains(&out),
            "output {out} should be in [0.0, 1.0]"
        );
    }

    /// 같은 error 값이라도 짧은 dt vs 긴 dt → 긴 dt 쪽이 더 높은 누적 출력.
    #[test]
    fn test_pi_spike_vs_sustained() {
        let kp = 0.0_f32;
        let ki = 1.0_f32;
        let setpoint = 0.5_f32;
        let measurement = 0.8_f32;

        let mut ctrl_short = PiController::new(kp, ki, setpoint, 100.0);
        let mut ctrl_long = PiController::new(kp, ki, setpoint, 100.0);

        // 같은 횟수로 update, 총 dt가 다르다.
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

    /// reset_integral 호출 후 적분이 0이 되어야 한다.
    #[test]
    fn test_pi_reset_integral() {
        let mut ctrl = PiController::new(0.0, 1.0, 0.5, 10.0);
        // 적분 누적
        ctrl.update(0.9, 1.0);
        ctrl.update(0.9, 1.0);
        // 리셋 후 출력이 0이어야 한다.
        ctrl.reset_integral();
        let out = ctrl.update(0.9, 0.0); // dt=0 이므로 적분 추가 없음
        assert!(
            out.abs() < EPS,
            "after reset, output should be 0.0, got {out}"
        );
    }

    /// gain_zones가 비어있으면 기존 고정 Kp와 동일하게 동작해야 한다.
    #[test]
    fn test_gain_zones_empty_uses_base_kp() {
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

    /// measurement가 첫 zone의 above 미만이면 base Kp를 사용해야 한다.
    #[test]
    fn test_gain_zones_low_measurement_uses_base() {
        let base_kp = 1.0_f32;
        let setpoint = 0.5_f32;
        // measurement=0.6, zone.above=0.8 → base Kp 적용
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

    /// measurement가 mid zone에 있으면 해당 zone의 Kp를 적용해야 한다.
    #[test]
    fn test_gain_zones_mid_zone() {
        let base_kp = 1.0_f32;
        let setpoint = 0.5_f32;
        // measurement=0.82 — zone1(above=0.8, kp=3.0) 적용, zone2(above=0.9, kp=8.0) 미적용
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

    /// measurement가 가장 높은 zone에 있으면 최고 Kp를 적용해야 한다.
    #[test]
    fn test_gain_zones_high_zone() {
        let base_kp = 1.0_f32;
        let setpoint = 0.5_f32;
        // measurement=0.95 — zone2(above=0.9, kp=8.0) 적용
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

    /// 90% 메모리 사용 + Kp=10 → 출력이 Critical threshold(0.70) 이상이어야 한다.
    #[test]
    fn test_gain_zones_critical_reaches_critical_threshold() {
        // memory_setpoint=0.75, measurement=0.90, high zone kp=10.0
        let setpoint = 0.75_f32;
        let zones = vec![
            GainZone {
                above: 0.80,
                kp: 4.0,
            },
            GainZone {
                above: 0.88,
                kp: 10.0,
            },
        ];
        let mut ctrl = PiController::new(2.0, 0.0, setpoint, 2.0).with_gain_zones(zones);
        let out = ctrl.update(0.90, 0.0);
        let critical_threshold = 0.70_f32;
        assert!(
            out >= critical_threshold,
            "90% memory with kp=10 should reach critical threshold {critical_threshold}, got {out}"
        );
    }
}
