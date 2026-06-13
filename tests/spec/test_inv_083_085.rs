//! INV-083: PI Controller output은 [0, 1] 범위 내.
//! INV-085: Normal 모드에서 액션 미발행.
//!
//! 원본:
//! - INV-083: spec/41-invariants.md, MGR-C06, pi_controller.rs clamp
//! - INV-085: spec/41-invariants.md, MGR-C10, pipeline.rs process_signal Normal guard

use argus_manager::pi_controller::{GainZone, PiController};
use argus_manager::pipeline::{HierarchicalPolicy, PolicyStrategy};
use argus_manager::types::OperatingMode;

use argus_shared::{ComputeReason, Level, RecommendedBackend, SystemSignal};

use super::helpers::fast_supervisory_config;

// ---------------------------------------------------------------------------
// INV-083: PI Controller output은 [0, 1] 범위 내
// ---------------------------------------------------------------------------

/// INV-083: 극단적으로 높은 Kp + Ki에서도 output이 1.0을 초과하지 않아야 한다.
#[test]
fn inv083_output_never_exceeds_one() {
    let mut ctrl = PiController::new(100.0, 50.0, 0.0, 1000.0);
    for _ in 0..100 {
        let out = ctrl.update(1.0, 10.0);
        assert!(out <= 1.0, "INV-083: PI output must be <= 1.0, got {}", out);
        assert!(out >= 0.0, "INV-083: PI output must be >= 0.0, got {}", out);
    }
}

/// INV-083: measurement < setpoint일 때 output이 0.0이어야 한다 (음수 불가).
#[test]
fn inv083_output_never_negative() {
    let mut ctrl = PiController::new(10.0, 5.0, 0.9, 10.0);
    // measurement = 0.1 < setpoint = 0.9 → error = 0
    let out = ctrl.update(0.1, 1.0);
    assert!(
        out >= 0.0,
        "INV-083: PI output must be >= 0.0 when below setpoint, got {}",
        out
    );
    assert!(
        out.abs() < 1e-6,
        "INV-083: PI output should be 0.0 when measurement < setpoint, got {}",
        out
    );
}

/// INV-083: gain scheduling이 적용된 상태에서도 output이 [0, 1] 범위 내.
#[test]
fn inv083_output_bounded_with_gain_scheduling() {
    let zones = vec![
        GainZone {
            above: 0.5,
            kp: 10.0,
        },
        GainZone {
            above: 0.8,
            kp: 50.0,
        },
        GainZone {
            above: 0.95,
            kp: 200.0,
        },
    ];
    let mut ctrl = PiController::new(1.0, 2.0, 0.3, 100.0).with_gain_zones(zones);

    // 극단 입력: measurement=1.0, 큰 dt
    for _ in 0..50 {
        let out = ctrl.update(1.0, 5.0);
        assert!(
            (0.0..=1.0).contains(&out),
            "INV-083: With gain scheduling, output must be in [0, 1], got {}",
            out
        );
    }
}

/// INV-083: setpoint=0.0, measurement=1.0으로 최대 error 상황.
#[test]
fn inv083_maximum_error_bounded() {
    let mut ctrl = PiController::new(5.0, 3.0, 0.0, 50.0);
    // 매 반복마다 error = 1.0, 적분이 빠르게 누적
    for i in 0..200 {
        let out = ctrl.update(1.0, 1.0);
        assert!(
            out <= 1.0 + f32::EPSILON,
            "INV-083: Iteration {}, output {} exceeds 1.0",
            i,
            out
        );
        assert!(
            out >= -f32::EPSILON,
            "INV-083: Iteration {}, output {} below 0.0",
            i,
            out
        );
    }
}

/// INV-083: dt=0일 때 적분 미누적, output은 P항만 반영.
#[test]
fn inv083_zero_dt_no_integral() {
    let mut ctrl = PiController::new(0.5, 10.0, 0.5, 100.0);
    let out = ctrl.update(0.8, 0.0);
    // error = 0.3, integral += 0 → output = 0.5 * 0.3 = 0.15
    assert!(
        (0.0..=1.0).contains(&out),
        "INV-083: With dt=0, output should be in [0, 1], got {}",
        out
    );
    let expected = 0.5 * 0.3;
    assert!(
        (out - expected).abs() < 1e-5,
        "INV-083: Expected P-only output {:.4}, got {:.4}",
        expected,
        out
    );
}

// ---------------------------------------------------------------------------
// INV-085: Normal 모드에서 액션 미발행
// ---------------------------------------------------------------------------

/// INV-085: pressure가 0이면 Normal 모드이고, 액션이 발행되지 않아야 한다.
#[test]
fn inv085_normal_mode_no_action_zero_pressure() {
    let mut config = argus_manager::config::PolicyConfig::default();
    config.supervisory = fast_supervisory_config();

    // 액션 등록
    config.actions.insert(
        "switch_hw".into(),
        argus_manager::config::ActionConfig {
            lossy: false,
            reversible: true,
            ..Default::default()
        },
    );
    config.actions.insert(
        "kv.evict_sliding".into(),
        argus_manager::config::ActionConfig {
            lossy: true,
            reversible: false,
            ..Default::default()
        },
    );

    let mut policy = HierarchicalPolicy::new(&config);
    assert_eq!(policy.mode(), OperatingMode::Normal);

    // 매우 낮은 pressure 신호 (memory 10% 사용 = 90% 여유)
    let signal = SystemSignal::MemoryPressure {
        level: Level::Normal,
        available_bytes: 9_000_000_000,
        total_bytes: 10_000_000_000,
        reclaim_target_bytes: 0,
    };

    let result = policy.process_signal(&signal);

    assert_eq!(
        policy.mode(),
        OperatingMode::Normal,
        "INV-085: Low pressure should keep Normal mode"
    );
    assert!(
        result.is_none(),
        "INV-085: Normal mode must not emit any directive"
    );
}

/// INV-085: Normal 모드에서 반복적으로 낮은 신호를 받아도 액션 미발행.
#[test]
fn inv085_normal_mode_repeated_low_signals() {
    let mut config = argus_manager::config::PolicyConfig::default();
    config.supervisory = fast_supervisory_config();
    config.actions.insert(
        "switch_hw".into(),
        argus_manager::config::ActionConfig {
            lossy: false,
            reversible: true,
            ..Default::default()
        },
    );

    let mut policy = HierarchicalPolicy::new(&config);

    // 10회 낮은 compute 신호 반복
    for i in 0..10 {
        let signal = SystemSignal::ComputeGuidance {
            level: Level::Normal,
            recommended_backend: RecommendedBackend::Any,
            reason: ComputeReason::Balanced,
            cpu_usage_pct: 20.0,
            gpu_usage_pct: 10.0,
        };
        let result = policy.process_signal(&signal);
        assert!(
            result.is_none(),
            "INV-085: Normal mode iteration {} must not emit directive",
            i
        );
    }
    assert_eq!(policy.mode(), OperatingMode::Normal);
}

/// INV-085: 다양한 도메인의 Normal-level 신호에서도 액션 미발행.
#[test]
fn inv085_normal_mode_multi_domain_signals() {
    let mut config = argus_manager::config::PolicyConfig::default();
    config.supervisory = fast_supervisory_config();
    config.actions.insert(
        "switch_hw".into(),
        argus_manager::config::ActionConfig {
            lossy: false,
            reversible: true,
            ..Default::default()
        },
    );
    config.actions.insert(
        "throttle".into(),
        argus_manager::config::ActionConfig {
            lossy: false,
            reversible: true,
            ..Default::default()
        },
    );

    let mut policy = HierarchicalPolicy::new(&config);

    // Memory: 정상
    let result_mem = policy.process_signal(&SystemSignal::MemoryPressure {
        level: Level::Normal,
        available_bytes: 8_000_000_000,
        total_bytes: 10_000_000_000,
        reclaim_target_bytes: 0,
    });
    assert!(
        result_mem.is_none(),
        "INV-085: Normal memory must not emit directive"
    );

    // Thermal: 정상 (30C)
    let result_thermal = policy.process_signal(&SystemSignal::ThermalAlert {
        level: Level::Normal,
        temperature_mc: 30_000,
        throttling_active: false,
        throttle_ratio: 0.0,
    });
    assert!(
        result_thermal.is_none(),
        "INV-085: Normal thermal must not emit directive"
    );

    // Compute: 정상 (20%)
    let result_compute = policy.process_signal(&SystemSignal::ComputeGuidance {
        level: Level::Normal,
        recommended_backend: RecommendedBackend::Any,
        reason: ComputeReason::Balanced,
        cpu_usage_pct: 20.0,
        gpu_usage_pct: 10.0,
    });
    assert!(
        result_compute.is_none(),
        "INV-085: Normal compute must not emit directive"
    );

    assert_eq!(policy.mode(), OperatingMode::Normal);
}

/// INV-085 대조: Warning 이상 모드에서는 액션이 발행될 수 있다.
#[test]
fn inv085_warning_mode_can_emit_actions() {
    let mut config = argus_manager::config::PolicyConfig::default();
    config.supervisory = fast_supervisory_config();
    // PI setpoint을 낮게 설정하여 Warning 전환을 쉽게 트리거
    config.pi_controller.memory_kp = 5.0;
    config.pi_controller.memory_setpoint = 0.3;
    config.actions.insert(
        "switch_hw".into(),
        argus_manager::config::ActionConfig {
            lossy: false,
            reversible: true,
            ..Default::default()
        },
    );

    let mut policy = HierarchicalPolicy::new(&config);

    // 높은 memory pressure (90% 사용)
    let signal = SystemSignal::MemoryPressure {
        level: Level::Critical,
        available_bytes: 1_000_000_000,
        total_bytes: 10_000_000_000,
        reclaim_target_bytes: 5_000_000_000,
    };

    // 여러 번 신호를 보내 Warning/Critical 진입 유도
    let mut any_directive = false;
    for _ in 0..5 {
        if let Some(_directive) = policy.process_signal(&signal) {
            any_directive = true;
        }
    }

    // Normal이 아닌 모드라면 directive가 발행될 수 있다
    let mode = policy.mode();
    if mode != OperatingMode::Normal {
        assert!(
            mode >= OperatingMode::Warning,
            "INV-085 contrast: Mode should be Warning or above, got {:?}",
            mode
        );
    }
    // 최소한 Warning/Critical 모드에 진입했는지 확인
    assert!(
        mode >= OperatingMode::Warning || any_directive,
        "INV-085 contrast: High pressure should trigger Warning/Critical mode or emit directive"
    );
}
