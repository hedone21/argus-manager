//! INV-050: 관찰 relief의 latency 차원은 항상 0.0이다.
//!
//! PressureVector 차감으로 관찰 relief를 계산할 때 (before - after),
//! latency 차원이 0.0이어야 한다.
//! 이는 PressureVector가 3차원(compute, memory, thermal)이고
//! Sub 구현이 latency를 항상 0.0으로 설정하기 때문이다.

use argus_manager::types::PressureVector;

/// PressureVector 차감 결과의 latency는 항상 0.0이다.
#[test]
fn test_inv_050_observed_relief_latency_is_zero() {
    let before = PressureVector {
        compute: 0.8,
        memory: 0.6,
        thermal: 0.5,
    };
    let after = PressureVector {
        compute: 0.3,
        memory: 0.4,
        thermal: 0.5,
    };

    let relief = before - after;
    assert!(
        relief.latency.abs() < f32::EPSILON,
        "INV-050: observed relief latency must be 0.0, got {}",
        relief.latency,
    );
}

/// 다양한 PressureVector 값의 차감에서 latency가 항상 0.0이다.
#[test]
fn test_inv_050_latency_zero_various_values() {
    let cases = [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((0.5, 0.3, 0.7), (0.1, 0.1, 0.1)),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), // negative relief
        ((0.99, 0.01, 0.5), (0.5, 0.5, 0.5)),
    ];

    for (before, after) in &cases {
        let pv_before = PressureVector {
            compute: before.0,
            memory: before.1,
            thermal: before.2,
        };
        let pv_after = PressureVector {
            compute: after.0,
            memory: after.1,
            thermal: after.2,
        };

        let relief = pv_before - pv_after;
        assert!(
            relief.latency.abs() < f32::EPSILON,
            "INV-050: latency must be 0.0 for before={:?}, after={:?}, got {}",
            before,
            after,
            relief.latency,
        );
    }
}

/// ReliefVector의 compute/memory/thermal는 정확히 차이값이어야 한다 (latency만 0.0).
#[test]
fn test_inv_050_other_dimensions_are_correct_difference() {
    let before = PressureVector {
        compute: 0.8,
        memory: 0.6,
        thermal: 0.4,
    };
    let after = PressureVector {
        compute: 0.3,
        memory: 0.2,
        thermal: 0.1,
    };

    let relief = before - after;

    let eps = f32::EPSILON;
    assert!((relief.compute - 0.5).abs() < eps);
    assert!((relief.memory - 0.4).abs() < eps);
    assert!((relief.thermal - 0.3).abs() < eps);
    assert!(relief.latency.abs() < eps, "INV-050: latency must be 0.0");
}
