//! Phase 3 단위 테스트: VirtualClock + Signal derive + Noise.
//!
//! 테스트 목록 (15개):
//! 1.  test_clock_advances_monotonically
//! 2.  test_clock_drain_returns_events_in_time_order
//! 3.  test_clock_schedule_periodic
//! 4.  test_clock_fires_exactly_at_sub_tick_boundary
//! 5.  test_noise_reproducibility
//! 6.  test_noise_disabled_returns_zero_sigma_equivalent
//! 7.  test_noise_independent_streams
//! 8.  test_heartbeat_roundtrip_preserves_state
//! 9.  test_heartbeat_self_cpu_pct_clamped
//! 10. test_derive_memory_signal_maps_level_correctly
//! 11. test_derive_thermal_signal_reports_throttling
//! 12. test_derive_compute_recommends_cpu_under_high_gpu_temp
//! 13. test_signal_polling_cadence_respects_config
//! 14. test_noise_injects_into_heartbeat_throughput
//! 15. test_physics_step_signature_accepts_virtual_clock

use std::path::PathBuf;
use std::time::Duration;

use crate::common::sim::{
    clock::{EventKind, VirtualClock},
    config::{ScenarioConfig, load_scenario},
    noise::NoiseRng,
    physics::step,
    signal::{derive_compute_signal_pub, derive_memory_signal_pub, derive_thermal_signal_pub},
    state::{EngineStateModel, PhysicalState},
};
use argus_manager::signal::{Level, RecommendedBackend, SystemSignal};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sim")
}

fn load_baseline() -> ScenarioConfig {
    load_scenario(fixtures_dir().join("baseline.yaml")).expect("baseline.yaml should load")
}

fn make_state_and_engine(cfg: &ScenarioConfig) -> (PhysicalState, EngineStateModel) {
    (
        PhysicalState::from_config(&cfg.initial_state),
        EngineStateModel::from_config(&cfg.initial_state),
    )
}

// ─────────────────────────────────────────────────────────
// (1) VirtualClock 테스트
// ─────────────────────────────────────────────────────────

#[test]
fn test_clock_advances_monotonically() {
    let mut clock = VirtualClock::new();
    clock.advance(Duration::from_millis(100));
    clock.advance(Duration::from_millis(200));
    clock.advance(Duration::from_millis(50));
    assert_eq!(
        clock.now(),
        Duration::from_millis(350),
        "누적 시간이 단조 증가해야 함"
    );
}

#[test]
fn test_clock_drain_returns_events_in_time_order() {
    let mut clock = VirtualClock::new();
    clock.schedule(EventKind::Custom("a".into()), Duration::from_millis(500));
    clock.schedule(EventKind::Custom("b".into()), Duration::from_millis(1000));
    clock.schedule(EventKind::Custom("c".into()), Duration::from_millis(300));

    let events = clock.drain_until(Duration::from_millis(1000));
    assert_eq!(events.len(), 3, "3개 이벤트 모두 반환");
    assert_eq!(events[0].at, Duration::from_millis(300), "첫 번째 = 300ms");
    assert_eq!(events[1].at, Duration::from_millis(500), "두 번째 = 500ms");
    assert_eq!(
        events[2].at,
        Duration::from_millis(1000),
        "세 번째 = 1000ms"
    );
}

#[test]
fn test_clock_schedule_periodic() {
    let mut clock = VirtualClock::new();
    let period = Duration::from_millis(250);
    let until = Duration::from_millis(1000);

    clock.schedule_periodic(|| EventKind::Heartbeat, period, period, until);

    // 250, 500, 750, 1000 → 4개
    assert_eq!(clock.pending(), 4, "period=0.25s, until=1.0s → 4개 이벤트");
    let events = clock.drain_until(until);
    assert_eq!(events.len(), 4);
    assert_eq!(events[0].at, Duration::from_millis(250));
    assert_eq!(events[1].at, Duration::from_millis(500));
    assert_eq!(events[2].at, Duration::from_millis(750));
    assert_eq!(events[3].at, Duration::from_millis(1000));
}

#[test]
fn test_clock_fires_exactly_at_sub_tick_boundary() {
    // dt=50ms tick, heartbeat period=1s → 20 ticks 후 정확히 1개 Heartbeat
    let mut clock = VirtualClock::new();
    let tick = Duration::from_millis(50);
    let period = Duration::from_secs(1);

    clock.schedule_periodic(|| EventKind::Heartbeat, period, period, period);

    let mut heartbeat_count = 0;
    for _ in 0..20 {
        let next = clock.now() + tick;
        let events = clock.drain_until(next);
        for ev in &events {
            if ev.kind == EventKind::Heartbeat {
                heartbeat_count += 1;
            }
        }
        clock.advance(tick);
    }
    assert_eq!(heartbeat_count, 1, "20 ticks(1s) 후 Heartbeat 정확히 1회");
}

// ─────────────────────────────────────────────────────────
// (2) Noise 테스트
// ─────────────────────────────────────────────────────────

#[test]
fn test_noise_reproducibility() {
    let n = 30;
    let key = "test.stream";
    let sigma = 1.5;

    let mut rng1 = NoiseRng::new(42);
    let samples1: Vec<f64> = (0..n).map(|_| rng1.gaussian(key, sigma)).collect();

    let mut rng2 = NoiseRng::new(42);
    let samples2: Vec<f64> = (0..n).map(|_| rng2.gaussian(key, sigma)).collect();

    assert_eq!(samples1, samples2, "같은 seed + 같은 key → 동일 시퀀스");
}

#[test]
fn test_noise_disabled_returns_zero_sigma_equivalent() {
    // sigma=0이면 항상 0 반환 (noise 비활성화 동작)
    let mut rng = NoiseRng::new(123);
    for _ in 0..20 {
        assert_eq!(rng.gaussian("any.key", 0.0), 0.0, "sigma=0 → 0 반환");
    }
}

#[test]
fn test_noise_independent_streams() {
    let n = 20;
    let sigma = 2.0;

    // 서로 다른 key → 서로 다른 스트림
    let mut rng_a = NoiseRng::new(42);
    let samples_a: Vec<f64> = (0..n).map(|_| rng_a.gaussian("stream.a", sigma)).collect();

    let mut rng_b = NoiseRng::new(42);
    let samples_b: Vec<f64> = (0..n).map(|_| rng_b.gaussian("stream.b", sigma)).collect();

    assert_ne!(
        samples_a, samples_b,
        "서로 다른 seed_key는 독립 시퀀스를 가져야 함"
    );
}

// ─────────────────────────────────────────────────────────
// (3) Signal derive 테스트
// ─────────────────────────────────────────────────────────

#[test]
fn test_derive_memory_signal_maps_level_correctly() {
    let cfg = load_baseline();
    let (mut state, _) = make_state_and_engine(&cfg);

    // used=90% → available=10% → Emergency (boundary: <=10 = Emergency)
    state.device_memory_total_mb = 8192.0;
    state.device_memory_used_mb = 8192.0 * 0.90;

    let sig = derive_memory_signal_pub(&state, &cfg, &mut None);
    if let SystemSignal::MemoryPressure { level, .. } = sig {
        assert!(
            level >= Level::Critical,
            "used=90% → Critical 이상 기대, got {:?}",
            level
        );
    } else {
        panic!("Expected MemoryPressure");
    }
}

#[test]
fn test_derive_thermal_signal_reports_throttling() {
    let cfg = load_baseline();
    let (mut state, _) = make_state_and_engine(&cfg);

    // thermal_c=80 > soft=78 → throttling_active=true, Level=Critical
    state.thermal_c = 80.0;
    state.cpu_freq_mhz = 2400.0;

    let sig = derive_thermal_signal_pub(&state, &cfg, &mut None);
    if let SystemSignal::ThermalAlert {
        throttling_active,
        level,
        ..
    } = sig
    {
        assert!(
            throttling_active,
            "thermal=80 > soft=78 → throttling_active=true"
        );
        assert_eq!(level, Level::Critical, "thermal=80°C → Critical");
    } else {
        panic!("Expected ThermalAlert");
    }
}

#[test]
fn test_derive_compute_recommends_cpu_under_high_gpu_temp() {
    let cfg = load_baseline();
    let (mut state, _) = make_state_and_engine(&cfg);

    // gpu_cluster_thermal_c=85 (>= CRITICAL=75) → CPU 권장
    state.gpu_cluster_thermal_c = 85.0;
    state.engine_gpu_pct = 30.0;
    state.engine_cpu_pct = 30.0;
    state.external_cpu_pct = 0.0;
    state.external_gpu_pct = 0.0;

    let sig = derive_compute_signal_pub(&state, &cfg, &mut None);
    if let SystemSignal::ComputeGuidance {
        recommended_backend,
        ..
    } = sig
    {
        assert_eq!(
            recommended_backend,
            RecommendedBackend::Cpu,
            "GPU thermal=85 → Cpu 권장"
        );
    } else {
        panic!("Expected ComputeGuidance");
    }
}

#[test]
fn test_signal_polling_cadence_respects_config() {
    let cfg = load_baseline();
    // memory poll=0.25s → 1초 시나리오에서 4회
    let poll_s = cfg.observation.signals.memory.poll_interval_s;
    let run_duration = Duration::from_secs(1);
    let period = Duration::from_secs_f64(poll_s);

    let mut clock = VirtualClock::new();
    clock.schedule_periodic(|| EventKind::SignalMemory, period, period, run_duration);

    let events = clock.drain_until(run_duration);
    let count = events
        .iter()
        .filter(|e| e.kind == EventKind::SignalMemory)
        .count();

    let expected = (run_duration.as_secs_f64() / poll_s).round() as usize;
    assert_eq!(
        count, expected,
        "memory polling cadence: expected={}, actual={}",
        expected, count
    );
}

// ─────────────────────────────────────────────────────────
// (4) physics::step VirtualClock 시그니처 검증
// ─────────────────────────────────────────────────────────

#[test]
fn test_physics_step_signature_accepts_virtual_clock() {
    use crate::common::sim::expr::ExprContext;

    let cfg = load_baseline();
    let (mut state, mut engine) = make_state_and_engine(&cfg);
    engine.phase = "decode".to_string();
    state.gpu_freq_mhz = state.gpu_max_freq_mhz;

    let mut clock = VirtualClock::new();
    let dt = Duration::from_millis(50);
    let mut ctx = ExprContext::new();

    // VirtualClock으로 step 호출 — 컴파일 + 실행 모두 통과해야 함
    step(&mut state, &engine, &cfg, &clock, dt, &mut ctx).expect("step with VirtualClock");

    clock.advance(dt);
    step(&mut state, &engine, &cfg, &clock, dt, &mut ctx).expect("2nd step");

    // decode 모드에서 throughput > 0 확인 (Phase 2 회귀 테스트)
    assert!(
        state.throughput_tps > 0.0,
        "decode 모드에서 throughput > 0: {}",
        state.throughput_tps
    );
}
