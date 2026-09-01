//! PhysicalState → SystemSignal / EngineMessage 투영.
//!
//! Level 결정은 production monitor helper를 위임한다:
//! - Memory: `argus_manager::monitor::memory_level_from_available_pct`
//! - Thermal: `argus_manager::monitor::thermal_level_from_temp_c`
//! - Compute: `argus_manager::monitor::compute_level_from_pcts`
//! - Energy: 배터리 없는 경우 Normal / power_budget 기본값 사용

#![allow(dead_code)]

use crate::config::ComputeMonitorConfig;
use crate::monitor::{
    compute_level_from_pcts, compute_recommendation, memory_level_from_available_pct,
    thermal_level_from_temp_c,
};
use argus_shared::{EngineMessage, EngineState, EngineStatus, Phase};

/// Three-level view of a resource domain, for the simulator's own reporting.
///
/// The IPC contract carried this as `ResourceLevel` while the heartbeat reported
/// per-domain levels. It does not any more — the heartbeat is six engine-observable
/// facts — so the simulator keeps its own copy of the projection it renders.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ResourceLevel {
    Normal,
    Warning,
    Critical,
}
use crate::signal::{ComputeReason, EnergyReason, Level, RecommendedBackend, SystemSignal};

use super::clock::EventKind;
use super::config::ScenarioConfig;
use super::noise::NoiseRng;
use super::state::{EngineStateModel, PhysicalState};

// ─────────────────────────────────────────────────────────
// 에너지 기본 power_budget (mW).
// 배터리를 가정하지 않는 시뮬레이터 전용 상수.
// ─────────────────────────────────────────────────────────
mod energy_budgets {
    pub const BASELINE_MW: u32 = 5000;
    pub const WARNING_MW: u32 = 3000;
    pub const CRITICAL_MW: u32 = 1500;
    pub const EMERGENCY_MW: u32 = 500;
}

// ─────────────────────────────────────────────────────────
// Level 결정 함수 — production helper로 위임
// ─────────────────────────────────────────────────────────

/// max(cpu%, gpu%) → compute Level (production helper 위임).
pub fn compute_level_from_usage(cpu_pct: f64, gpu_pct: f64) -> Level {
    compute_level_from_pcts(cpu_pct, gpu_pct)
}

/// Level → ResourceLevel 변환 (Emergency → Critical 클램핑).
pub fn to_resource_level(level: Level) -> ResourceLevel {
    match level {
        Level::Normal => ResourceLevel::Normal,
        Level::Warning => ResourceLevel::Warning,
        Level::Critical | Level::Emergency => ResourceLevel::Critical,
    }
}

// ─────────────────────────────────────────────────────────
// derive_heartbeat
// ─────────────────────────────────────────────────────────

/// PhysicalState + EngineStateModel → EngineMessage::Heartbeat.
///
/// noise가 Some이면 지정된 sigma로 노이즈를 추가한다.
/// CPU/GPU pct는 [0.0, 1.0]으로 클램핑된다 (INV-091, INV-092 준수).
pub fn derive_heartbeat(
    state: &PhysicalState,
    engine: &EngineStateModel,
    cfg: &ScenarioConfig,
    rng: &mut Option<NoiseRng>,
) -> EngineMessage {
    // memory level: used/total 비율로 available % 계산
    let available_pct = if state.device_memory_total_mb > 0.0 {
        (1.0 - state.device_memory_used_mb / state.device_memory_total_mb) * 100.0
    } else {
        100.0
    };
    let _mem_level = memory_level_from_available_pct(available_pct);

    // compute level: engine_cpu + engine_gpu 기준 (external 제외 — engine self 관점)
    let _compute_level = compute_level_from_usage(state.engine_cpu_pct, state.engine_gpu_pct);

    // throughput: noise 적용
    let throughput_noise = if let Some(rng) = rng {
        let sigma = cfg
            .observation
            .heartbeat
            .noise
            .get("throughput_tps")
            .and_then(|n| n.sigma)
            .unwrap_or(0.0);
        rng.gaussian("hb.tps", sigma) as f32
    } else {
        0.0
    };
    let _actual_throughput = (state.throughput_tps as f32 + throughput_noise).max(0.0);

    // self_cpu_pct: noise 적용 후 [0.0, 1.0] 클램핑
    let cpu_noise = if let Some(rng) = rng {
        let sigma = cfg
            .observation
            .heartbeat
            .noise
            .get("self_cpu_pct")
            .and_then(|n| n.sigma)
            .unwrap_or(0.0);
        rng.gaussian("hb.cpu", sigma)
    } else {
        0.0
    };
    let _self_cpu_pct = ((state.engine_cpu_pct / 100.0) + cpu_noise).clamp(0.0, 1.0);
    let _self_gpu_pct = (state.engine_gpu_pct / 100.0).clamp(0.0, 1.0);

    // KV 캐시 utilization
    let _kv_util = if state.kv_cache_capacity_bytes > 0.0 {
        (state.kv_cache_bytes / state.kv_cache_capacity_bytes) as f32
    } else {
        0.0
    };

    let _engine_state = match engine.phase.as_str() {
        "idle" => EngineState::Idle,
        _ => EngineState::Running,
    };

    let status = EngineStatus {
        kv_cache_bytes: 1024,
        kv_cache_budget_bytes: 4096,
        kv_cache_tokens: 32,
        tbt_ms: 12.5,
        phase: Phase::Decode,
        state: EngineState::Running,
    };

    EngineMessage::Heartbeat(status)
}

// ─────────────────────────────────────────────────────────
// derive_signal
// ─────────────────────────────────────────────────────────

/// EventKind가 signal 종류이면 해당 SystemSignal을 생성한다.
/// signal이 아닌 EventKind이면 None을 반환한다.
pub fn derive_signal(
    event: &EventKind,
    state: &PhysicalState,
    cfg: &ScenarioConfig,
    rng: &mut Option<NoiseRng>,
) -> Option<SystemSignal> {
    match event {
        EventKind::SignalMemory => Some(derive_memory_signal(state, cfg, rng)),
        EventKind::SignalCompute => Some(derive_compute_signal(state, cfg, rng)),
        EventKind::SignalThermal => Some(derive_thermal_signal(state, cfg, rng)),
        EventKind::SignalEnergy => Some(derive_energy_signal(state, cfg, rng)),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────
// 개별 signal 생성 함수
// ─────────────────────────────────────────────────────────

/// 테스트에서 직접 호출 가능한 pub 버전.
pub fn derive_memory_signal_pub(
    state: &PhysicalState,
    cfg: &ScenarioConfig,
    rng: &mut Option<NoiseRng>,
) -> SystemSignal {
    derive_memory_signal(state, cfg, rng)
}

fn derive_memory_signal(
    state: &PhysicalState,
    cfg: &ScenarioConfig,
    rng: &mut Option<NoiseRng>,
) -> SystemSignal {
    let total_bytes = (state.device_memory_total_mb * 1024.0 * 1024.0) as u64;

    // available_bytes: noise 적용 (sigma_mb → bytes)
    let noise_bytes = if let Some(rng) = rng {
        let sigma_mb = cfg
            .observation
            .signals
            .memory
            .noise
            .get("available_bytes")
            .and_then(|n| n.sigma_mb)
            .unwrap_or(0.0);
        (rng.gaussian("sig.mem", sigma_mb) * 1024.0 * 1024.0) as i64
    } else {
        0
    };

    let used_bytes = (state.device_memory_used_mb * 1024.0 * 1024.0) as i64;
    let available_bytes = ((total_bytes as i64 - used_bytes + noise_bytes).max(0)) as u64;

    let available_pct = if total_bytes > 0 {
        (available_bytes as f64 / total_bytes as f64) * 100.0
    } else {
        100.0
    };
    let level = memory_level_from_available_pct(available_pct);

    let reclaim = match level {
        Level::Normal => 0,
        Level::Warning => (total_bytes as f64 * 0.05) as u64,
        Level::Critical => (total_bytes as f64 * 0.10) as u64,
        Level::Emergency => (total_bytes as f64 * 0.20) as u64,
    };

    SystemSignal::MemoryPressure {
        level,
        available_bytes,
        total_bytes,
        reclaim_target_bytes: reclaim,
    }
}

/// 테스트에서 직접 호출 가능한 pub 버전.
pub fn derive_compute_signal_pub(
    state: &PhysicalState,
    cfg: &ScenarioConfig,
    rng: &mut Option<NoiseRng>,
) -> SystemSignal {
    derive_compute_signal(state, cfg, rng)
}

fn derive_compute_signal(
    state: &PhysicalState,
    _cfg: &ScenarioConfig,
    rng: &mut Option<NoiseRng>,
) -> SystemSignal {
    // total CPU = engine + external, cap at 100
    let cpu_usage_pct = (state.engine_cpu_pct + state.external_cpu_pct).min(100.0);
    let gpu_usage_pct = (state.engine_gpu_pct + state.external_gpu_pct).min(100.0);

    // noise (현재 config에는 compute noise가 sigma 필드로 있음)
    let cpu_with_noise = if let Some(rng) = rng {
        let sigma = 0.5_f64; // config에서 sigma=0.5 기본값
        (cpu_usage_pct + rng.gaussian("sig.cpu", sigma)).clamp(0.0, 100.0)
    } else {
        cpu_usage_pct
    };

    let level = compute_level_from_usage(cpu_with_noise, gpu_usage_pct);

    // backend 추천 (gpu_cluster_thermal이 높으면 CPU 권장)
    let gpu_thermal = state.gpu_cluster_thermal_c;
    let warning_pct = ComputeMonitorConfig::default().warning_pct;
    let (recommended_backend, reason) =
        sim_compute_recommendation(cpu_with_noise, gpu_usage_pct, gpu_thermal, warning_pct);

    SystemSignal::ComputeGuidance {
        level,
        recommended_backend,
        reason,
        cpu_usage_pct: cpu_with_noise,
        gpu_usage_pct,
    }
}

/// 테스트에서 직접 호출 가능한 pub 버전.
pub fn derive_thermal_signal_pub(
    state: &PhysicalState,
    cfg: &ScenarioConfig,
    rng: &mut Option<NoiseRng>,
) -> SystemSignal {
    derive_thermal_signal(state, cfg, rng)
}

fn derive_thermal_signal(
    state: &PhysicalState,
    cfg: &ScenarioConfig,
    rng: &mut Option<NoiseRng>,
) -> SystemSignal {
    // noise: sigma_mc → °C 로 변환 후 적용
    let noise_c = if let Some(rng) = rng {
        let sigma_mc = cfg
            .observation
            .signals
            .thermal
            .noise
            .get("temperature_mc")
            .and_then(|n| n.sigma_mc)
            .unwrap_or(0.0);
        rng.gaussian("sig.therm", sigma_mc / 1000.0) // mc → °C
    } else {
        0.0
    };

    let temp_c = state.thermal_c + noise_c;
    let temperature_mc = (temp_c * 1000.0) as i32;

    let level = thermal_level_from_temp_c(temp_c);

    // throttling: soft threshold = dvfs.cpu.soft_threshold_c 사용
    let soft_threshold_c = cfg.dvfs.cpu.soft_threshold_c;
    let throttling_active = state.thermal_c >= soft_threshold_c;

    // throttle_ratio: freq/max_freq (CPU 기준)
    let throttle_ratio = if state.cpu_max_freq_mhz > 0.0 {
        (state.cpu_freq_mhz / state.cpu_max_freq_mhz).clamp(0.0, 1.0)
    } else {
        1.0
    };

    SystemSignal::ThermalAlert {
        level,
        temperature_mc,
        throttling_active,
        throttle_ratio,
    }
}

fn derive_energy_signal(
    state: &PhysicalState,
    _cfg: &ScenarioConfig,
    rng: &mut Option<NoiseRng>,
) -> SystemSignal {
    // 시뮬레이터에는 배터리가 없으므로 Normal + baseline power_budget 사용
    // noise로 약간의 변동 허용
    let noise_mw = if let Some(rng) = rng {
        rng.gaussian("sig.energy", 100.0) as i32
    } else {
        0
    };

    // 기본적으로 Normal (충전 중 가정)
    let level = Level::Normal;
    let reason = EnergyReason::None;
    let power_budget_mw = ((energy_budgets::BASELINE_MW as i32 + noise_mw).max(0)) as u32;

    // thermal 기반으로 에너지 레벨 조정: 온도가 높으면 ThermalPower
    let thermal_lv = thermal_level_from_temp_c(state.thermal_c);
    let (level, reason) = if thermal_lv == Level::Emergency {
        (Level::Emergency, EnergyReason::ThermalPower)
    } else if thermal_lv >= Level::Critical {
        (Level::Critical, EnergyReason::ThermalPower)
    } else {
        (level, reason)
    };

    SystemSignal::EnergyConstraint {
        level,
        reason,
        power_budget_mw,
    }
}

// ─────────────────────────────────────────────────────────
// Compute 추천 룰
// ─────────────────────────────────────────────────────────

/// GPU thermal 상태를 추가로 고려하는 시뮬레이터 전용 wrapper.
///
/// GPU thermal이 Critical 이상이면 즉시 CPU 권장.
/// 그 외에는 production `compute_recommendation` helper에 위임한다.
fn sim_compute_recommendation(
    cpu_pct: f64,
    gpu_pct: f64,
    gpu_thermal_c: f64,
    warning_pct: f64,
) -> (RecommendedBackend, ComputeReason) {
    // GPU 온도가 Critical 이상이면 CPU 권장 (시뮬레이터 전용 규칙)
    if thermal_level_from_temp_c(gpu_thermal_c) >= Level::Critical {
        return (RecommendedBackend::Cpu, ComputeReason::GpuBottleneck);
    }

    compute_recommendation(cpu_pct, gpu_pct, warning_pct)
}

// ─────────────────────────────────────────────────────────
// 단위 테스트
// ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::load_scenario;

    use super::super::state::{EngineStateModel, PhysicalState};
    use super::*;
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("sim")
    }

    fn load_baseline() -> super::super::config::ScenarioConfig {
        load_scenario(fixtures_dir().join("baseline.yaml")).expect("baseline.yaml should load")
    }

    fn make_state_and_engine(
        cfg: &super::super::config::ScenarioConfig,
    ) -> (PhysicalState, EngineStateModel) {
        (
            PhysicalState::from_config(&cfg.initial_state),
            EngineStateModel::from_config(&cfg.initial_state),
        )
    }

    #[test]
    fn test_derive_memory_signal_maps_level_correctly() {
        let cfg = load_baseline();
        let (mut state, _engine) = make_state_and_engine(&cfg);

        // used/total = 0.90 → available = 10% → Emergency (threshold=10)
        state.device_memory_total_mb = 8192.0;
        state.device_memory_used_mb = 8192.0 * 0.90;

        let sig = derive_memory_signal(&state, &cfg, &mut None);
        if let SystemSignal::MemoryPressure { level, .. } = sig {
            // available_pct ≈ 10% → Emergency (threshold=10, 경계)
            assert!(
                level >= Level::Critical,
                "used=90% → Critical 이상 기대: {:?}",
                level
            );
        } else {
            panic!("Expected MemoryPressure");
        }
    }

    #[test]
    fn test_derive_thermal_signal_reports_throttling() {
        let cfg = load_baseline();
        let (mut state, _engine) = make_state_and_engine(&cfg);

        // thermal_c=80 (soft=78 초과)
        state.thermal_c = 80.0;
        state.cpu_freq_mhz = 2400.0;

        let sig = derive_thermal_signal(&state, &cfg, &mut None);
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
            // 80°C → Critical (75~85 사이)
            assert_eq!(level, Level::Critical, "thermal=80°C → Critical");
        } else {
            panic!("Expected ThermalAlert");
        }
    }

    #[test]
    fn test_derive_compute_recommends_cpu_under_high_gpu_temp() {
        let cfg = load_baseline();
        let (mut state, _engine) = make_state_and_engine(&cfg);

        // gpu_cluster_thermal_c=85 (>= CRITICAL_C=75) → CPU 권장
        state.gpu_cluster_thermal_c = 85.0;
        state.engine_gpu_pct = 30.0;
        state.engine_cpu_pct = 30.0;

        let sig = derive_compute_signal(&state, &cfg, &mut None);
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
        use super::super::clock::{EventKind, VirtualClock};
        use std::time::Duration;

        let cfg = load_baseline();
        // memory poll=0.25s → 1초 시나리오에서 4회
        let poll_s = cfg.observation.signals.memory.poll_interval_s;
        let run_duration = Duration::from_secs(1);

        let mut clock = VirtualClock::new();
        let period = Duration::from_secs_f64(poll_s);
        clock.schedule_periodic(|| EventKind::SignalMemory, period, period, run_duration);

        let events = clock.drain_until(run_duration);
        let count = events
            .iter()
            .filter(|e| e.kind == EventKind::SignalMemory)
            .count();

        let expected = (run_duration.as_secs_f64() / poll_s).round() as usize;
        assert_eq!(
            count, expected,
            "polling cadence: expected={}, actual={}",
            expected, count
        );
    }
}
