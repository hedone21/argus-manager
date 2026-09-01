//! Phase 5 시나리오 기반 spec 테스트 — insta 스냅샷 중심.
//!
//! 각 테스트는:
//!   1. 시나리오 YAML 로드 + LuaPolicy 또는 MockPolicy 주입
//!   2. Simulator.run_for(30s)
//!   3. TrajectorySummary + relief 스냅샷을 insta로 고정
//!
//! 초기 실행: `INSTA_UPDATE=always cargo test -p argus_manager --test sim`
//! 스냅샷 검토: `cargo insta review` 또는 자동 accept

use std::path::PathBuf;
use std::time::Duration;

use crate::common::sim::{
    config::load_scenario, harness::Simulator, trajectory::TrajectorySummary,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sim")
}

fn scenarios_dir() -> PathBuf {
    fixtures_dir().join("scenarios")
}

fn lua_dir() -> PathBuf {
    fixtures_dir().join("lua")
}

// ─────────────────────────────────────────────────────────
// 시나리오 1: memory_pressure_steady
// ─────────────────────────────────────────────────────────

/// 시나리오: decode 중 메모리 사용량 선형 상승 → Warning/Critical → Evict directive.
#[cfg(feature = "lua")]
#[test]
fn scenario_memory_pressure_steady() {
    use argus_manager::config::AdaptationConfig;

    let scenario_path = scenarios_dir().join("memory_pressure_steady.yaml");
    let lua_path = lua_dir().join("memory_evict_graduated.lua");

    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, AdaptationConfig::default())
        .expect("Simulator::with_lua_policy 생성 실패");
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("memory_pressure_summary", summary);
    });

    // 기본 검증: 30초 실행, signal 존재
    assert!(
        sim.trajectory().signal_count_by_kind("memory_pressure") >= 1,
        "memory_pressure signal이 기록되어야 함"
    );
}

/// lua feature 없을 때 MockPolicy로 memory_pressure_steady 기본 동작 확인.
#[cfg(not(feature = "lua"))]
#[test]
fn scenario_memory_pressure_steady() {
    use argus_manager::signal::{Level, SystemSignal};
    use argus_shared::{EngineCommand, EngineDirective};

    use crate::common::sim::mock_policy::MockPolicy;

    let scenario_path = scenarios_dir().join("memory_pressure_steady.yaml");
    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut mock = MockPolicy::new();
    mock.directive_on_signal = Some(Box::new(|sig| {
        if let SystemSignal::MemoryPressure { level, .. } = sig {
            if *level >= Level::Warning {
                return Some(EngineDirective {
                    seq_id: 100,
                    commands: vec![EngineCommand::KvCompress { budget: 0.5 }],
                });
            }
        }
        None
    }));

    let mut sim = Simulator::new(cfg, Box::new(mock));
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("memory_pressure_summary_mock", summary);
    });

    assert!(
        sim.trajectory().signal_count_by_kind("memory_pressure") >= 1,
        "memory_pressure signal이 기록되어야 함"
    );
}

// ─────────────────────────────────────────────────────────
// 시나리오 2: thermal_ramp_with_decode
// ─────────────────────────────────────────────────────────

/// 시나리오: decode + GPU 과열 → ThermalAlert → SwitchHw/Throttle directive.
#[cfg(feature = "lua")]
#[test]
fn scenario_thermal_ramp_with_decode() {
    use argus_manager::config::AdaptationConfig;

    let scenario_path = scenarios_dir().join("thermal_ramp_with_decode.yaml");
    let lua_path = lua_dir().join("thermal_switch_backend.lua");

    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, AdaptationConfig::default())
        .expect("Simulator::with_lua_policy 생성 실패");
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("thermal_ramp_summary", summary);
    });

    // thermal signal이 기록되어야 함
    assert!(
        sim.trajectory().signal_count_by_kind("thermal_alert") >= 1,
        "thermal_alert signal이 기록되어야 함"
    );
}

/// lua feature 없을 때 MockPolicy로 thermal_ramp 기본 동작 확인.
#[cfg(not(feature = "lua"))]
#[test]
fn scenario_thermal_ramp_with_decode() {
    use argus_manager::signal::{Level, SystemSignal};
    use argus_shared::{EngineCommand, EngineDirective};

    use crate::common::sim::mock_policy::MockPolicy;

    let scenario_path = scenarios_dir().join("thermal_ramp_with_decode.yaml");
    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut mock = MockPolicy::new();
    mock.directive_on_signal = Some(Box::new(|sig| {
        if let SystemSignal::ThermalAlert { level, .. } = sig {
            if *level >= Level::Warning {
                return Some(EngineDirective {
                    seq_id: 101,
                    commands: vec![EngineCommand::Suspend],
                });
            }
        }
        None
    }));

    let mut sim = Simulator::new(cfg, Box::new(mock));
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("thermal_ramp_summary_mock", summary);
    });

    assert!(
        sim.trajectory().signal_count_by_kind("thermal_alert") >= 1,
        "thermal_alert signal이 기록되어야 함"
    );
}

// ─────────────────────────────────────────────────────────
// 시나리오 3: partition_contention
// ─────────────────────────────────────────────────────────

/// 시나리오: partition_ratio=0.5 decode + BW 경합 → ComputeGuidance → SetPartitionRatio.
#[cfg(feature = "lua")]
#[test]
fn scenario_partition_contention() {
    use argus_manager::config::AdaptationConfig;

    let scenario_path = scenarios_dir().join("partition_contention.yaml");
    let lua_path = lua_dir().join("partition_adaptive.lua");

    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, AdaptationConfig::default())
        .expect("Simulator::with_lua_policy 생성 실패");
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("partition_contention_summary", summary);
    });

    // compute signal이 기록되어야 함
    assert!(
        sim.trajectory().signal_count_by_kind("compute_guidance") >= 1,
        "compute_guidance signal이 기록되어야 함"
    );
}

/// lua feature 없을 때 MockPolicy로 partition_contention 기본 동작 확인.
#[cfg(not(feature = "lua"))]
#[test]
fn scenario_partition_contention() {
    use argus_manager::signal::{Level, SystemSignal};
    use argus_shared::{EngineCommand, EngineDirective};

    use crate::common::sim::mock_policy::MockPolicy;

    let scenario_path = scenarios_dir().join("partition_contention.yaml");
    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut mock = MockPolicy::new();
    mock.directive_on_signal = Some(Box::new(|sig| {
        if let SystemSignal::ComputeGuidance { level, .. } = sig {
            if *level >= Level::Warning {
                return Some(EngineDirective {
                    seq_id: 102,
                    commands: vec![EngineCommand::KvCompress { budget: 0.5 }],
                });
            }
        }
        None
    }));

    let mut sim = Simulator::new(cfg, Box::new(mock));
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("partition_contention_summary_mock", summary);
    });

    assert!(
        sim.trajectory().signal_count_by_kind("compute_guidance") >= 1,
        "compute_guidance signal이 기록되어야 함"
    );
}

// ─────────────────────────────────────────────────────────
// 시나리오 4: memory + thermal 복합 신호
// ─────────────────────────────────────────────────────────

/// 시나리오: memory + thermal 두 신호 동시 발생 → composition 처리 검증.
/// baseline 시나리오에 높은 초기값으로 복합 압력 유도.
#[cfg(feature = "lua")]
#[test]
fn scenario_memory_and_thermal_combined() {
    use argus_manager::config::AdaptationConfig;

    let baseline_path = fixtures_dir().join("baseline.yaml");
    let lua_path = lua_dir().join("memory_and_thermal_combined.lua");

    let mut cfg =
        load_scenario(&baseline_path).unwrap_or_else(|e| panic!("baseline 로드 실패: {e}"));

    // 복합 압력 초기값 설정
    cfg.initial_state.device_memory_used_mb = 7000;
    cfg.initial_state.gpu_cluster_thermal_c = 69.0;
    cfg.initial_state.phase = "decode".to_string();
    cfg.rng_seed = Some(42);

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, AdaptationConfig::default())
        .expect("Simulator::with_lua_policy 생성 실패");
    sim.run_for(Duration::from_secs(20)).expect("20s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("combined_signals_summary", summary);
    });

    // 두 신호 종류가 모두 기록되어야 함
    assert!(
        sim.trajectory().signal_count_by_kind("memory_pressure") >= 1,
        "memory_pressure signal이 기록되어야 함"
    );
    assert!(
        sim.trajectory().signal_count_by_kind("thermal_alert") >= 1,
        "thermal_alert signal이 기록되어야 함"
    );
}

/// lua feature 없을 때 MockPolicy로 복합 신호 기본 동작 확인.
#[cfg(not(feature = "lua"))]
#[test]
fn scenario_memory_and_thermal_combined() {
    use argus_manager::signal::{Level, SystemSignal};
    use argus_shared::{EngineCommand, EngineDirective};

    use crate::common::sim::mock_policy::MockPolicy;

    let baseline_path = fixtures_dir().join("baseline.yaml");
    let mut cfg =
        load_scenario(&baseline_path).unwrap_or_else(|e| panic!("baseline 로드 실패: {e}"));

    cfg.initial_state.device_memory_used_mb = 7000;
    cfg.initial_state.gpu_cluster_thermal_c = 69.0;
    cfg.initial_state.phase = "decode".to_string();
    cfg.rng_seed = Some(42);

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut mock = MockPolicy::new();
    mock.directive_on_signal = Some(Box::new(|sig| match sig {
        SystemSignal::MemoryPressure { level, .. } if *level >= Level::Warning => {
            Some(EngineDirective {
                seq_id: 103,
                commands: vec![EngineCommand::KvCompress { budget: 0.5 }],
            })
        }
        SystemSignal::ThermalAlert { level, .. } if *level >= Level::Warning => {
            Some(EngineDirective {
                seq_id: 104,
                commands: vec![EngineCommand::Suspend],
            })
        }
        _ => None,
    }));

    let mut sim = Simulator::new(cfg, Box::new(mock));
    sim.run_for(Duration::from_secs(20)).expect("20s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("combined_signals_summary_mock", summary);
    });

    // 두 신호 모두 기록
    assert!(
        sim.trajectory().signal_count_by_kind("memory_pressure") >= 1,
        "memory_pressure signal이 기록되어야 함"
    );
    assert!(
        sim.trajectory().signal_count_by_kind("thermal_alert") >= 1,
        "thermal_alert signal이 기록되어야 함"
    );
}

// ─────────────────────────────────────────────────────────
// s25_galaxy.yaml 디바이스 preset smoke 테스트 (Phase 6)
// ─────────────────────────────────────────────────────────

/// smoke 2: s25_galaxy.yaml + memory_evict_graduated Lua로 5초 시뮬 실행.
/// 정상 종료 + heartbeat 기록 확인. extends + physical simulation 통합 검증.
#[cfg(feature = "lua")]
#[test]
fn test_s25_galaxy_preset_runs_with_lua_policy() {
    use argus_manager::config::AdaptationConfig;

    let preset_path = fixtures_dir().join("s25_galaxy.yaml");
    let lua_path = lua_dir().join("memory_evict_graduated.lua");

    let cfg =
        load_scenario(&preset_path).unwrap_or_else(|e| panic!("s25_galaxy.yaml 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, AdaptationConfig::default())
        .expect("Simulator::with_lua_policy 생성 실패");

    sim.run_for(Duration::from_secs(5)).expect("5초 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    // 5초 실행 후 heartbeat >= 1 (interval_s=1.0)
    assert!(
        summary.heartbeat_count >= 1,
        "5s 실행 후 heartbeat >= 1, actual={}",
        summary.heartbeat_count
    );
    // signal이 기록되어야 함
    assert!(
        !summary.signal_count_by_kind.is_empty(),
        "signal이 1종 이상 기록되어야 함"
    );
}

/// smoke 2 (lua feature 없을 때): MockPolicy로 s25_galaxy.yaml 기본 동작 확인.
#[cfg(not(feature = "lua"))]
#[test]
fn test_s25_galaxy_preset_runs_with_lua_policy() {
    use crate::common::sim::mock_policy::MockPolicy;

    let preset_path = fixtures_dir().join("s25_galaxy.yaml");
    let cfg =
        load_scenario(&preset_path).unwrap_or_else(|e| panic!("s25_galaxy.yaml 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut sim = Simulator::new(cfg, Box::new(MockPolicy::new()));
    sim.run_for(Duration::from_secs(5)).expect("5초 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    assert!(
        summary.heartbeat_count >= 1,
        "5s 실행 후 heartbeat >= 1, actual={}",
        summary.heartbeat_count
    );
}
