//! Phase 5 시나리오 기반 spec 테스트 — insta 스냅샷 중심.
//!
//! 각 테스트는:
//!   1. 시나리오 YAML 로드 + LuaPolicy 또는 MockPolicy 주입
//!   2. Simulator.run_for(30s)
//!   3. TrajectorySummary + relief 스냅샷을 insta로 고정
//!
//! 초기 실행: `INSTA_UPDATE=always cargo test -p argus_manager --test sim`
//! 스냅샷 검토: `cargo insta review` 또는 자동 accept

use std::collections::BTreeMap;
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

/// ReliefTable 스냅샷을 안정적으로 직렬화하기 위해
/// BTreeMap<String, Vec<f32(3자리 반올림)>> 형태로 변환한다.
fn format_relief(
    relief: &std::collections::HashMap<String, [f32; 6]>,
) -> BTreeMap<String, Vec<f32>> {
    relief
        .iter()
        .map(|(k, vals)| {
            let rounded: Vec<f32> = vals
                .iter()
                .map(|&v| (v * 1000.0).round() / 1000.0)
                .collect();
            (k.clone(), rounded)
        })
        .collect()
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

    let relief = argus_manager::pipeline::get_relief_snapshot(sim.policy.as_mut());
    if !relief.is_empty() {
        let formatted = format_relief(&relief);
        insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
            insta::assert_yaml_snapshot!("memory_pressure_relief", formatted);
        });
    }

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
    use argus_shared::{EngineCommand, EngineDirective, Level, SystemSignal};

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
                    commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.8 }],
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

    let relief = argus_manager::pipeline::get_relief_snapshot(sim.policy.as_mut());
    if !relief.is_empty() {
        let formatted = format_relief(&relief);
        insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
            insta::assert_yaml_snapshot!("thermal_ramp_relief", formatted);
        });
    }

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
    use argus_shared::{EngineCommand, EngineDirective, Level, SystemSignal};

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
                    commands: vec![EngineCommand::Throttle { delay_ms: 100 }],
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

    let relief = argus_manager::pipeline::get_relief_snapshot(sim.policy.as_mut());
    if !relief.is_empty() {
        let formatted = format_relief(&relief);
        insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
            insta::assert_yaml_snapshot!("partition_contention_relief", formatted);
        });
    }

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
    use argus_shared::{EngineCommand, EngineDirective, Level, SystemSignal};

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
                    commands: vec![EngineCommand::SetPartitionRatio { ratio: 0.0 }],
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

    let relief = argus_manager::pipeline::get_relief_snapshot(sim.policy.as_mut());
    if !relief.is_empty() {
        let formatted = format_relief(&relief);
        insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
            insta::assert_yaml_snapshot!("combined_signals_relief", formatted);
        });
    }

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
    use argus_shared::{EngineCommand, EngineDirective, Level, SystemSignal};

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
                commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.7 }],
            })
        }
        SystemSignal::ThermalAlert { level, .. } if *level >= Level::Warning => {
            Some(EngineDirective {
                seq_id: 104,
                commands: vec![EngineCommand::Throttle { delay_ms: 100 }],
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

/// smoke 1: s25_galaxy.yaml이 올바르게 파싱되고 initial_state 값이 기대와 일치하는지 검증.
/// extends: baseline.yaml 상속 + S25 override가 올바르게 적용되어야 한다.
#[test]
fn test_s25_galaxy_preset_loads_and_initial_state_correct() {
    let preset_path = fixtures_dir().join("s25_galaxy.yaml");
    let cfg =
        load_scenario(&preset_path).unwrap_or_else(|e| panic!("s25_galaxy.yaml 로드 실패: {e}"));

    // override된 필드 검증
    assert_eq!(
        cfg.initial_state.device_memory_total_mb, 12288,
        "S25 RAM = 12 GB → 12288 MB"
    );
    assert_eq!(
        cfg.initial_state.gpu_max_freq_mhz, 1100,
        "Adreno 750 max freq = 1100 MHz"
    );
    assert_eq!(
        cfg.initial_state.cpu_max_freq_mhz, 4200,
        "Snapdragon 8 Elite big cluster max = 4200 MHz"
    );
    // throttle threshold override 확인
    assert_eq!(
        cfg.initial_state.throttle_threshold_c as u32, 82,
        "S25 throttle threshold = 82 °C"
    );
    // baseline에서 상속된 필드 (override 없음) — kv_dtype 확인
    assert_eq!(
        cfg.initial_state.kv_dtype, "f16",
        "kv_dtype baseline 상속 확인"
    );
}

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

// ─────────────────────────────────────────────────────────
// relief_snapshot 비공허 검증 (PR 2+3 핵심 회귀 테스트)
// ─────────────────────────────────────────────────────────

/// VirtualClockHandle이 LuaPolicy에 주입되어 30s 시뮬에서 relief가 실제로 학습되는지 검증.
/// 이전에는 wall-clock 기반 observation이 ~100ms 안에 끝나는 harness에서 3s delay를
/// 충족하지 못해 relief_snapshot이 항상 공허했다.
#[cfg(feature = "lua")]
#[test]
fn scenario_partition_contention_produces_non_empty_relief() {
    use argus_manager::config::AdaptationConfig;

    let scenario_path = scenarios_dir().join("partition_contention.yaml");
    let lua_path = lua_dir().join("partition_adaptive.lua");

    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, AdaptationConfig::default())
        .expect("Simulator::with_lua_policy 생성 실패");
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let relief = argus_manager::pipeline::get_relief_snapshot(sim.policy.as_mut());
    assert!(
        !relief.is_empty(),
        "30s 시뮬 후 relief_snapshot이 비어있지 않아야 함 (VirtualClockHandle이 3s 관측 지연을 충족해야 함)"
    );

    // Phase A: trajectory에 ReliefUpdate 이벤트가 기록되어야 함.
    let update_count = sim
        .trajectory()
        .entries
        .iter()
        .filter(|e| {
            matches!(
                e,
                crate::common::sim::trajectory::TrajectoryEntry::ReliefUpdate { .. }
            )
        })
        .count();
    assert!(
        update_count >= 1,
        "trajectory에 ReliefUpdate 이벤트가 >=1건 기록되어야 함 (학습 경로 trace 가능 확인)"
    );
}

// ─────────────────────────────────────────────────────────
// S25 + 메모리 램프 + production 범용 Lua 정책 (policy_example.lua)
// ─────────────────────────────────────────────────────────

/// Galaxy S25 preset에서 메모리 사용량 점진 증가 시나리오를 production용
/// 범용 Lua 정책(`manager/scripts/policy_example.lua`)으로 실행한다.
///
/// 검증 포인트:
/// - 12 GB total + 8192 → 11500 MB 램프 → Warning + Critical 시그널 발생
/// - policy_example.lua는 pressure가 가장 높은 도메인을 찾아 best-relief action 선택
/// - 30s 실행 후 적어도 하나의 evict directive가 방출되고 relief 학습이 발생
///
/// 주의: policy_example.lua는 `c.relief[*]` 값 > 0 인 action만 선택하므로
///       relief 테이블이 비면 cold-start 시 directive를 0건 방출한다.
///       production과 동일하게 `manager/policy_config.toml`의 default_relief를
///       시드로 사용해 부트스트랩을 가능하게 한다.
///
/// 디버깅: `SIM_TIMELINE=compact cargo test ... -- --nocapture`
#[cfg(feature = "lua")]
#[test]
fn scenario_s25_memory_pressure_with_general_policy() {
    let scenario_path = scenarios_dir().join("s25_memory_pressure_steady.yaml");
    let lua_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("policy_example.lua");
    let policy_toml_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("policy_config.toml");

    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    // production policy_config.toml에서 default_relief 시드 로드
    let manager_cfg = argus_manager::config::Config::from_file(&policy_toml_path)
        .expect("policy_config.toml 로드 실패");
    let adaptation_cfg = manager_cfg.adaptation;
    assert!(
        !adaptation_cfg.default_relief.is_empty(),
        "default_relief가 비어있으면 cold-start 시 정책이 무효이므로 테스트 사전조건 위반"
    );

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, adaptation_cfg)
        .expect("Simulator::with_lua_policy 생성 실패");
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let relief = argus_manager::pipeline::get_relief_snapshot(sim.policy.as_mut());
    let overrun = argus_manager::pipeline::get_observation_overrun_count(sim.policy.as_mut());

    let traj = sim.trajectory();
    traj.print_timeline_if_enabled();

    let summary = TrajectorySummary::from_trajectory(traj, cpu_max, gpu_max);
    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("s25_memory_pressure_general_summary", summary);
    });

    if !relief.is_empty() {
        let formatted = format_relief(&relief);
        insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
            insta::assert_yaml_snapshot!("s25_memory_pressure_general_relief", formatted);
        });
    }

    // 1) memory_pressure signal 발생
    let mem_sigs = traj.signal_count_by_kind("memory_pressure");
    assert!(
        mem_sigs >= 1,
        "memory_pressure signal이 1개 이상 기록되어야 함, actual={mem_sigs}"
    );

    // 2) Warning 이상 진입 시점에 directive 방출
    assert!(
        traj.directive_count() >= 1,
        "범용 정책은 pressure 진입 시 최소 1개 directive를 방출해야 함, actual={}",
        traj.directive_count()
    );

    // 3) policy_example.lua는 evict 계열을 선택해야 함 (memory가 dominant pressure)
    traj.assert_contains_directive_kind("Evict")
        .or_else(|_| traj.assert_contains_directive_kind("Quant"))
        .expect("memory dominant pressure 시 Evict/Quant 계열 directive가 등장해야 함");

    // 4) Multi-slot observation queue는 관측 소실 없이 학습을 누적해야 한다.
    //    30s 시뮬 동안 5~6 Hz directive rate × 3 s 지연 = 동시 in-flight ≤ 20,
    //    MAX_PENDING_OBSERVATIONS=32 용량 안에서 전부 수용되어 overrun=0이어야 함.
    assert_eq!(
        overrun, 0,
        "multi-slot 큐는 현재 rate를 수용해야 함 — overrun 발생은 용량 조정 신호: {overrun}"
    );

    // 5) relief 테이블 업데이트가 실제로 발생해야 함 (학습 경로 동작 확인).
    //    suppress된 directive의 observation은 취소되므로, relief update 수는
    //    실제로 엔진에 전달된 directive 수와 일치한다 (>= 1 이면 충분).
    let update_count = traj
        .entries
        .iter()
        .filter(|e| {
            matches!(
                e,
                crate::common::sim::trajectory::TrajectoryEntry::ReliefUpdate { .. }
            )
        })
        .count();
    assert!(
        update_count >= 1,
        "directive 방출 후 성숙한 observation이 relief 테이블에 누적되어야 함: \
         ReliefUpdate 이벤트 {update_count}건"
    );

    // 6) summary에 overrun 경고가 **없어야** 함 (multi-slot 수용 증거).
    let summary = traj.format_session_summary();
    assert!(
        !summary.contains("observation overruns"),
        "multi-slot 큐는 overrun이 없어야 하므로 summary에 경고가 표시되면 안 됨:\n{summary}"
    );
}
