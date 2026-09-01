//! `sim_run` — Policy simulator runner.
//!
//! 시나리오 YAML + Lua policy 스크립트를 받아 시뮬레이션을 실행하고
//! Session Summary를 출력한다.
//!
//! 사용 예:
//! ```
//! sim_run --scenario tests/fixtures/sim/baseline.yaml \
//!         --lua manager/scripts/policy_example.lua \
//!         --duration 30
//! ```

use clap::Parser;

#[derive(Parser)]
#[command(name = "sim_run", about = "Policy simulator runner")]
struct Args {
    /// 시나리오 YAML 파일 경로
    #[arg(long)]
    scenario: std::path::PathBuf,

    /// Lua policy 스크립트 경로
    #[arg(long)]
    lua: std::path::PathBuf,

    /// 시뮬레이션 실행 시간 (초, 기본 30)
    #[arg(long, default_value = "30")]
    duration: f64,

    /// 전체 타임라인 출력 (compact 형식)
    #[arg(long)]
    verbose: bool,

    /// trajectory를 JSON으로 저장할 경로
    #[arg(long)]
    output_json: Option<std::path::PathBuf>,

    /// policy_config.toml 경로 (AdaptationConfig 로드용)
    /// 미지정 시 AdaptationConfig::default() 사용
    #[arg(long)]
    config: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    env_logger::init();

    let cfg = argus_manager::sim::config::load_scenario(&args.scenario)
        .map_err(|e| anyhow::anyhow!("시나리오 로드 실패: {e}"))?;

    let adaptation = if let Some(cfg_path) = &args.config {
        argus_manager::config::Config::from_file(cfg_path)
            .map_err(|e| anyhow::anyhow!("설정 로드 실패: {e}"))?
            .adaptation
    } else {
        argus_manager::config::AdaptationConfig::default()
    };

    let mut sim =
        argus_manager::sim::harness::Simulator::with_lua_policy(cfg, &args.lua, adaptation)
            .map_err(|e| anyhow::anyhow!("Simulator 초기화 실패: {e}"))?;

    sim.run_for(std::time::Duration::from_secs_f64(args.duration))?;

    let traj = sim.trajectory();

    if args.verbose {
        println!("{}", traj.format_timeline_compact());
        println!();
    }

    println!("{}", traj.format_session_summary());

    if let Some(out) = args.output_json {
        traj.dump_json(&out)?;
        eprintln!("trajectory 저장: {}", out.display());
    }

    Ok(())
}
