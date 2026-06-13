//! Passive dynamics + DVFS + 1차 수렴 적분.
//!
//! tick 순서:
//! 1. DVFS 업데이트 (thermal → target_freq → freq 수렴)
//! 2. Passive heating (cluster thermal, coupling, KV grow)
//! 3. External injection 적용
//! 4. Action effects (compose.rs)
//! 5. Derived expression 평가 (derived.rs)

#![allow(dead_code)]

use std::time::Duration;

use super::clock::VirtualClock;
use super::compose;
use super::config::ScenarioConfig;
use super::derived;
use super::expr::ExprContext;
use super::state::{EngineStateModel, PhysicalState};

// ─────────────────────────────────────────────────────────
// SimError
// ─────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum SimError {
    #[error("Expression evaluation error at {location}: {source}")]
    Expr {
        location: String,
        #[source]
        source: evalexpr::EvalexprError,
    },
    #[error("Unknown field: {0}")]
    UnknownField(String),
}

// ─────────────────────────────────────────────────────────
// 1차 수렴 공식
// ─────────────────────────────────────────────────────────

/// x(t+dt) = x(t) + (target - x(t)) * (1 - exp(-dt/tau))
/// tau_s <= 0이면 즉시 target으로 설정한다.
pub fn lag_step(current: f64, target: f64, tau_s: f64, dt_s: f64) -> f64 {
    if tau_s <= 0.0 {
        return target;
    }
    current + (target - current) * (1.0 - (-dt_s / tau_s).exp())
}

// ─────────────────────────────────────────────────────────
// 메인 step 함수
// ─────────────────────────────────────────────────────────

/// 1 tick(dt 기간)을 진행한다.
///
/// `clock`의 현재 시각(`clock.now_secs()`)과 dt를 사용하므로
/// 호출 전에 clock.advance(dt)를 해서는 안 된다.
/// (advance는 harness에서 drain_until 이후에 별도로 수행한다.)
pub fn step(
    state: &mut PhysicalState,
    engine: &EngineStateModel,
    cfg: &ScenarioConfig,
    clock: &VirtualClock,
    dt: Duration,
    ctx: &mut ExprContext,
) -> Result<(), SimError> {
    let now_s = clock.now_secs();
    let dt_s = dt.as_secs_f64();

    // 1. DVFS 업데이트
    dvfs_update(state, cfg, dt_s);

    // 2. Passive dynamics (heating + coupling + KV grow)
    passive_dynamics(state, engine, cfg, dt_s);

    // 3. External injection 윈도우 적용
    external_injection_apply(state, cfg, now_s, dt_s);

    // 4. Action effects (compose.rs)
    compose::apply_actions(state, engine, cfg, dt_s, ctx)?;

    // 5. Derived expression 평가
    derived::evaluate(state, engine, cfg, ctx)?;

    Ok(())
}

/// 하위 호환성 래퍼: now_s, dt_s를 직접 받아 step을 호출한다.
/// Phase 2 테스트 코드에서 사용.
pub fn step_raw(
    state: &mut PhysicalState,
    engine: &EngineStateModel,
    cfg: &ScenarioConfig,
    now_s: f64,
    dt_s: f64,
    ctx: &mut ExprContext,
) -> Result<(), SimError> {
    let mut clock = VirtualClock::new();
    // now_s 만큼 시계를 미리 전진시킨다.
    clock.advance(Duration::from_secs_f64(now_s));
    step(
        state,
        engine,
        cfg,
        &clock,
        Duration::from_secs_f64(dt_s),
        ctx,
    )
}

// ─────────────────────────────────────────────────────────
// DVFS 업데이트
// ─────────────────────────────────────────────────────────

/// DVFS 피드백: thermal → target_freq 계산 → freq 1차 수렴.
fn dvfs_update(state: &mut PhysicalState, cfg: &ScenarioConfig, dt_s: f64) {
    // CPU
    let cpu_target_freq = dvfs_target_freq(
        state.cpu_cluster_thermal_c,
        cfg.dvfs.cpu.soft_threshold_c,
        cfg.dvfs.cpu.hard_threshold_c,
        cfg.dvfs.cpu.k_thermal,
        state.cpu_min_freq_mhz,
        state.cpu_max_freq_mhz,
    );
    state.cpu_freq_mhz = lag_step(
        state.cpu_freq_mhz,
        cpu_target_freq,
        cfg.dvfs.cpu.tau_s,
        dt_s,
    );

    // GPU
    let gpu_target_freq = dvfs_target_freq(
        state.gpu_cluster_thermal_c,
        cfg.dvfs.gpu.soft_threshold_c,
        cfg.dvfs.gpu.hard_threshold_c,
        cfg.dvfs.gpu.k_thermal,
        state.gpu_min_freq_mhz,
        state.gpu_max_freq_mhz,
    );
    state.gpu_freq_mhz = lag_step(
        state.gpu_freq_mhz,
        gpu_target_freq,
        cfg.dvfs.gpu.tau_s,
        dt_s,
    );
}

/// DVFS target freq 계산.
/// thermal < soft → max freq
/// thermal >= hard → min freq
/// 중간 → 선형 감소
fn dvfs_target_freq(
    thermal_c: f64,
    soft: f64,
    hard: f64,
    k_thermal: f64,
    min_freq: f64,
    max_freq: f64,
) -> f64 {
    if thermal_c < soft {
        max_freq
    } else if thermal_c >= hard {
        min_freq
    } else {
        let ratio = 1.0 - k_thermal * (thermal_c - soft) / (hard - soft);
        let ratio = ratio.clamp(min_freq / max_freq, 1.0);
        max_freq * ratio
    }
}

// ─────────────────────────────────────────────────────────
// Passive dynamics
// ─────────────────────────────────────────────────────────

fn passive_dynamics(
    state: &mut PhysicalState,
    engine: &EngineStateModel,
    cfg: &ScenarioConfig,
    dt_s: f64,
) {
    // ── CPU cluster thermal ──
    let cpu_dyn = &cfg.passive_dynamics.cpu_cluster_thermal_c;
    let cpu_load = state.engine_cpu_pct + state.external_cpu_pct;
    let cpu_coeff = cpu_dyn
        .heating
        .get("engine_cpu_pct_coeff")
        .copied()
        .unwrap_or(0.05);
    let cpu_freq_exp = cpu_dyn
        .heating
        .get("cpu_freq_mhz_exp")
        .copied()
        .unwrap_or(1.6);
    let cpu_freq_ratio = state.cpu_freq_mhz / state.cpu_max_freq_mhz;
    // target: baseline + 부하 발열 + freq 발열 항
    let cpu_target_thermal = cpu_dyn.baseline
        + cpu_coeff * cpu_load
        + (cpu_freq_ratio.powf(cpu_freq_exp) - 0.5).max(0.0) * 20.0;

    // ── GPU cluster thermal ──
    let gpu_dyn = &cfg.passive_dynamics.gpu_cluster_thermal_c;
    let gpu_load = state.engine_gpu_pct + state.external_gpu_pct;
    let gpu_coeff = gpu_dyn
        .heating
        .get("engine_gpu_pct_coeff")
        .copied()
        .unwrap_or(0.09);
    let gpu_freq_exp = gpu_dyn
        .heating
        .get("gpu_freq_mhz_exp")
        .copied()
        .unwrap_or(1.4);
    let gpu_freq_ratio = state.gpu_freq_mhz / state.gpu_max_freq_mhz;
    let gpu_target_thermal = gpu_dyn.baseline
        + gpu_coeff * gpu_load
        + (gpu_freq_ratio.powf(gpu_freq_exp) - 0.5).max(0.0) * 20.0;

    // ── 1차 수렴 적용 ──
    let new_cpu_thermal = lag_step(
        state.cpu_cluster_thermal_c,
        cpu_target_thermal,
        cpu_dyn.tau_s,
        dt_s,
    );
    let new_gpu_thermal = lag_step(
        state.gpu_cluster_thermal_c,
        gpu_target_thermal,
        gpu_dyn.tau_s,
        dt_s,
    );

    // ── 클러스터 간 열전도 (coupling) ──
    let cpu_to_gpu = cfg.passive_dynamics.thermal_coupling.cpu_to_gpu;
    let gpu_to_cpu = cfg.passive_dynamics.thermal_coupling.gpu_to_cpu;
    let gap = new_cpu_thermal - new_gpu_thermal;
    let cpu_thermal = new_cpu_thermal - cpu_to_gpu * gap * dt_s;
    let gpu_thermal = new_gpu_thermal + cpu_to_gpu * gap * dt_s;
    // GPU → CPU 방향
    let gap2 = new_gpu_thermal - new_cpu_thermal;
    let cpu_thermal = cpu_thermal + gpu_to_cpu * gap2 * dt_s;
    let gpu_thermal = gpu_thermal - gpu_to_cpu * gap2 * dt_s;

    state.cpu_cluster_thermal_c = cpu_thermal;
    state.gpu_cluster_thermal_c = gpu_thermal;

    // ── KV cache 성장 ──
    let grow = &cfg.passive_dynamics.kv_cache_bytes;
    let dtype = state.kv_dtype.as_str();
    // decode 또는 prefill 중이면 성장
    if (engine.phase == "decode" || engine.phase == "prefill")
        && let Some(&bytes_per_token) = grow.growth_per_token.get(dtype)
    {
        // dt_s 초 동안 처리되는 tokens 수 추정: throughput * dt_s
        let tokens_per_dt = if state.throughput_tps > 0.0 {
            state.throughput_tps * dt_s
        } else {
            // 첫 tick이면 기본 1 token/tick 가정
            1.0
        };
        let growth = bytes_per_token as f64 * tokens_per_dt;
        state.kv_cache_bytes = (state.kv_cache_bytes + growth).min(state.kv_cache_capacity_bytes);
    }
}

// ─────────────────────────────────────────────────────────
// External injection
// ─────────────────────────────────────────────────────────

fn external_injection_apply(
    state: &mut PhysicalState,
    cfg: &ScenarioConfig,
    now_s: f64,
    dt_s: f64,
) {
    for inj in &cfg.external_injections {
        let in_window = now_s >= inj.t_start && now_s < inj.t_start + inj.duration;
        if !in_window {
            continue;
        }

        // delta 결정: delta 필드 우선, delta_mb는 MB 단위
        let delta = if let Some(d) = inj.delta {
            d
        } else if let Some(dmb) = inj.delta_mb {
            dmb
        } else {
            continue;
        };

        match inj.op.as_str() {
            "add" => {
                if let Some(current) = state.get_f64(&inj.var) {
                    let target = current + delta;
                    let new_val = if inj.tau_s > 0.0 {
                        lag_step(current, target, inj.tau_s, dt_s)
                    } else {
                        target
                    };
                    state.set_f64(&inj.var, new_val);
                }
            }
            "set" => {
                let new_val = if inj.tau_s > 0.0 {
                    if let Some(current) = state.get_f64(&inj.var) {
                        lag_step(current, delta, inj.tau_s, dt_s)
                    } else {
                        delta
                    }
                } else {
                    delta
                };
                state.set_f64(&inj.var, new_val);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lag_step_zero_tau_is_instant() {
        let result = lag_step(0.0, 100.0, 0.0, 1.0);
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_lag_step_convergence() {
        let mut x = 0.0_f64;
        let target = 100.0;
        let tau_s = 1.0;
        let dt = 0.05; // 50ms
        for _ in 0..200 {
            // 10초 = 200 ticks
            x = lag_step(x, target, tau_s, dt);
        }
        // 10 tau 후에는 99.99% 수렴
        assert!((x - target).abs() < 0.01, "x={}", x);
    }

    #[test]
    fn test_dvfs_target_below_soft() {
        let t = dvfs_target_freq(60.0, 78.0, 95.0, 0.08, 400.0, 3200.0);
        assert_eq!(t, 3200.0);
    }

    #[test]
    fn test_dvfs_target_above_hard() {
        let t = dvfs_target_freq(100.0, 78.0, 95.0, 0.08, 400.0, 3200.0);
        assert_eq!(t, 400.0);
    }

    #[test]
    fn test_dvfs_target_in_range() {
        // thermal = 86.5 → 중간
        let t = dvfs_target_freq(86.5, 78.0, 95.0, 0.08, 400.0, 3200.0);
        assert!(t > 400.0 && t < 3200.0, "t={}", t);
    }
}
