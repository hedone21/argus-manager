//! Derived expression 평가.
//!
//! cfg.derived의 각 항목에 대해:
//! 1. state + engine 필드를 ExprContext에 바인딩
//! 2. expression 평가
//! 3. tau_s가 있으면 lag_step, 없으면 즉시 대입
//!
//! 평가 순서: cfg.derived 삽입 순서(insertion-order preserving).
//! 현재는 state + engine 필드만 참조 가능 (derived → derived 순환 금지).

#![allow(dead_code)]

use super::config::ScenarioConfig;
use super::expr::ExprContext;
use super::physics::{SimError, lag_step};
use super::state::{EngineStateModel, PhysicalState};

pub fn evaluate(
    state: &mut PhysicalState,
    engine: &EngineStateModel,
    cfg: &ScenarioConfig,
    ctx: &mut ExprContext,
) -> Result<(), SimError> {
    // state와 engine을 최신 값으로 context에 바인딩
    state.bind_to_context(ctx);
    engine.bind_to_context(ctx);

    for (name, derived) in &cfg.derived {
        // f64 평가 시도
        let result = derived.expr.tree.eval_float_with_context(&ctx.ctx);

        match result {
            Ok(new_val) => {
                let current = state.get_f64(name).unwrap_or(0.0);
                let applied = lag_step(current, new_val, derived.tau_s, 0.05); // dt는 여기서 고정 50ms
                state.set_f64(name, applied);
                // context도 갱신하여 이후 derived가 참조 가능하게 한다
                ctx.set_f64(name, applied);
            }
            Err(e) => {
                // 문자열 결과 시도 (kv_dtype 같은 케이스)
                match derived.expr.tree.eval_string_with_context(&ctx.ctx) {
                    Ok(_s) => {
                        // 문자열 derived는 state에 직접 저장 경로가 제한적
                        // 향후 확장 시 처리
                    }
                    Err(_) => {
                        return Err(SimError::Expr {
                            location: format!("derived.{}", name),
                            source: e,
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::config::load_scenario;
    use super::*;
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("sim")
    }

    #[test]
    fn test_derived_tbt_ms_from_throughput() {
        let path = fixtures_dir().join("baseline.yaml");
        let cfg = load_scenario(&path).expect("baseline should load");
        let mut state = PhysicalState::from_config(&cfg.initial_state);
        let engine = super::super::state::EngineStateModel::from_config(&cfg.initial_state);
        let mut ctx = ExprContext::new();

        // throughput_tps를 수동으로 설정
        state.throughput_tps = 10.0;
        ctx.set_f64("throughput_tps", 10.0);
        ctx.set_f64("tbt_target_ms", 100.0);

        // tbt_ms만 평가 (derived 전체 말고 직접)
        let tbt_expr = &cfg.derived["tbt_ms"];
        state.bind_to_context(&mut ctx);
        engine.bind_to_context(&mut ctx);
        let result = tbt_expr.expr.tree.eval_float_with_context(&ctx.ctx);
        assert!(result.is_ok(), "tbt_ms expr should evaluate: {:?}", result);
        let tbt = result.unwrap();
        // 1000 / 10 = 100
        assert!((tbt - 100.0).abs() < 1e-6);
    }
}
