//! Action composition: dimension-wise 결합 규칙 + interaction_term.
//!
//! engine.active_actions의 각 액션에 대해:
//! 1. cfg.actions[name] lookup
//! 2. when 조건 평가
//! 3. effects 집계 (같은 dimension은 per_dimension 룰로 결합)
//! 4. interactions 매칭 후 추가 effect 적용
//! 5. lag_step으로 state 업데이트

#![allow(dead_code)]

use std::collections::HashMap;

use super::config::{CompositionOp, Effect, ExprOrValue, ScenarioConfig};
use super::expr::ExprContext;
use super::physics::{SimError, lag_step};
use super::state::{EngineStateModel, PhysicalState};

// ─────────────────────────────────────────────────────────
// 집계된 effect 표현
// ─────────────────────────────────────────────────────────

/// 단일 dimension에 대해 수집된 적용 결과.
#[derive(Debug)]
enum CollectedEffect {
    /// Scale factor 목록 (multiply 결합)
    Scales(Vec<f64>),
    /// Add delta 목록 (add/add_delta 결합)
    Deltas(Vec<f64>),
    /// Set target 후보 목록 (max/min 결합)
    Targets(Vec<f64>),
    /// tau_s 변경 (set_tau)
    SetTau(f64),
}

// ─────────────────────────────────────────────────────────
// apply_actions
// ─────────────────────────────────────────────────────────

pub fn apply_actions(
    state: &mut PhysicalState,
    engine: &EngineStateModel,
    cfg: &ScenarioConfig,
    dt_s: f64,
    ctx: &mut ExprContext,
) -> Result<(), SimError> {
    if engine.active_actions.is_empty() {
        return Ok(());
    }

    // ── 현재 state + engine을 context에 바인딩 ──
    state.bind_to_context(ctx);
    engine.bind_to_context(ctx);

    // ── 각 dimension에 대해 효과 수집 ──
    // dimension → (composition_op, Vec<효과값>)
    let mut dim_effects: HashMap<String, (CompositionOp, Vec<f64>)> = HashMap::new();
    // dimension → tau_s (set_tau 오버라이드)
    let mut dim_tau: HashMap<String, f64> = HashMap::new();
    // 문자열 set (active_device 등)
    let mut string_sets: HashMap<String, String> = HashMap::new();

    for action_name in &engine.active_actions {
        let spec = match cfg.actions.get(action_name) {
            Some(s) => s,
            None => continue, // 알려지지 않은 action은 skip
        };

        // when 조건 평가
        if let Some(when_expr) = &spec.when {
            let cond = when_expr
                .tree
                .eval_boolean_with_context(&ctx.ctx)
                .unwrap_or(false);
            if !cond {
                continue;
            }
        }

        // effects 순회
        for (dim, effect) in &spec.effects {
            let composition_op = cfg
                .composition
                .per_dimension
                .get(dim)
                .copied()
                .unwrap_or(cfg.composition.default);

            match effect {
                Effect::Set { value, tau_s } => {
                    match value {
                        ExprOrValue::StringLiteral(s) => {
                            string_sets.insert(dim.clone(), s.clone());
                        }
                        ExprOrValue::Expr(e) => {
                            // 문자열 평가 먼저 시도
                            if let Ok(s) = e.tree.eval_string_with_context(&ctx.ctx) {
                                string_sets.insert(dim.clone(), s);
                            } else if let Ok(v) = e.tree.eval_float_with_context(&ctx.ctx) {
                                collect_effect(
                                    &mut dim_effects,
                                    dim,
                                    CompositionOp::Max, // Set은 단일값 → max로 처리
                                    v,
                                );
                                if *tau_s > 0.0 {
                                    dim_tau.insert(dim.clone(), *tau_s);
                                }
                            }
                        }
                        ExprOrValue::Literal(v) => {
                            collect_effect(
                                &mut dim_effects,
                                dim,
                                CompositionOp::Max, // Set은 단일값
                                *v,
                            );
                            if *tau_s > 0.0 {
                                dim_tau.insert(dim.clone(), *tau_s);
                            }
                        }
                    }
                }
                Effect::Scale { factor, tau_s } => {
                    let v = eval_expr_or_value(factor, ctx).unwrap_or(1.0);
                    // Scale은 항상 Multiply로 집계한다.
                    // per_dimension 룰은 Set/Add에 적용되며 Scale factor는 곱산으로만 처리한다.
                    collect_effect(&mut dim_effects, dim, CompositionOp::Multiply, v);
                    if *tau_s > 0.0 {
                        dim_tau.entry(dim.clone()).or_insert(*tau_s);
                    }
                }
                Effect::Add { value, tau_s } => {
                    let v = eval_expr_or_value(value, ctx).unwrap_or(0.0);
                    let add_op = match composition_op {
                        CompositionOp::Add | CompositionOp::AddDelta => composition_op,
                        _ => CompositionOp::Add,
                    };
                    collect_effect(&mut dim_effects, dim, add_op, v);
                    if *tau_s > 0.0 {
                        dim_tau.entry(dim.clone()).or_insert(*tau_s);
                    }
                }
                Effect::SetTau { value } => {
                    dim_tau.insert(dim.clone(), *value);
                }
            }
        }
    }

    // ── interactions 매칭 ──
    for term in &cfg.interactions {
        let all_active = term
            .actions
            .iter()
            .all(|a| engine.active_actions.contains(a));
        if !all_active {
            continue;
        }
        for (dim, effect) in &term.effects {
            let composition_op = cfg
                .composition
                .per_dimension
                .get(dim)
                .copied()
                .unwrap_or(cfg.composition.default);

            match effect {
                Effect::Scale { factor, tau_s } => {
                    let v = eval_expr_or_value(factor, ctx).unwrap_or(1.0);
                    // interaction Scale도 항상 Multiply
                    collect_effect(&mut dim_effects, dim, CompositionOp::Multiply, v);
                    if *tau_s > 0.0 {
                        dim_tau.entry(dim.clone()).or_insert(*tau_s);
                    }
                }
                Effect::Add { value, tau_s } => {
                    let v = eval_expr_or_value(value, ctx).unwrap_or(0.0);
                    let add_op = match composition_op {
                        CompositionOp::Add | CompositionOp::AddDelta => composition_op,
                        _ => CompositionOp::Add,
                    };
                    collect_effect(&mut dim_effects, dim, add_op, v);
                    if *tau_s > 0.0 {
                        dim_tau.entry(dim.clone()).or_insert(*tau_s);
                    }
                }
                Effect::Set { value, tau_s } => {
                    if let ExprOrValue::Literal(v) = value {
                        let set_op = match composition_op {
                            CompositionOp::Max | CompositionOp::Min => composition_op,
                            _ => CompositionOp::Max,
                        };
                        collect_effect(&mut dim_effects, dim, set_op, *v);
                        if *tau_s > 0.0 {
                            dim_tau.insert(dim.clone(), *tau_s);
                        }
                    }
                }
                Effect::SetTau { value } => {
                    dim_tau.insert(dim.clone(), *value);
                }
            }
        }
    }

    // ── state에 최종 효과 적용 ──
    for (dim, (op, values)) in &dim_effects {
        if values.is_empty() {
            continue;
        }

        let tau_s = dim_tau.get(dim).copied().unwrap_or(0.0);

        let current = match state.get_f64(dim) {
            Some(v) => v,
            None => continue, // 알 수 없는 dimension은 skip
        };

        let target = match op {
            CompositionOp::Multiply => {
                // Scale 효과: factor들을 곱해서 current에 적용
                let factor: f64 = values.iter().product();
                current * factor
            }
            CompositionOp::Add | CompositionOp::AddDelta => {
                // delta 합산
                let delta: f64 = values.iter().sum();
                current + delta
            }
            CompositionOp::Max => {
                // Set 효과의 max: target 후보 중 최대값
                values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            }
            CompositionOp::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
        };

        let new_val = lag_step(current, target, tau_s, dt_s);
        state.set_f64(dim, new_val);
    }

    // 문자열 필드 즉시 적용
    for (dim, val) in &string_sets {
        if dim == "active_device" || dim == "kv_dtype" {
            // string 필드는 PhysicalState에 직접 설정
            if dim == "kv_dtype" {
                state.kv_dtype = val.clone();
            }
            // active_device는 EngineStateModel에 속하지만 시뮬레이터에서 PhysicalState로 미러링
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────
// 헬퍼
// ─────────────────────────────────────────────────────────

fn collect_effect(
    dim_effects: &mut HashMap<String, (CompositionOp, Vec<f64>)>,
    dim: &str,
    op: CompositionOp,
    value: f64,
) {
    let entry = dim_effects
        .entry(dim.to_string())
        .or_insert_with(|| (op, Vec::new()));
    entry.1.push(value);
}

fn eval_expr_or_value(eov: &ExprOrValue, ctx: &ExprContext) -> Option<f64> {
    match eov {
        ExprOrValue::Literal(v) => Some(*v),
        ExprOrValue::Expr(e) => e.tree.eval_float_with_context(&ctx.ctx).ok(),
        ExprOrValue::StringLiteral(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // apply_actions 테스트는 test_physics.rs에서 통합 테스트로 수행
    // 여기서는 헬퍼 함수만 테스트

    #[test]
    fn test_collect_effect_multiply() {
        let mut map: HashMap<String, (CompositionOp, Vec<f64>)> = HashMap::new();
        collect_effect(&mut map, "cpu_pct", CompositionOp::Multiply, 0.8);
        collect_effect(&mut map, "cpu_pct", CompositionOp::Multiply, 0.9);
        let (_, vals) = &map["cpu_pct"];
        let product: f64 = vals.iter().product();
        assert!((product - 0.72).abs() < 1e-9);
    }
}
