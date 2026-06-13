//! evalexpr 래퍼: 로드 시 AST 컴파일, 런타임 HashMapContext 바인딩.
//!
//! Phase 1에서는 ExprContext와 CompiledExpr을 제공한다.
//! Phase 2에서 register_builtins()에 실제 내장 함수를 구현한다.

// Phase 2+에서 사용될 API stub이 포함되므로 dead_code 경고를 억제한다.
#![allow(dead_code)]

use evalexpr::{
    ContextWithMutableFunctions, ContextWithMutableVariables, EvalexprError, Function,
    HashMapContext, Node, Value,
};
use serde::{Deserialize, Deserializer, de};

/// 컴파일된 expression 노드 + 원본 소스 문자열.
/// `serde::Deserialize` 시 즉시 AST를 컴파일한다.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    pub source: String,
    pub tree: Node,
}

impl<'de> Deserialize<'de> for CompiledExpr {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        let tree = evalexpr::build_operator_tree(&s).map_err(|e| {
            de::Error::custom(format!("expression compile error in {:?}: {}", s, e))
        })?;
        Ok(CompiledExpr { source: s, tree })
    }
}

// ─────────────────────────────────────────────────────────
// ExprContext
// ─────────────────────────────────────────────────────────

/// evalexpr HashMapContext 래퍼. 변수 바인딩 및 평가 API를 제공한다.
pub struct ExprContext {
    pub ctx: HashMapContext,
}

impl ExprContext {
    pub fn new() -> Self {
        let mut ctx = HashMapContext::new();
        register_builtins(&mut ctx);
        ExprContext { ctx }
    }

    pub fn set_f64(&mut self, key: &str, v: f64) {
        self.ctx
            .set_value(key.to_string(), Value::Float(v))
            .expect("set_f64 should not fail");
    }

    pub fn set_str(&mut self, key: &str, v: &str) {
        self.ctx
            .set_value(key.to_string(), Value::String(v.to_string()))
            .expect("set_str should not fail");
    }

    pub fn eval_f64(&self, expr: &Node) -> Result<f64, EvalexprError> {
        expr.eval_float_with_context(&self.ctx)
    }

    pub fn eval_bool(&self, expr: &Node) -> Result<bool, EvalexprError> {
        expr.eval_boolean_with_context(&self.ctx)
    }
}

impl Default for ExprContext {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────
// Builtin 함수 등록
// ─────────────────────────────────────────────────────────

/// evalexpr에서 Tuple 인자를 꺼내는 헬퍼.
fn get_tuple(arg: &Value) -> Result<&[Value], EvalexprError> {
    match arg {
        Value::Tuple(v) => Ok(v.as_slice()),
        other => Err(EvalexprError::expected_tuple(other.clone())),
    }
}

fn get_float(v: &Value) -> Result<f64, EvalexprError> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        other => Err(EvalexprError::expected_float(other.clone())),
    }
}

fn get_string(v: &Value) -> Result<&str, EvalexprError> {
    match v {
        Value::String(s) => Ok(s.as_str()),
        other => Err(EvalexprError::expected_string(other.clone())),
    }
}

/// 내장 함수를 HashMapContext에 등록한다.
///
/// 등록 함수:
/// - `phase_throughput(phase, device, ratio, cpu_freq_ratio, gpu_freq_ratio, bw_util,
///                     base_tps_decode_gpu, base_tps_decode_cpu, base_tps_decode_partition,
///                     base_tps_prefill_gpu)` → f64
/// - `throttle_factor(delay_ms)` → f64
/// - `skip_boost(skip_ratio)` → f64
/// - `base_by_phase(phase)` → f64
/// - `dtype_from_bits(bits)` → String
/// - `merge_overhead(ratio)` → f64
pub fn register_builtins(ctx: &mut HashMapContext) {
    // ── phase_throughput ──────────────────────────────────────────
    // 인자: (phase, device, partition_ratio,
    //         cpu_freq_ratio, gpu_freq_ratio, bw_util,
    //         base_tps_decode_gpu, base_tps_decode_cpu,
    //         base_tps_decode_partition, base_tps_prefill_gpu)
    ctx.set_function(
        "phase_throughput".to_string(),
        Function::new(|arg| {
            let args = get_tuple(arg)?;
            if args.len() != 10 {
                return Err(EvalexprError::WrongOperatorArgumentAmount {
                    expected: 10,
                    actual: args.len(),
                });
            }
            let phase = get_string(&args[0])?;
            let device = get_string(&args[1])?;
            let partition_ratio = get_float(&args[2])?;
            let cpu_freq_ratio = get_float(&args[3])?;
            let gpu_freq_ratio = get_float(&args[4])?;
            let bw_util = get_float(&args[5])?;
            let base_tps_decode_gpu = get_float(&args[6])?;
            let base_tps_decode_cpu = get_float(&args[7])?;
            let base_tps_decode_partition = get_float(&args[8])?;
            let base_tps_prefill_gpu = get_float(&args[9])?;

            let tps = compute_phase_throughput(
                phase,
                device,
                partition_ratio,
                cpu_freq_ratio,
                gpu_freq_ratio,
                bw_util,
                base_tps_decode_gpu,
                base_tps_decode_cpu,
                base_tps_decode_partition,
                base_tps_prefill_gpu,
            );
            Ok(Value::Float(tps))
        }),
    )
    .expect("register phase_throughput");

    // ── throttle_factor ───────────────────────────────────────────
    // 인자: (delay_ms)
    // 공식: 1.0 / (1.0 + delay_ms / 10.0)
    ctx.set_function(
        "throttle_factor".to_string(),
        Function::new(|arg| {
            let delay_ms = match arg {
                Value::Tuple(v) if v.len() == 1 => get_float(&v[0])?,
                other => get_float(other)?,
            };
            let factor = 1.0 / (1.0 + delay_ms / 10.0);
            Ok(Value::Float(factor))
        }),
    )
    .expect("register throttle_factor");

    // ── skip_boost ────────────────────────────────────────────────
    // 인자: (skip_ratio)
    // 공식: 1.0 + skip_ratio * 0.8
    ctx.set_function(
        "skip_boost".to_string(),
        Function::new(|arg| {
            let ratio = match arg {
                Value::Tuple(v) if v.len() == 1 => get_float(&v[0])?,
                other => get_float(other)?,
            };
            let boost = 1.0 + ratio * 0.8;
            Ok(Value::Float(boost))
        }),
    )
    .expect("register skip_boost");

    // ── base_by_phase ─────────────────────────────────────────────
    // 인자: (phase)
    // idle→0, prefill→70, decode→10
    ctx.set_function(
        "base_by_phase".to_string(),
        Function::new(|arg| {
            let phase = match arg {
                Value::Tuple(v) if v.len() == 1 => get_string(&v[0])?.to_string(),
                other => get_string(other)?.to_string(),
            };
            let base = match phase.as_str() {
                "idle" => 0.0,
                "prefill" => 70.0,
                "decode" => 10.0,
                _ => 0.0,
            };
            Ok(Value::Float(base))
        }),
    )
    .expect("register base_by_phase");

    // ── dtype_from_bits ───────────────────────────────────────────
    // 인자: (bits) — 2|4|8|16|32
    ctx.set_function(
        "dtype_from_bits".to_string(),
        Function::new(|arg| {
            let bits = match arg {
                Value::Tuple(v) if v.len() == 1 => get_float(&v[0])?,
                other => get_float(other)?,
            };
            let dtype = match bits as u32 {
                2 => "q2",
                4 => "q4",
                8 => "q8",
                16 => "f16",
                32 => "f32",
                _ => "f16",
            };
            Ok(Value::String(dtype.to_string()))
        }),
    )
    .expect("register dtype_from_bits");

    // ── merge_overhead ────────────────────────────────────────────
    // 인자: (ratio)
    // 공식: 4 * ratio * (1-ratio) * 0.15  (ratio=0.5 근처에서 최대 페널티 ~0.15)
    ctx.set_function(
        "merge_overhead".to_string(),
        Function::new(|arg| {
            let ratio = match arg {
                Value::Tuple(v) if v.len() == 1 => get_float(&v[0])?,
                other => get_float(other)?,
            };
            let overhead = 4.0 * ratio * (1.0 - ratio) * 0.15;
            Ok(Value::Float(overhead))
        }),
    )
    .expect("register merge_overhead");
}

/// phase_throughput 내부 계산 로직.
/// 이 함수는 `physics.rs`에서도 직접 호출할 수 있다.
#[allow(clippy::too_many_arguments)]
pub fn compute_phase_throughput(
    phase: &str,
    device: &str,
    partition_ratio: f64,
    cpu_freq_ratio: f64,
    gpu_freq_ratio: f64,
    bw_util: f64,
    base_tps_decode_gpu: f64,
    base_tps_decode_cpu: f64,
    base_tps_decode_partition: f64,
    base_tps_prefill_gpu: f64,
) -> f64 {
    match phase {
        "idle" => 0.0,
        "prefill" => {
            // prefill은 항상 GPU 사용
            base_tps_prefill_gpu * gpu_freq_ratio
        }
        "decode" => match device {
            "cpu" => base_tps_decode_cpu * cpu_freq_ratio,
            "opencl" | "cuda" => {
                // BW contention penalty: bw_util이 100%에 가까울수록 페널티 증가
                let bw_penalty = (bw_util / 200.0).min(0.4); // 최대 40% 패널티
                base_tps_decode_gpu * gpu_freq_ratio * (1.0 - bw_penalty)
            }
            "partition" => {
                if partition_ratio <= 0.0 || partition_ratio >= 1.0 {
                    // partition 비활성화 → GPU-only
                    base_tps_decode_gpu * gpu_freq_ratio
                } else {
                    let bw_penalty = (bw_util / 200.0).min(0.4);
                    let base = base_tps_decode_partition
                        * (1.0 - compute_merge_overhead(partition_ratio))
                        * (1.0 - bw_penalty);
                    // GPU쪽과 CPU쪽 freq를 각각 반영
                    let gpu_part = partition_ratio * gpu_freq_ratio;
                    let cpu_part = (1.0 - partition_ratio) * cpu_freq_ratio;
                    base * (gpu_part + cpu_part)
                }
            }
            _ => base_tps_decode_gpu * gpu_freq_ratio,
        },
        _ => 0.0,
    }
}

fn compute_merge_overhead(ratio: f64) -> f64 {
    4.0 * ratio * (1.0 - ratio) * 0.15
}

// ─────────────────────────────────────────────────────────
// Dry-run 바인딩용 더미 context
// ─────────────────────────────────────────────────────────

/// config 로드 후 expression 전수 검증에 사용하는 더미 context.
/// initial_state의 모든 필드 + 알려진 action parameter를 0으로 바인딩한다.
pub fn make_dry_run_context() -> ExprContext {
    let mut ctx = ExprContext::new();

    // initial_state fields (숫자형)
    let numeric_fields = [
        "kv_cache_bytes",
        "kv_cache_capacity_bytes",
        "kv_cache_tokens",
        "kv_cache_token_capacity",
        "device_memory_total_mb",
        "device_memory_used_mb",
        "memory_bw_utilization_pct",
        "engine_cpu_pct",
        "external_cpu_pct",
        "cpu_freq_mhz",
        "cpu_max_freq_mhz",
        "cpu_min_freq_mhz",
        "engine_gpu_pct",
        "external_gpu_pct",
        "gpu_freq_mhz",
        "gpu_max_freq_mhz",
        "gpu_min_freq_mhz",
        "thermal_c",
        "cpu_cluster_thermal_c",
        "gpu_cluster_thermal_c",
        "throttle_threshold_c",
        "base_tps_decode_gpu",
        "base_tps_decode_cpu",
        "base_tps_decode_partition",
        "base_tps_prefill_gpu",
        "partition_ratio",
        "throttle_delay_ms",
        "tbt_target_ms",
        "skip_ratio",
        // action parameters
        "target_ratio",
        "delay_ms",
        "ratio",
        "bits",
    ];
    for f in &numeric_fields {
        ctx.set_f64(f, 0.0);
    }

    // derived 변수 (expression 참조용)
    let derived_fields = [
        "throughput_tps",
        "tbt_ms",
        "latency_degradation",
        "skip_ratio",
    ];
    for f in &derived_fields {
        ctx.set_f64(f, 0.0);
    }

    // 문자열 필드
    ctx.set_str("phase", "idle");
    ctx.set_str("active_device", "opencl");
    ctx.set_str("kv_dtype", "f16");
    ctx.set_str("device", "opencl");

    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiled_expr_basic() {
        let src = "1.0 + 2.0";
        let tree = evalexpr::build_operator_tree(src).unwrap();
        let expr = CompiledExpr {
            source: src.to_string(),
            tree,
        };
        let ctx = ExprContext::new();
        let v = ctx.eval_f64(&expr.tree).unwrap();
        assert!((v - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_dry_run_context_has_target_ratio() {
        let ctx = make_dry_run_context();
        // evalexpr v11: 중괄호 없이 식별자 참조
        let src = "target_ratio * 2.0";
        let tree = evalexpr::build_operator_tree(src).unwrap();
        let v = ctx.eval_f64(&tree).unwrap();
        assert!((v - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_throttle_factor_builtin() {
        let mut ctx = ExprContext::new();
        ctx.set_f64("throttle_delay_ms", 0.0);
        let tree = evalexpr::build_operator_tree("throttle_factor(throttle_delay_ms)").unwrap();
        let v = ctx.eval_f64(&tree).unwrap();
        // delay=0 → factor=1.0
        assert!((v - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_throttle_factor_reduces_with_delay() {
        let mut ctx = ExprContext::new();
        ctx.set_f64("d", 100.0);
        let tree = evalexpr::build_operator_tree("throttle_factor(d)").unwrap();
        let v = ctx.eval_f64(&tree).unwrap();
        // delay=100 → 1/(1+10) = 1/11 ≈ 0.0909
        assert!((v - 1.0 / 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_skip_boost_builtin() {
        let mut ctx = ExprContext::new();
        ctx.set_f64("s", 0.5);
        let tree = evalexpr::build_operator_tree("skip_boost(s)").unwrap();
        let v = ctx.eval_f64(&tree).unwrap();
        // 0.5 → 1.0 + 0.5 * 0.8 = 1.4
        assert!((v - 1.4).abs() < 1e-9);
    }

    #[test]
    fn test_merge_overhead_builtin() {
        let mut ctx = ExprContext::new();
        ctx.set_f64("r", 0.5);
        let tree = evalexpr::build_operator_tree("merge_overhead(r)").unwrap();
        let v = ctx.eval_f64(&tree).unwrap();
        // 4 * 0.5 * 0.5 * 0.15 = 0.15
        assert!((v - 0.15).abs() < 1e-9);
    }

    #[test]
    fn test_phase_throughput_idle() {
        let tps =
            compute_phase_throughput("idle", "opencl", 0.0, 1.0, 1.0, 0.0, 18.5, 4.2, 22.0, 145.0);
        assert_eq!(tps, 0.0);
    }

    #[test]
    fn test_phase_throughput_decode_gpu() {
        let tps = compute_phase_throughput(
            "decode", "opencl", 0.0, 1.0, 1.0, 0.0, 18.5, 4.2, 22.0, 145.0,
        );
        // bw_util=0 → no penalty
        assert!((tps - 18.5).abs() < 1e-6);
    }

    #[test]
    fn test_dtype_from_bits_builtin() {
        let mut ctx = ExprContext::new();
        ctx.set_f64("b", 4.0);
        let tree = evalexpr::build_operator_tree("dtype_from_bits(b)").unwrap();
        let result = tree.eval_string_with_context(&ctx.ctx).unwrap();
        assert_eq!(result, "q4");
    }
}
