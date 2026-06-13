//! YAML 시나리오 스키마: serde + validator 기반.
//!
//! 모든 struct에 `#[serde(deny_unknown_fields)]`를 적용한다.
//! `extends` 키로 parent YAML을 상속(deep merge)할 수 있다.
//! validator는 deep merge 완료 후에만 실행한다.

// Phase 2+에서 사용될 필드/타입들이 많으므로 dead_code 경고를 억제한다.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer, de};
use validator::{Validate, ValidationError};

use super::expr::{CompiledExpr, make_dry_run_context};

// ─────────────────────────────────────────────────────────
// Byte / Megabyte newtypes
// ─────────────────────────────────────────────────────────

/// 바이트 수를 나타내는 newtype.
/// YAML에서 정수 또는 "2 GiB", "512 MiB", "1024 KB" 형식의 문자열로 지정.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bytes(pub u64);

#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrU64 {
    Str(String),
    Int(u64),
}

impl<'de> Deserialize<'de> for Bytes {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = StringOrU64::deserialize(d)?;
        match v {
            StringOrU64::Int(n) => Ok(Bytes(n)),
            StringOrU64::Str(s) => parse_byte_string(&s).map(Bytes).map_err(de::Error::custom),
        }
    }
}

/// 메가바이트를 나타내는 newtype.
/// YAML에서 정수(MB 단위) 또는 "512 MB", "4096 MiB" 문자열로 지정.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Megabytes(pub u32);

impl<'de> Deserialize<'de> for Megabytes {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = StringOrU64::deserialize(d)?;
        match v {
            StringOrU64::Int(n) => Ok(Megabytes(n as u32)),
            StringOrU64::Str(s) => parse_megabyte_string(&s)
                .map(Megabytes)
                .map_err(de::Error::custom),
        }
    }
}

/// "2 GiB", "512 MiB", "1024 KB", "1024 B" 형식을 바이트 수로 변환.
/// 대소문자 구분 없음. 지원 단위: B, KB/KiB, MB/MiB, GB/GiB.
pub fn parse_byte_string(s: &str) -> Result<u64, String> {
    let s = s.trim();
    // 숫자/공백/단위로 분리
    let (num_part, unit_part) = split_num_unit(s)?;
    let n: f64 = num_part
        .trim()
        .parse()
        .map_err(|_| format!("cannot parse number in {:?}", s))?;
    let mul: u64 = match unit_part.trim().to_uppercase().as_str() {
        "B" | "" => 1,
        "KB" | "KIB" => 1024,
        "MB" | "MIB" => 1024 * 1024,
        "GB" | "GIB" => 1024 * 1024 * 1024,
        other => return Err(format!("unsupported byte unit {:?} in {:?}", other, s)),
    };
    Ok((n * mul as f64) as u64)
}

/// "512 MB", "4096 MiB" 형식을 MB 정수로 변환 (MB/MiB 단위만 허용).
pub fn parse_megabyte_string(s: &str) -> Result<u32, String> {
    let s = s.trim();
    let (num_part, unit_part) = split_num_unit(s)?;
    let n: f64 = num_part
        .trim()
        .parse()
        .map_err(|_| format!("cannot parse number in {:?}", s))?;
    match unit_part.trim().to_uppercase().as_str() {
        "MB" | "MIB" | "" => Ok(n as u32),
        other => Err(format!(
            "Megabytes only supports MB/MiB, got {:?} in {:?}",
            other, s
        )),
    }
}

fn split_num_unit(s: &str) -> Result<(&str, &str), String> {
    // 숫자(정수 or 소수)가 끝나는 위치를 찾는다
    let idx = s.find(|c: char| c.is_alphabetic()).unwrap_or(s.len());
    Ok((&s[..idx], &s[idx..]))
}

// ─────────────────────────────────────────────────────────
// ExprOrValue: f64 상수 또는 CompiledExpr
// ─────────────────────────────────────────────────────────

/// Effect의 value/factor에 사용. 문자열이면 expression, 숫자면 리터럴.
#[derive(Debug, Clone)]
pub enum ExprOrValue {
    Literal(f64),
    Expr(CompiledExpr),
    // 문자열 리터럴 (set op에서 device 이름 등)
    StringLiteral(String),
}

impl<'de> Deserialize<'de> for ExprOrValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Float(f64),
            Int(i64),
            Bool(bool),
            Str(String),
        }
        match Raw::deserialize(d)? {
            Raw::Float(f) => Ok(ExprOrValue::Literal(f)),
            Raw::Int(i) => Ok(ExprOrValue::Literal(i as f64)),
            Raw::Bool(b) => Ok(ExprOrValue::Literal(if b { 1.0 } else { 0.0 })),
            Raw::Str(s) => {
                // expression 컴파일 시도. 실패하면 문자열 리터럴로.
                match evalexpr::build_operator_tree(&s) {
                    Ok(tree) => {
                        // 순수 identifier/상수 expression인지 확인
                        // → 일단 CompiledExpr로 저장
                        Ok(ExprOrValue::Expr(CompiledExpr { source: s, tree }))
                    }
                    Err(_) => Ok(ExprOrValue::StringLiteral(s)),
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────
// Effect
// ─────────────────────────────────────────────────────────

/// Action이 특정 상태 차원에 미치는 효과.
#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Effect {
    Set {
        value: ExprOrValue,
        #[serde(default)]
        tau_s: f64,
    },
    Scale {
        factor: ExprOrValue,
        #[serde(default)]
        tau_s: f64,
    },
    Add {
        #[serde(alias = "delta")]
        value: ExprOrValue,
        #[serde(default)]
        tau_s: f64,
    },
    SetTau {
        value: f64,
    },
}

// ─────────────────────────────────────────────────────────
// ActionSpec
// ─────────────────────────────────────────────────────────

/// EngineCommand variant 하나에 대응하는 물리 효과 명세.
/// flatten + deny_unknown_fields 조합은 serde 제한으로 불가하므로 flatten만 사용.
#[derive(Deserialize, Debug, Clone)]
pub struct ActionSpec {
    /// 이 action이 적용되는 조건 (evalexpr). None이면 항상 적용.
    #[serde(default)]
    pub when: Option<CompiledExpr>,
    /// 차원 이름 → Effect 매핑 (나머지 필드).
    #[serde(flatten)]
    pub effects: HashMap<String, Effect>,
}

// ─────────────────────────────────────────────────────────
// Composition
// ─────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CompositionOp {
    Multiply,
    Add,
    AddDelta,
    Max,
    Min,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Composition {
    pub default: CompositionOp,
    #[serde(default)]
    pub per_dimension: HashMap<String, CompositionOp>,
}

// ─────────────────────────────────────────────────────────
// InteractionTerm
// ─────────────────────────────────────────────────────────

/// 특정 action 조합이 동시에 활성화될 때 추가로 적용되는 효과.
/// flatten + deny_unknown_fields 조합은 serde 제한으로 불가하므로 flatten만 사용.
#[derive(Deserialize, Debug, Clone)]
pub struct InteractionTerm {
    /// 조합에 참여해야 하는 action 이름 목록.
    pub actions: Vec<String>,
    /// 차원 이름 → Effect.
    #[serde(flatten)]
    pub effects: HashMap<String, Effect>,
}

// ─────────────────────────────────────────────────────────
// PassiveDynamics
// ─────────────────────────────────────────────────────────

/// 클러스터별 열역학 passive 파라미터.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ClusterThermalDynamics {
    /// 열 평형 온도 (°C).
    pub baseline: f64,
    /// 1st-order lag 시정수 (s).
    pub tau_s: f64,
    /// 부하 발열 계수.
    #[serde(default)]
    pub heating: HashMap<String, f64>,
}

/// 클러스터 간 열전도 계수.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ThermalCoupling {
    /// CPU→GPU 열전도 (°C/s per °C gap).
    pub cpu_to_gpu: f64,
    /// GPU→CPU 열전도 (°C/s per °C gap).
    pub gpu_to_cpu: f64,
}

/// KV 캐시 passive 성장 파라미터.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct KvCacheGrowth {
    /// dtype → 토큰당 바이트 수.
    pub growth_per_token: HashMap<String, u64>,
    /// 성장이 적용될 조건 expression.
    pub applies_when: CompiledExpr,
}

/// DerivedExpr: expression + 선택적 tau_s.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct DerivedExpr {
    pub expr: CompiledExpr,
    #[serde(default)]
    pub tau_s: f64,
}

/// Passive dynamics 전체 섹션.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct PassiveDynamics {
    pub cpu_cluster_thermal_c: ClusterThermalDynamics,
    pub gpu_cluster_thermal_c: ClusterThermalDynamics,
    pub thermal_coupling: ThermalCoupling,
    pub thermal_c: DerivedExpr,
    pub memory_bw_utilization_pct: DerivedExpr,
    pub kv_cache_bytes: KvCacheGrowth,
}

// ─────────────────────────────────────────────────────────
// DvfsConfig
// ─────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct DvfsCluster {
    pub soft_threshold_c: f64,
    pub hard_threshold_c: f64,
    pub k_thermal: f64,
    pub tau_s: f64,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct DvfsConfig {
    pub cpu: DvfsCluster,
    pub gpu: DvfsCluster,
}

// ─────────────────────────────────────────────────────────
// ExternalInjection
// ─────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ExternalInjection {
    /// 주입 시작 시각 (s).
    pub t_start: f64,
    /// 주입 지속 시간 (s).
    pub duration: f64,
    /// 적용할 상태 변수 이름.
    pub var: String,
    /// 연산자 ("add", "set" 등).
    pub op: String,
    /// add 연산의 delta (float).
    #[serde(default)]
    pub delta: Option<f64>,
    /// memory delta (MB 단위 alias).
    #[serde(default)]
    pub delta_mb: Option<f64>,
    #[serde(default)]
    pub tau_s: f64,
}

// ─────────────────────────────────────────────────────────
// ObservationConfig
// ─────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct NoiseSpec {
    #[serde(default)]
    pub sigma: Option<f64>,
    #[serde(default)]
    pub sigma_mb: Option<f64>,
    #[serde(default)]
    pub sigma_mc: Option<f64>,
    pub seed_key: String,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct HeartbeatObservation {
    pub interval_s: f64,
    #[serde(default)]
    pub noise: HashMap<String, NoiseSpec>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct SignalSourceConfig {
    pub source: String,
    pub poll_interval_s: f64,
    #[serde(default)]
    pub noise: HashMap<String, NoiseSpec>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct SignalsObservation {
    pub memory: SignalSourceConfig,
    pub compute: SignalSourceConfig,
    pub thermal: SignalSourceConfig,
    pub energy: SignalSourceConfig,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ObservationConfig {
    pub heartbeat: HeartbeatObservation,
    pub signals: SignalsObservation,
}

// ─────────────────────────────────────────────────────────
// InitialState
// ─────────────────────────────────────────────────────────

fn validate_kv_cache_capacity(state: &InitialState) -> Result<(), ValidationError> {
    if state.kv_cache_bytes.0 > state.kv_cache_capacity_bytes.0 {
        let mut e = ValidationError::new("kv_cache_bytes_exceeds_capacity");
        e.message = Some(
            format!(
                "kv_cache_bytes ({}) > kv_cache_capacity_bytes ({})",
                state.kv_cache_bytes.0, state.kv_cache_capacity_bytes.0
            )
            .into(),
        );
        return Err(e);
    }
    Ok(())
}

fn validate_device_memory(state: &InitialState) -> Result<(), ValidationError> {
    if state.device_memory_used_mb > state.device_memory_total_mb {
        let mut e = ValidationError::new("memory_used_exceeds_total");
        e.message = Some(
            format!(
                "device_memory_used_mb ({}) > device_memory_total_mb ({})",
                state.device_memory_used_mb, state.device_memory_total_mb
            )
            .into(),
        );
        return Err(e);
    }
    Ok(())
}

fn validate_kv_token_capacity(state: &InitialState) -> Result<(), ValidationError> {
    if state.kv_cache_tokens > state.kv_cache_token_capacity {
        let mut e = ValidationError::new("kv_tokens_exceeds_capacity");
        e.message = Some(
            format!(
                "kv_cache_tokens ({}) > kv_cache_token_capacity ({})",
                state.kv_cache_tokens, state.kv_cache_token_capacity
            )
            .into(),
        );
        return Err(e);
    }
    Ok(())
}

#[derive(Deserialize, Validate, Debug, Clone)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "validate_kv_cache_capacity"))]
#[validate(schema(function = "validate_device_memory"))]
#[validate(schema(function = "validate_kv_token_capacity"))]
pub struct InitialState {
    // KV
    pub kv_cache_bytes: Bytes,
    pub kv_cache_capacity_bytes: Bytes,
    pub kv_cache_tokens: u32,
    pub kv_cache_token_capacity: u32,
    pub kv_dtype: String,
    // Memory
    pub device_memory_total_mb: u32,
    pub device_memory_used_mb: u32,
    pub memory_bw_utilization_pct: f64,
    // CPU
    pub engine_cpu_pct: f64,
    pub external_cpu_pct: f64,
    pub cpu_freq_mhz: u32,
    pub cpu_max_freq_mhz: u32,
    pub cpu_min_freq_mhz: u32,
    // GPU
    pub engine_gpu_pct: f64,
    pub external_gpu_pct: f64,
    pub gpu_freq_mhz: u32,
    pub gpu_max_freq_mhz: u32,
    pub gpu_min_freq_mhz: u32,
    // Thermal
    pub thermal_c: f64,
    pub cpu_cluster_thermal_c: f64,
    pub gpu_cluster_thermal_c: f64,
    pub throttle_threshold_c: f64,
    // Phase / perf
    pub phase: String,
    pub base_tps_decode_gpu: f64,
    pub base_tps_decode_cpu: f64,
    pub base_tps_decode_partition: f64,
    pub base_tps_prefill_gpu: f64,
    // Engine 상태
    pub active_device: String,
    pub active_actions: Vec<String>,
    pub partition_ratio: f64,
    pub throttle_delay_ms: f64,
    pub tbt_target_ms: f64,
}

// ─────────────────────────────────────────────────────────
// Top-level ScenarioConfig
// ─────────────────────────────────────────────────────────

#[derive(Deserialize, Validate, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ScenarioConfig {
    /// 상속할 parent YAML 경로 (상대 경로, 로드 시 기준 디렉토리에서 해석).
    pub extends: Option<PathBuf>,
    #[validate(nested)]
    pub initial_state: InitialState,
    pub actions: HashMap<String, ActionSpec>,
    pub composition: Composition,
    #[serde(default)]
    pub interactions: Vec<InteractionTerm>,
    pub passive_dynamics: PassiveDynamics,
    pub dvfs: DvfsConfig,
    pub derived: HashMap<String, DerivedExpr>,
    #[serde(default)]
    pub external_injections: Vec<ExternalInjection>,
    pub observation: ObservationConfig,
    #[serde(default)]
    pub rng_seed: Option<u64>,
}

impl ScenarioConfig {
    /// 모든 CompiledExpr을 더미 context로 dry-run 평가하여 unknown identifier를 조기 탐지.
    pub fn validate_expressions(&self) -> Result<(), Vec<String>> {
        let ctx = make_dry_run_context();
        let mut errors = Vec::new();

        // passive_dynamics
        check_expr(
            &ctx,
            &self.passive_dynamics.thermal_c.expr,
            "passive_dynamics.thermal_c",
            &mut errors,
        );
        check_expr(
            &ctx,
            &self.passive_dynamics.memory_bw_utilization_pct.expr,
            "passive_dynamics.memory_bw_utilization_pct",
            &mut errors,
        );
        check_expr(
            &ctx,
            &self.passive_dynamics.kv_cache_bytes.applies_when,
            "passive_dynamics.kv_cache_bytes.applies_when",
            &mut errors,
        );

        // derived
        for (name, d) in &self.derived {
            check_expr(&ctx, &d.expr, &format!("derived.{}", name), &mut errors);
        }

        // actions: when + effect expressions
        for (action_name, spec) in &self.actions {
            if let Some(when) = &spec.when {
                check_expr_node(
                    &ctx,
                    when,
                    &format!("actions.{}.when", action_name),
                    &mut errors,
                );
            }
            for (dim, effect) in &spec.effects {
                let loc = format!("actions.{}.{}", action_name, dim);
                check_effect_exprs(&ctx, effect, &loc, &mut errors);
            }
        }

        // interactions
        for (i, term) in self.interactions.iter().enumerate() {
            for (dim, effect) in &term.effects {
                let loc = format!("interactions[{}].{}", i, dim);
                check_effect_exprs(&ctx, effect, &loc, &mut errors);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

fn check_expr(
    ctx: &super::expr::ExprContext,
    expr: &CompiledExpr,
    location: &str,
    errors: &mut Vec<String>,
) {
    check_expr_node(ctx, expr, location, errors);
}

fn check_expr_node(
    ctx: &super::expr::ExprContext,
    expr: &CompiledExpr,
    location: &str,
    errors: &mut Vec<String>,
) {
    // float/bool/string 중 하나라도 성공하면 통과.
    // 모두 실패한 경우에만 에러로 기록한다.
    if ctx.eval_f64(&expr.tree).is_ok() {
        return;
    }
    if ctx.eval_bool(&expr.tree).is_ok() {
        return;
    }
    if expr.tree.eval_string_with_context(&ctx.ctx).is_ok() {
        return;
    }
    // 모두 실패 — float 에러 메시지를 대표로 사용
    let err = ctx.eval_f64(&expr.tree).unwrap_err();
    errors.push(format!(
        "{}: expression {:?} dry-run failed: {}",
        location, expr.source, err
    ));
}

fn check_effect_exprs(
    ctx: &super::expr::ExprContext,
    effect: &Effect,
    location: &str,
    errors: &mut Vec<String>,
) {
    match effect {
        Effect::Set { value, .. } | Effect::Add { value, .. } => {
            if let ExprOrValue::Expr(e) = value {
                check_expr(ctx, e, location, errors);
            }
        }
        Effect::Scale { factor, .. } => {
            if let ExprOrValue::Expr(e) = factor {
                check_expr(ctx, e, location, errors);
            }
        }
        Effect::SetTau { .. } => {}
    }
}

// ─────────────────────────────────────────────────────────
// LoadError
// ─────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("IO error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("YAML parse error in {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("Validation error in {path}: {errors:?}")]
    Validation { path: PathBuf, errors: Vec<String> },
    #[error("Expression error in {path}: {errors:?}")]
    Expression { path: PathBuf, errors: Vec<String> },
    #[error("Circular extends detected: {cycle:?}")]
    CircularExtends { cycle: Vec<PathBuf> },
}

// ─────────────────────────────────────────────────────────
// load_scenario + deep_merge
// ─────────────────────────────────────────────────────────

/// YAML 파일을 로드하여 ScenarioConfig를 반환한다.
/// `extends` 키가 있으면 parent를 재귀적으로 로드하여 deep merge한다.
pub fn load_scenario<P: AsRef<Path>>(path: P) -> Result<ScenarioConfig, LoadError> {
    let path = path.as_ref();
    let base_dir = path.parent().unwrap_or(Path::new("."));
    let raw = load_yaml_with_extends(path, base_dir, &mut HashSet::new())?;
    let cfg: ScenarioConfig = serde_yaml::from_value(raw).map_err(|e| LoadError::Parse {
        path: path.to_path_buf(),
        source: e,
    })?;
    cfg.validate().map_err(|e| LoadError::Validation {
        path: path.to_path_buf(),
        errors: e
            .field_errors()
            .iter()
            .map(|(k, v)| format!("{}: {:?}", k, v))
            .chain(e.errors().iter().map(|(k, v)| format!("{}: {:?}", k, v)))
            .collect(),
    })?;
    cfg.validate_expressions()
        .map_err(|errors| LoadError::Expression {
            path: path.to_path_buf(),
            errors,
        })?;
    Ok(cfg)
}

fn load_yaml_with_extends(
    path: &Path,
    base_dir: &Path,
    visited: &mut HashSet<PathBuf>,
) -> Result<serde_yaml::Value, LoadError> {
    let canonical = path.canonicalize().unwrap_or_else(|_| base_dir.join(path));

    if visited.contains(&canonical) {
        return Err(LoadError::CircularExtends {
            cycle: visited.iter().cloned().collect(),
        });
    }
    visited.insert(canonical.clone());

    let content = std::fs::read_to_string(&canonical).map_err(|e| LoadError::Io {
        path: canonical.clone(),
        source: e,
    })?;
    let mut value: serde_yaml::Value =
        serde_yaml::from_str(&content).map_err(|e| LoadError::Parse {
            path: canonical.clone(),
            source: e,
        })?;

    // extends 처리
    if let Some(extends_val) = value.as_mapping_mut().and_then(|m| m.remove("extends")) {
        let parent_rel: PathBuf =
            serde_yaml::from_value(extends_val).map_err(|e| LoadError::Parse {
                path: canonical.clone(),
                source: e,
            })?;
        let parent_path = canonical
            .parent()
            .unwrap_or(Path::new("."))
            .join(&parent_rel);

        let parent_value = load_yaml_with_extends(&parent_path, base_dir, visited)?;
        // parent에서 child로 deep merge (child가 parent를 override)
        value = deep_merge(parent_value, value);
    }

    visited.remove(&canonical);
    Ok(value)
}

/// Mapping끼리는 재귀 merge, 그 외(Sequence, Scalar)는 override가 대체.
fn deep_merge(base: serde_yaml::Value, override_: serde_yaml::Value) -> serde_yaml::Value {
    match (base, override_) {
        (serde_yaml::Value::Mapping(mut b), serde_yaml::Value::Mapping(o)) => {
            for (k, v) in o {
                let entry = b.entry(k.clone());
                match entry {
                    serde_yaml::mapping::Entry::Occupied(mut oe) => {
                        let merged = deep_merge(oe.get().clone(), v);
                        *oe.get_mut() = merged;
                    }
                    serde_yaml::mapping::Entry::Vacant(ve) => {
                        ve.insert(v);
                    }
                }
            }
            serde_yaml::Value::Mapping(b)
        }
        // override가 우선
        (_, o) => o,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_byte_string_gib() {
        assert_eq!(parse_byte_string("2 GiB").unwrap(), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_byte_string_mib() {
        assert_eq!(parse_byte_string("512 MiB").unwrap(), 512 * 1024 * 1024);
    }

    #[test]
    fn test_parse_byte_string_int() {
        assert_eq!(parse_byte_string("1073741824").unwrap(), 1073741824);
    }

    #[test]
    fn test_parse_byte_string_invalid_unit() {
        assert!(parse_byte_string("5 PB").is_err());
    }

    #[test]
    fn test_deep_merge_basic() {
        let base: serde_yaml::Value = serde_yaml::from_str("a: 1\nb: 2").unwrap();
        let child: serde_yaml::Value = serde_yaml::from_str("b: 99").unwrap();
        let merged = deep_merge(base, child);
        let map = merged.as_mapping().unwrap();
        assert_eq!(map["a"], serde_yaml::Value::Number(1.into()));
        assert_eq!(map["b"], serde_yaml::Value::Number(99.into()));
    }
}
