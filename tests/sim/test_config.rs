//! YAML 스키마 파싱/검증/상속 단위 테스트 (10개).

use std::path::PathBuf;

use crate::common::sim::config::{LoadError, load_scenario, parse_byte_string};

fn fixtures_dir() -> PathBuf {
    // tests/fixtures/sim/
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sim")
}

// ─── 1. baseline.yaml 파싱/검증 성공 ─────────────────────────────────────────

#[test]
fn test_load_baseline_succeeds() {
    let path = fixtures_dir().join("baseline.yaml");
    let cfg = load_scenario(&path).expect("baseline.yaml should load successfully");
    // 기본값 확인
    assert_eq!(cfg.initial_state.active_device, "opencl");
    assert_eq!(cfg.initial_state.phase, "idle");
    assert!(cfg.rng_seed.is_none());
    // actions 존재 확인
    assert!(cfg.actions.contains_key("Evict"));
    assert!(cfg.actions.contains_key("Throttle"));
}

// ─── 2. extends deep merge ────────────────────────────────────────────────────

#[test]
fn test_extends_deep_merge() {
    let path = fixtures_dir().join("s25_override.yaml");
    let cfg = load_scenario(&path).expect("s25_override.yaml should load");
    // child override
    assert_eq!(cfg.initial_state.cpu_max_freq_mhz, 2800);
    // parent 값 유지
    assert_eq!(cfg.initial_state.gpu_max_freq_mhz, 1100);
    assert_eq!(cfg.initial_state.active_device, "opencl");
    // actions도 parent에서 상속됨
    assert!(cfg.actions.contains_key("Evict"));
}

// ─── 3. Bytes 파싱 성공 ──────────────────────────────────────────────────────

#[test]
fn test_bytes_parse() {
    assert_eq!(
        parse_byte_string("2 GiB").unwrap(),
        2u64 * 1024 * 1024 * 1024
    );
    assert_eq!(parse_byte_string("512 MB").unwrap(), 512u64 * 1024 * 1024);
    assert_eq!(parse_byte_string("1073741824").unwrap(), 1073741824u64);
}

// ─── 4. 잘못된 단위 실패 ─────────────────────────────────────────────────────

#[test]
fn test_bytes_parse_invalid_unit() {
    let result = parse_byte_string("5 PB");
    assert!(result.is_err(), "PB 단위는 지원하지 않으므로 에러여야 함");
}

// ─── 5. deny_unknown_fields ──────────────────────────────────────────────────

#[test]
fn test_deny_unknown_fields() {
    // unknown_key를 가진 임시 YAML을 인라인으로 테스트
    // baseline에 직접 unknown_key를 삽입한 variant를 serde_yaml로 파싱
    let fixtures = fixtures_dir();
    let baseline_content = std::fs::read_to_string(fixtures.join("baseline.yaml")).unwrap();
    // unknown 필드 추가
    let patched = format!("{}\nunknown_key: 42\n", baseline_content);
    let result: Result<crate::common::sim::config::ScenarioConfig, _> =
        serde_yaml::from_str(&patched);
    assert!(
        result.is_err(),
        "unknown_key가 있으면 deny_unknown_fields로 파싱 실패해야 함"
    );
}

// ─── 6. cross-field invariant 검증 ──────────────────────────────────────────

#[test]
fn test_invariant_validate() {
    // device_memory_used_mb > device_memory_total_mb → validator 실패
    let fixtures = fixtures_dir();
    let baseline_content = std::fs::read_to_string(fixtures.join("baseline.yaml")).unwrap();

    // device_memory_used_mb를 total보다 크게 설정
    let patched = baseline_content.replace(
        "device_memory_used_mb: 3500",
        "device_memory_used_mb: 99999",
    );

    // serde_yaml::Value로 파싱 후 ScenarioConfig 역직렬화 → validator 에러
    let raw: serde_yaml::Value = serde_yaml::from_str(&patched).unwrap();
    // extends 없으므로 바로 deserialize
    let cfg: crate::common::sim::config::ScenarioConfig =
        serde_yaml::from_value(raw).expect("serde parse should succeed");
    // validate는 실패해야 함
    use validator::Validate;
    let result = cfg.validate();
    assert!(
        result.is_err(),
        "used_mb > total_mb 이면 validator 실패해야 함"
    );
}

// ─── 7. expression 컴파일 실패 ───────────────────────────────────────────────

#[test]
fn test_expression_compile_failure() {
    // 잘못된 expression ("1 +") 이 들어간 YAML을 파싱하면 실패
    let yaml = r#"
initial_state:
  kv_cache_bytes: 0
  kv_cache_capacity_bytes: 4294967296
  kv_cache_tokens: 0
  kv_cache_token_capacity: 2048
  kv_dtype: "f16"
  device_memory_total_mb: 8192
  device_memory_used_mb: 3500
  memory_bw_utilization_pct: 20.0
  engine_cpu_pct: 30.0
  external_cpu_pct: 15.0
  cpu_freq_mhz: 2400
  cpu_max_freq_mhz: 3200
  cpu_min_freq_mhz: 400
  engine_gpu_pct: 65.0
  external_gpu_pct: 5.0
  gpu_freq_mhz: 810
  gpu_max_freq_mhz: 1100
  gpu_min_freq_mhz: 300
  thermal_c: 42.0
  cpu_cluster_thermal_c: 44.0
  gpu_cluster_thermal_c: 48.0
  throttle_threshold_c: 85.0
  phase: "idle"
  base_tps_decode_gpu: 18.5
  base_tps_decode_cpu: 4.2
  base_tps_decode_partition: 22.0
  base_tps_prefill_gpu: 145.0
  active_device: "opencl"
  active_actions: []
  partition_ratio: 0.0
  throttle_delay_ms: 0.0
  tbt_target_ms: 0.0
actions: {}
composition:
  default: multiply
passive_dynamics:
  cpu_cluster_thermal_c:
    baseline: 42.0
    tau_s: 20.0
  gpu_cluster_thermal_c:
    baseline: 44.0
    tau_s: 18.0
  thermal_coupling:
    cpu_to_gpu: 0.15
    gpu_to_cpu: 0.12
  thermal_c:
    expr: "1 2"
  memory_bw_utilization_pct:
    expr: "0.5"
    tau_s: 0.5
  kv_cache_bytes:
    growth_per_token:
      f16: 131072
    applies_when: "true"
dvfs:
  cpu:
    soft_threshold_c: 78.0
    hard_threshold_c: 95.0
    k_thermal: 0.08
    tau_s: 0.2
  gpu:
    soft_threshold_c: 72.0
    hard_threshold_c: 90.0
    k_thermal: 0.10
    tau_s: 0.3
derived: {}
observation:
  heartbeat:
    interval_s: 1.0
  signals:
    memory:
      source: os_probe
      poll_interval_s: 0.5
    compute:
      source: os_probe
      poll_interval_s: 0.25
    thermal:
      source: os_probe
      poll_interval_s: 1.0
    energy:
      source: os_probe
      poll_interval_s: 2.0
"#;
    let result: Result<crate::common::sim::config::ScenarioConfig, _> = serde_yaml::from_str(yaml);
    assert!(
        result.is_err(),
        "잘못된 expression ('1 2') 은 deserialization 단계에서 컴파일 실패해야 함"
    );
}

// ─── 8. dry-run unknown variable 참조 실패 ───────────────────────────────────

#[test]
fn test_expression_dry_run_unknown_ref() {
    use crate::common::sim::config::ScenarioConfig;

    // validate_expressions()가 unknown variable 참조를 잡아야 함
    // derived에 nonexistent_field 참조
    let yaml = r#"
initial_state:
  kv_cache_bytes: 0
  kv_cache_capacity_bytes: 4294967296
  kv_cache_tokens: 0
  kv_cache_token_capacity: 2048
  kv_dtype: "f16"
  device_memory_total_mb: 8192
  device_memory_used_mb: 3500
  memory_bw_utilization_pct: 20.0
  engine_cpu_pct: 30.0
  external_cpu_pct: 15.0
  cpu_freq_mhz: 2400
  cpu_max_freq_mhz: 3200
  cpu_min_freq_mhz: 400
  engine_gpu_pct: 65.0
  external_gpu_pct: 5.0
  gpu_freq_mhz: 810
  gpu_max_freq_mhz: 1100
  gpu_min_freq_mhz: 300
  thermal_c: 42.0
  cpu_cluster_thermal_c: 44.0
  gpu_cluster_thermal_c: 48.0
  throttle_threshold_c: 85.0
  phase: "idle"
  base_tps_decode_gpu: 18.5
  base_tps_decode_cpu: 4.2
  base_tps_decode_partition: 22.0
  base_tps_prefill_gpu: 145.0
  active_device: "opencl"
  active_actions: []
  partition_ratio: 0.0
  throttle_delay_ms: 0.0
  tbt_target_ms: 0.0
actions: {}
composition:
  default: multiply
passive_dynamics:
  cpu_cluster_thermal_c:
    baseline: 42.0
    tau_s: 20.0
  gpu_cluster_thermal_c:
    baseline: 44.0
    tau_s: 18.0
  thermal_coupling:
    cpu_to_gpu: 0.15
    gpu_to_cpu: 0.12
  thermal_c:
    expr: "42.0"
  memory_bw_utilization_pct:
    expr: "0.5"
    tau_s: 0.5
  kv_cache_bytes:
    growth_per_token:
      f16: 131072
    applies_when: "true"
dvfs:
  cpu:
    soft_threshold_c: 78.0
    hard_threshold_c: 95.0
    k_thermal: 0.08
    tau_s: 0.2
  gpu:
    soft_threshold_c: 72.0
    hard_threshold_c: 90.0
    k_thermal: 0.10
    tau_s: 0.3
derived:
  bad_expr:
    expr: "nonexistent_field * 2.0"
observation:
  heartbeat:
    interval_s: 1.0
  signals:
    memory:
      source: os_probe
      poll_interval_s: 0.5
    compute:
      source: os_probe
      poll_interval_s: 0.25
    thermal:
      source: os_probe
      poll_interval_s: 1.0
    energy:
      source: os_probe
      poll_interval_s: 2.0
"#;
    let cfg: ScenarioConfig = serde_yaml::from_str(yaml).expect("parse should succeed");
    let result = cfg.validate_expressions();
    assert!(
        result.is_err(),
        "nonexistent_field 참조는 validate_expressions에서 실패해야 함"
    );
    let errors = result.unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("nonexistent_field") || e.contains("bad_expr")),
        "에러 메시지에 문제 위치 포함: {:?}",
        errors
    );
}

// ─── 9. 순환 extends 감지 ────────────────────────────────────────────────────

#[test]
fn test_circular_extends_detected() {
    let path = fixtures_dir().join("circular_a.yaml");
    let result = load_scenario(&path);
    assert!(result.is_err(), "순환 extends는 LoadError로 실패해야 함");
    match result.unwrap_err() {
        LoadError::CircularExtends { .. } => {} // 예상된 에러
        other => panic!("CircularExtends 에러를 기대했지만 {:?} 가 반환됨", other),
    }
}

// ─── 10. rng_seed 미설정 시 None ─────────────────────────────────────────────

#[test]
fn test_rng_seed_optional_defaults_null() {
    let path = fixtures_dir().join("baseline.yaml");
    let cfg = load_scenario(&path).expect("baseline.yaml should load");
    assert!(
        cfg.rng_seed.is_none(),
        "baseline.yaml은 rng_seed: ~ 이므로 None이어야 함"
    );
}
