//! PhysicalState + EngineStateModel 분리 선언.
//!
//! PhysicalState: 물리적으로 관측 가능한 연속 변수.
//! EngineStateModel: 엔진의 "의도" 상태 (active_device, active_actions, throttle_delay_ms 등).
//! Phase 2에서 PhysicalState.from_config(), EngineStateModel.apply_directive()를 구현한다.

#![allow(dead_code)]

use argus_shared::{EngineCommand, EngineDirective};

use super::config::InitialState;

// ─────────────────────────────────────────────────────────
// PhysicalState
// ─────────────────────────────────────────────────────────

/// 물리적으로 진화하는 연속 상태 변수.
/// 모든 수치를 f64로 저장하여 1차 수렴 연산이 매끄럽게 동작하도록 한다.
#[derive(Debug, Clone)]
pub struct PhysicalState {
    // KV cache
    pub kv_cache_bytes: f64,
    pub kv_cache_capacity_bytes: f64,
    pub kv_cache_tokens: f64,
    pub kv_cache_token_capacity: f64,
    pub kv_dtype: String,

    // Memory
    pub device_memory_total_mb: f64,
    pub device_memory_used_mb: f64,
    pub memory_bw_utilization_pct: f64,

    // CPU compute + freq
    pub engine_cpu_pct: f64,
    pub external_cpu_pct: f64,
    pub cpu_freq_mhz: f64,
    pub cpu_max_freq_mhz: f64,
    pub cpu_min_freq_mhz: f64,

    // GPU compute + freq
    pub engine_gpu_pct: f64,
    pub external_gpu_pct: f64,
    pub gpu_freq_mhz: f64,
    pub gpu_max_freq_mhz: f64,
    pub gpu_min_freq_mhz: f64,

    // Thermal
    pub thermal_c: f64,
    pub cpu_cluster_thermal_c: f64,
    pub gpu_cluster_thermal_c: f64,
    pub throttle_threshold_c: f64,

    // Base throughput constants (per-model, @ max freq, 0 contention)
    pub base_tps_decode_gpu: f64,
    pub base_tps_decode_cpu: f64,
    pub base_tps_decode_partition: f64,
    pub base_tps_prefill_gpu: f64,

    // Derived (physics 스텝 후 갱신)
    pub throughput_tps: f64,
    pub tbt_ms: f64,
    pub latency_degradation: f64,
}

impl PhysicalState {
    /// InitialState로부터 PhysicalState를 초기화한다.
    pub fn from_config(init: &InitialState) -> Self {
        PhysicalState {
            kv_cache_bytes: init.kv_cache_bytes.0 as f64,
            kv_cache_capacity_bytes: init.kv_cache_capacity_bytes.0 as f64,
            kv_cache_tokens: init.kv_cache_tokens as f64,
            kv_cache_token_capacity: init.kv_cache_token_capacity as f64,
            kv_dtype: init.kv_dtype.clone(),

            device_memory_total_mb: init.device_memory_total_mb as f64,
            device_memory_used_mb: init.device_memory_used_mb as f64,
            memory_bw_utilization_pct: init.memory_bw_utilization_pct,

            engine_cpu_pct: init.engine_cpu_pct,
            external_cpu_pct: init.external_cpu_pct,
            cpu_freq_mhz: init.cpu_freq_mhz as f64,
            cpu_max_freq_mhz: init.cpu_max_freq_mhz as f64,
            cpu_min_freq_mhz: init.cpu_min_freq_mhz as f64,

            engine_gpu_pct: init.engine_gpu_pct,
            external_gpu_pct: init.external_gpu_pct,
            gpu_freq_mhz: init.gpu_freq_mhz as f64,
            gpu_max_freq_mhz: init.gpu_max_freq_mhz as f64,
            gpu_min_freq_mhz: init.gpu_min_freq_mhz as f64,

            thermal_c: init.thermal_c,
            cpu_cluster_thermal_c: init.cpu_cluster_thermal_c,
            gpu_cluster_thermal_c: init.gpu_cluster_thermal_c,
            throttle_threshold_c: init.throttle_threshold_c,

            base_tps_decode_gpu: init.base_tps_decode_gpu,
            base_tps_decode_cpu: init.base_tps_decode_cpu,
            base_tps_decode_partition: init.base_tps_decode_partition,
            base_tps_prefill_gpu: init.base_tps_prefill_gpu,

            throughput_tps: 0.0,
            tbt_ms: 0.0,
            latency_degradation: 0.0,
        }
    }

    /// 모든 f64 필드를 evalexpr context에 바인딩한다.
    /// string 필드는 별도 set_str 호출이 필요하다.
    pub fn bind_to_context(&self, ctx: &mut super::expr::ExprContext) {
        ctx.set_f64("kv_cache_bytes", self.kv_cache_bytes);
        ctx.set_f64("kv_cache_capacity_bytes", self.kv_cache_capacity_bytes);
        ctx.set_f64("kv_cache_tokens", self.kv_cache_tokens);
        ctx.set_f64("kv_cache_token_capacity", self.kv_cache_token_capacity);
        ctx.set_f64("device_memory_total_mb", self.device_memory_total_mb);
        ctx.set_f64("device_memory_used_mb", self.device_memory_used_mb);
        ctx.set_f64("memory_bw_utilization_pct", self.memory_bw_utilization_pct);
        ctx.set_f64("engine_cpu_pct", self.engine_cpu_pct);
        ctx.set_f64("external_cpu_pct", self.external_cpu_pct);
        ctx.set_f64("cpu_freq_mhz", self.cpu_freq_mhz);
        ctx.set_f64("cpu_max_freq_mhz", self.cpu_max_freq_mhz);
        ctx.set_f64("cpu_min_freq_mhz", self.cpu_min_freq_mhz);
        ctx.set_f64("engine_gpu_pct", self.engine_gpu_pct);
        ctx.set_f64("external_gpu_pct", self.external_gpu_pct);
        ctx.set_f64("gpu_freq_mhz", self.gpu_freq_mhz);
        ctx.set_f64("gpu_max_freq_mhz", self.gpu_max_freq_mhz);
        ctx.set_f64("gpu_min_freq_mhz", self.gpu_min_freq_mhz);
        ctx.set_f64("thermal_c", self.thermal_c);
        ctx.set_f64("cpu_cluster_thermal_c", self.cpu_cluster_thermal_c);
        ctx.set_f64("gpu_cluster_thermal_c", self.gpu_cluster_thermal_c);
        ctx.set_f64("throttle_threshold_c", self.throttle_threshold_c);
        ctx.set_f64("base_tps_decode_gpu", self.base_tps_decode_gpu);
        ctx.set_f64("base_tps_decode_cpu", self.base_tps_decode_cpu);
        ctx.set_f64("base_tps_decode_partition", self.base_tps_decode_partition);
        ctx.set_f64("base_tps_prefill_gpu", self.base_tps_prefill_gpu);
        ctx.set_f64("throughput_tps", self.throughput_tps);
        ctx.set_f64("tbt_ms", self.tbt_ms);
        ctx.set_f64("latency_degradation", self.latency_degradation);

        ctx.set_str("kv_dtype", &self.kv_dtype);
    }

    /// 필드 이름으로 f64 값을 가져온다.
    pub fn get_f64(&self, field: &str) -> Option<f64> {
        match field {
            "kv_cache_bytes" => Some(self.kv_cache_bytes),
            "kv_cache_capacity_bytes" => Some(self.kv_cache_capacity_bytes),
            "kv_cache_tokens" => Some(self.kv_cache_tokens),
            "kv_cache_token_capacity" => Some(self.kv_cache_token_capacity),
            "device_memory_total_mb" => Some(self.device_memory_total_mb),
            "device_memory_used_mb" => Some(self.device_memory_used_mb),
            "memory_bw_utilization_pct" => Some(self.memory_bw_utilization_pct),
            "engine_cpu_pct" => Some(self.engine_cpu_pct),
            "external_cpu_pct" => Some(self.external_cpu_pct),
            "cpu_freq_mhz" => Some(self.cpu_freq_mhz),
            "cpu_max_freq_mhz" => Some(self.cpu_max_freq_mhz),
            "cpu_min_freq_mhz" => Some(self.cpu_min_freq_mhz),
            "engine_gpu_pct" => Some(self.engine_gpu_pct),
            "external_gpu_pct" => Some(self.external_gpu_pct),
            "gpu_freq_mhz" => Some(self.gpu_freq_mhz),
            "gpu_max_freq_mhz" => Some(self.gpu_max_freq_mhz),
            "gpu_min_freq_mhz" => Some(self.gpu_min_freq_mhz),
            "thermal_c" => Some(self.thermal_c),
            "cpu_cluster_thermal_c" => Some(self.cpu_cluster_thermal_c),
            "gpu_cluster_thermal_c" => Some(self.gpu_cluster_thermal_c),
            "throttle_threshold_c" => Some(self.throttle_threshold_c),
            "base_tps_decode_gpu" => Some(self.base_tps_decode_gpu),
            "base_tps_decode_cpu" => Some(self.base_tps_decode_cpu),
            "base_tps_decode_partition" => Some(self.base_tps_decode_partition),
            "base_tps_prefill_gpu" => Some(self.base_tps_prefill_gpu),
            "throughput_tps" => Some(self.throughput_tps),
            "tbt_ms" => Some(self.tbt_ms),
            "latency_degradation" => Some(self.latency_degradation),
            _ => None,
        }
    }

    /// 필드 이름으로 f64 값을 설정한다.
    pub fn set_f64(&mut self, field: &str, value: f64) -> bool {
        match field {
            "kv_cache_bytes" => self.kv_cache_bytes = value,
            "kv_cache_capacity_bytes" => self.kv_cache_capacity_bytes = value,
            "kv_cache_tokens" => self.kv_cache_tokens = value,
            "kv_cache_token_capacity" => self.kv_cache_token_capacity = value,
            "device_memory_total_mb" => self.device_memory_total_mb = value,
            "device_memory_used_mb" => self.device_memory_used_mb = value,
            "memory_bw_utilization_pct" => self.memory_bw_utilization_pct = value,
            "engine_cpu_pct" => self.engine_cpu_pct = value,
            "external_cpu_pct" => self.external_cpu_pct = value,
            "cpu_freq_mhz" => self.cpu_freq_mhz = value,
            "cpu_max_freq_mhz" => self.cpu_max_freq_mhz = value,
            "cpu_min_freq_mhz" => self.cpu_min_freq_mhz = value,
            "engine_gpu_pct" => self.engine_gpu_pct = value,
            "external_gpu_pct" => self.external_gpu_pct = value,
            "gpu_freq_mhz" => self.gpu_freq_mhz = value,
            "gpu_max_freq_mhz" => self.gpu_max_freq_mhz = value,
            "gpu_min_freq_mhz" => self.gpu_min_freq_mhz = value,
            "thermal_c" => self.thermal_c = value,
            "cpu_cluster_thermal_c" => self.cpu_cluster_thermal_c = value,
            "gpu_cluster_thermal_c" => self.gpu_cluster_thermal_c = value,
            "throttle_threshold_c" => self.throttle_threshold_c = value,
            "base_tps_decode_gpu" => self.base_tps_decode_gpu = value,
            "base_tps_decode_cpu" => self.base_tps_decode_cpu = value,
            "base_tps_decode_partition" => self.base_tps_decode_partition = value,
            "base_tps_prefill_gpu" => self.base_tps_prefill_gpu = value,
            "throughput_tps" => self.throughput_tps = value,
            "tbt_ms" => self.tbt_ms = value,
            "latency_degradation" => self.latency_degradation = value,
            _ => return false,
        }
        true
    }
}

// ─────────────────────────────────────────────────────────
// EngineStateModel
// ─────────────────────────────────────────────────────────

/// 엔진의 "의도" 상태 모델.
/// apply_directive()가 EngineDirective를 수신하면 의도를 기록한다.
/// 실제 물리 효과는 compose.rs에서 처리한다.
#[derive(Debug, Clone)]
pub struct EngineStateModel {
    pub active_device: String,
    pub active_actions: Vec<String>,
    pub phase: String,
    pub partition_ratio: f64,
    pub throttle_delay_ms: f64,
    pub tbt_target_ms: f64,
    pub skip_ratio: f64,
    pub kv_quant_bits: Option<u8>,

    // Action 파라미터 캐시 (compose.rs에서 참조)
    pub last_evict_ratio: Option<f64>,
    pub last_switch_device: Option<String>,
}

impl EngineStateModel {
    /// InitialState로부터 EngineStateModel을 초기화한다.
    pub fn from_config(init: &InitialState) -> Self {
        EngineStateModel {
            active_device: init.active_device.clone(),
            active_actions: init.active_actions.clone(),
            phase: init.phase.clone(),
            partition_ratio: init.partition_ratio,
            throttle_delay_ms: init.throttle_delay_ms,
            tbt_target_ms: init.tbt_target_ms,
            skip_ratio: 0.0,
            kv_quant_bits: None,
            last_evict_ratio: None,
            last_switch_device: None,
        }
    }

    /// EngineDirective를 수신하여 의도 상태를 갱신한다.
    /// 실제 물리 효과는 compose.rs에서 담당하고, 여기서는 의도만 기록한다.
    pub fn apply_directive(&mut self, directive: &EngineDirective, state: &mut PhysicalState) {
        for cmd in &directive.commands {
            self.apply_command(cmd, state);
        }
    }

    fn apply_command(&mut self, cmd: &EngineCommand, state: &mut PhysicalState) {
        match cmd {
            // ── One-shot 액션 (KV cache eviction/quantization) ──
            // 즉시 적용 후 종료되는 액션이므로 active_actions에 등록하지 않는다.
            // 등록하면 is_active_any() 체크가 재발행을 영구 차단하여
            // 메모리가 계속 높아도 추가 eviction이 일어나지 않는 문제가 발생한다.
            EngineCommand::KvEvictH2o { keep_ratio } => {
                self.last_evict_ratio = Some(*keep_ratio as f64);
            }
            EngineCommand::KvEvictSliding { keep_ratio } => {
                self.last_evict_ratio = Some(*keep_ratio as f64);
            }
            EngineCommand::KvMergeD2o { keep_ratio } => {
                self.last_evict_ratio = Some(*keep_ratio as f64);
            }
            EngineCommand::KvQuantDynamic { target_bits } => {
                self.kv_quant_bits = Some(*target_bits);
                state.kv_dtype = bits_to_dtype(*target_bits);
            }
            // ── Persistent 모드 액션 ──
            // RestoreDefaults가 올 때까지 유지되는 액션. active_actions에 등록.
            // 이름은 Lua 정책과 일치하도록 snake_case 사용.
            EngineCommand::Throttle { delay_ms } => {
                self.throttle_delay_ms = *delay_ms as f64;
                add_action(&mut self.active_actions, "throttle");
            }
            EngineCommand::SwitchHw { device } => {
                self.last_switch_device = Some(device.clone());
                self.active_device = device.clone();
                add_action(&mut self.active_actions, "switch_hw");
            }
            EngineCommand::SetPartitionRatio { ratio } => {
                self.partition_ratio = *ratio as f64;
                add_action(&mut self.active_actions, "set_partition_ratio");
            }
            EngineCommand::RestoreDefaults => {
                self.active_actions.clear();
                self.throttle_delay_ms = 0.0;
                self.skip_ratio = 0.0;
                self.partition_ratio = 0.0;
                self.last_evict_ratio = None;
            }
            EngineCommand::LayerSkip { skip_ratio } => {
                self.skip_ratio = *skip_ratio as f64;
                add_action(&mut self.active_actions, "weight.skip");
            }
            EngineCommand::SetTargetTbt { target_ms } => {
                self.tbt_target_ms = *target_ms as f64;
            }
            // 기타 명령은 물리 시뮬레이터에서 직접 처리하지 않음
            _ => {}
        }
    }

    /// evalexpr context에 engine 상태 변수를 바인딩한다.
    pub fn bind_to_context(&self, ctx: &mut super::expr::ExprContext) {
        ctx.set_str("active_device", &self.active_device);
        ctx.set_str("phase", &self.phase);
        ctx.set_f64("partition_ratio", self.partition_ratio);
        ctx.set_f64("throttle_delay_ms", self.throttle_delay_ms);
        ctx.set_f64("tbt_target_ms", self.tbt_target_ms);
        ctx.set_f64("skip_ratio", self.skip_ratio);

        // action 파라미터 (compose.rs에서 참조)
        if let Some(ratio) = self.last_evict_ratio {
            ctx.set_f64("target_ratio", ratio);
        } else {
            ctx.set_f64("target_ratio", 0.0);
        }
        ctx.set_f64("delay_ms", self.throttle_delay_ms);
        ctx.set_f64("ratio", self.partition_ratio);
        ctx.set_str(
            "device",
            self.last_switch_device
                .as_deref()
                .unwrap_or(&self.active_device),
        );
    }
}

// ─────────────────────────────────────────────────────────
// 헬퍼 함수
// ─────────────────────────────────────────────────────────

fn add_action(actions: &mut Vec<String>, name: &str) {
    if !actions.iter().any(|a| a == name) {
        actions.push(name.to_string());
    }
}

/// bits → dtype 문자열 변환 (KvQuantDynamic 처리용).
pub fn bits_to_dtype(bits: u8) -> String {
    match bits {
        2 => "q2".to_string(),
        4 => "q4".to_string(),
        8 => "q8".to_string(),
        16 => "f16".to_string(),
        32 => "f32".to_string(),
        _ => "f16".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_to_dtype() {
        assert_eq!(bits_to_dtype(4), "q4");
        assert_eq!(bits_to_dtype(8), "q8");
        assert_eq!(bits_to_dtype(16), "f16");
        assert_eq!(bits_to_dtype(32), "f32");
    }

    #[test]
    fn test_apply_directive_restore_defaults() {
        use argus_shared::EngineCommand;

        let init = make_test_initial_state();
        let mut engine = EngineStateModel::from_config(&init);
        let mut state = PhysicalState::from_config(&init);

        engine.apply_command(&EngineCommand::Throttle { delay_ms: 100 }, &mut state);
        assert!(!engine.active_actions.is_empty());

        engine.apply_command(&EngineCommand::RestoreDefaults, &mut state);
        assert!(engine.active_actions.is_empty());
        assert_eq!(engine.throttle_delay_ms, 0.0);
    }

    fn make_test_initial_state() -> InitialState {
        use super::super::config::{Bytes, InitialState};
        InitialState {
            kv_cache_bytes: Bytes(0),
            kv_cache_capacity_bytes: Bytes(4 * 1024 * 1024 * 1024),
            kv_cache_tokens: 0,
            kv_cache_token_capacity: 2048,
            kv_dtype: "f16".to_string(),
            device_memory_total_mb: 8192,
            device_memory_used_mb: 3500,
            memory_bw_utilization_pct: 20.0,
            engine_cpu_pct: 30.0,
            external_cpu_pct: 15.0,
            cpu_freq_mhz: 2400,
            cpu_max_freq_mhz: 3200,
            cpu_min_freq_mhz: 400,
            engine_gpu_pct: 65.0,
            external_gpu_pct: 5.0,
            gpu_freq_mhz: 810,
            gpu_max_freq_mhz: 1100,
            gpu_min_freq_mhz: 300,
            thermal_c: 42.0,
            cpu_cluster_thermal_c: 44.0,
            gpu_cluster_thermal_c: 48.0,
            throttle_threshold_c: 85.0,
            phase: "idle".to_string(),
            base_tps_decode_gpu: 18.5,
            base_tps_decode_cpu: 4.2,
            base_tps_decode_partition: 22.0,
            base_tps_prefill_gpu: 145.0,
            active_device: "opencl".to_string(),
            active_actions: vec![],
            partition_ratio: 0.0,
            throttle_delay_ms: 0.0,
            tbt_target_ms: 0.0,
        }
    }
}
