use crate::config::TriggerConfig;
use serde::{Deserialize, Serialize};

/// 6D relief 벡터 차원 수 (gpu, cpu, memory, thermal, latency, main_app_qos).
pub const RELIEF_DIMS: usize = 6;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Pressure6D {
    pub gpu: f32,
    pub cpu: f32,
    pub memory: f32,
    pub thermal: f32,
    pub latency: f32,
    pub main_app: f32,
}

impl From<Pressure6D> for [f32; 6] {
    fn from(p: Pressure6D) -> Self {
        [p.gpu, p.cpu, p.memory, p.thermal, p.latency, p.main_app]
    }
}

#[derive(Debug, Default)]
pub struct SignalState {
    pub cpu_pct: f64,
    pub gpu_pct: f64,
    pub mem_available: u64,
    pub mem_total: u64,
    pub temp_mc: i32,
    pub throttling: bool,
}

impl SignalState {
    pub fn update_compute(&mut self, cpu_pct: f64, gpu_pct: f64) {
        self.cpu_pct = cpu_pct;
        self.gpu_pct = gpu_pct;
    }

    pub fn update_memory(&mut self, available: u64, total: u64) {
        self.mem_available = available;
        self.mem_total = total;
    }

    pub fn update_thermal(&mut self, temp_mc: i32, throttling: bool) {
        self.temp_mc = temp_mc;
        self.throttling = throttling;
    }

    pub fn pressure_with_thermal(
        &self,
        temp_safe_c: f32,
        temp_critical_c: f32,
        latency_ratio: Option<f64>,
    ) -> Pressure6D {
        let mem_pressure = if self.mem_total > 0 {
            1.0 - (self.mem_available as f32 / self.mem_total as f32)
        } else {
            0.0
        };

        let temp_c = self.temp_mc as f32 / 1000.0;
        let temp_range = temp_critical_c - temp_safe_c;
        let thermal = if temp_range > 0.0 {
            ((temp_c - temp_safe_c) / temp_range).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Pressure6D {
            gpu: (self.gpu_pct as f32 / 100.0).clamp(0.0, 1.0),
            cpu: (self.cpu_pct as f32 / 100.0).clamp(0.0, 1.0),
            memory: mem_pressure.clamp(0.0, 1.0),
            thermal,
            latency: latency_ratio.unwrap_or(0.0) as f32,
            main_app: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TriggerState {
    pub tbt_degraded: bool,
    pub mem_low: bool,
    pub temp_high: bool,
}

#[derive(Debug)]
pub struct TbtTracker {
    pub ewma: f64,
    pub baseline: Option<f64>,
    pub warmup_count: u32,
    pub warmup_target: u32,
}

impl TbtTracker {
    pub fn new(warmup_target: u32) -> Self {
        Self {
            ewma: 0.0,
            baseline: None,
            warmup_count: 0,
            warmup_target,
        }
    }

    pub fn observe(&mut self, tbt_ms: f64) {
        if self.warmup_count == 0 {
            self.ewma = tbt_ms;
        } else {
            self.ewma = 0.875 * self.ewma + 0.125 * tbt_ms;
        }
        self.warmup_count += 1;

        if self.baseline.is_none() && self.warmup_count >= self.warmup_target {
            self.baseline = Some(self.ewma);
        }
    }

    pub fn degradation_ratio(&self) -> Option<f64> {
        self.baseline
            .map(|b| if b > 0.0 { (self.ewma - b) / b } else { 0.0 })
    }
}

#[derive(Debug)]
pub struct TriggerEngine {
    pub config: TriggerConfig,
    pub tbt: TbtTracker,
    pub trigger: TriggerState,
}

impl TriggerEngine {
    pub fn new(config: TriggerConfig) -> Self {
        Self {
            tbt: TbtTracker::new(config.tbt_warmup_tokens),
            config,
            trigger: TriggerState::default(),
        }
    }

    /// Feed the heartbeat's time-between-tokens.
    ///
    /// This took tokens per second and inverted it while the heartbeat reported
    /// throughput. The heartbeat reports ms/token now, so the reciprocal — and the
    /// division-by-zero it needed guarding for — is gone.
    pub fn update_tbt_ms(&mut self, tbt_ms: f64) {
        // NaN and non-positive alike: neither is an observation.
        if !tbt_ms.is_finite() || tbt_ms <= 0.0 {
            return;
        }
        self.tbt.observe(tbt_ms);

        if let Some(ratio) = self.tbt.degradation_ratio() {
            if self.trigger.tbt_degraded {
                if ratio < self.config.tbt_exit {
                    self.trigger.tbt_degraded = false;
                }
            } else if ratio > self.config.tbt_enter {
                self.trigger.tbt_degraded = true;
            }
        }
    }

    pub fn update_mem(&mut self, pressure: f64) {
        if self.trigger.mem_low {
            if pressure < self.config.mem_exit {
                self.trigger.mem_low = false;
            }
        } else if pressure > self.config.mem_enter {
            self.trigger.mem_low = true;
        }
    }

    pub fn update_temp(&mut self, normalized: f64) {
        if self.trigger.temp_high {
            if normalized < self.config.temp_exit {
                self.trigger.temp_high = false;
            }
        } else if normalized > self.config.temp_enter {
            self.trigger.temp_high = true;
        }
    }

    pub fn state(&self) -> &TriggerState {
        &self.trigger
    }

    pub fn tbt_degradation_ratio(&self) -> Option<f64> {
        self.tbt.degradation_ratio()
    }
}
