use crate::config::TriggerConfig;
#[cfg(feature = "hierarchical")]
use crate::types::{PressureVector, ReliefVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

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

#[cfg(feature = "hierarchical")]
impl From<PressureVector> for Pressure6D {
    fn from(pv: PressureVector) -> Self {
        Pressure6D {
            gpu: 0.0,
            cpu: pv.compute,
            memory: pv.memory,
            thermal: pv.thermal,
            latency: 0.0,
            main_app: 0.0,
        }
    }
}

#[cfg(feature = "hierarchical")]
impl From<ReliefVector> for Pressure6D {
    fn from(rv: ReliefVector) -> Self {
        Pressure6D {
            gpu: 0.0,
            cpu: rv.compute,
            memory: rv.memory,
            thermal: rv.thermal,
            latency: rv.latency,
            main_app: 0.0,
        }
    }
}

impl From<[f32; 6]> for Pressure6D {
    fn from(arr: [f32; 6]) -> Self {
        Pressure6D {
            gpu: arr[0],
            cpu: arr[1],
            memory: arr[2],
            thermal: arr[3],
            latency: arr[4],
            main_app: arr[5],
        }
    }
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

    pub fn update_tbt_from_throughput(&mut self, throughput: f32) {
        if throughput <= 0.0 {
            return;
        }
        let tbt_ms = 1000.0 / throughput as f64;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliefEntry {
    pub relief: [f32; RELIEF_DIMS],
    pub observation_count: u32,
}

pub struct EwmaReliefTable {
    pub entries: HashMap<String, ReliefEntry>,
    pub alpha: f32,
    pub defaults: HashMap<String, Vec<f32>>,
}

impl EwmaReliefTable {
    pub fn new(alpha: f32, defaults: HashMap<String, Vec<f32>>) -> Self {
        Self {
            entries: HashMap::new(),
            alpha,
            defaults,
        }
    }

    pub fn predict(&self, action: &str) -> [f32; RELIEF_DIMS] {
        if let Some(entry) = self.entries.get(action) {
            return entry.relief;
        }
        if let Some(default) = self.defaults.get(action) {
            let mut relief = [0.0f32; RELIEF_DIMS];
            for (i, v) in default.iter().enumerate().take(RELIEF_DIMS) {
                relief[i] = *v;
            }
            return relief;
        }
        [0.0; RELIEF_DIMS]
    }

    pub fn observe(&mut self, action: &str, observed: &[f32; RELIEF_DIMS]) {
        let default = self
            .defaults
            .get(action)
            .map(|v| {
                let mut r = [0.0f32; RELIEF_DIMS];
                for (i, &val) in v.iter().enumerate().take(RELIEF_DIMS) {
                    r[i] = val;
                }
                r
            })
            .unwrap_or([0.0f32; RELIEF_DIMS]);

        let entry = self
            .entries
            .entry(action.to_string())
            .or_insert_with(|| ReliefEntry {
                relief: default,
                observation_count: 0,
            });

        let a = self.alpha;
        for (i, &obs_val) in observed.iter().enumerate() {
            entry.relief[i] = a * entry.relief[i] + (1.0 - a) * obs_val;
        }
        entry.observation_count += 1;
    }

    pub fn observation_count(&self, action: &str) -> u32 {
        self.entries.get(action).map_or(0, |e| e.observation_count)
    }

    pub fn snapshot(&self) -> HashMap<String, [f32; RELIEF_DIMS]> {
        self.entries
            .iter()
            .map(|(k, v)| (k.clone(), v.relief))
            .collect()
    }

    pub fn initial_snapshot(&self) -> HashMap<String, [f32; RELIEF_DIMS]> {
        self.defaults
            .iter()
            .map(|(k, v)| {
                let mut arr = [0.0f32; RELIEF_DIMS];
                for (i, &val) in v.iter().enumerate().take(RELIEF_DIMS) {
                    arr[i] = val;
                }
                (k.clone(), arr)
            })
            .collect()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.entries).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    pub fn load(
        path: &Path,
        alpha: f32,
        defaults: HashMap<String, Vec<f32>>,
    ) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let entries: HashMap<String, ReliefEntry> =
            serde_json::from_str(&json).map_err(std::io::Error::other)?;
        Ok(Self {
            entries,
            alpha,
            defaults,
        })
    }
}

/// Relief 테이블 업데이트 이벤트 (관측성 훅).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReliefUpdateEvent {
    /// observe() 대상 action 이름.
    pub action: String,
    /// observe() 호출 전 relief 벡터.
    pub before: [f32; RELIEF_DIMS],
    /// observe() 호출 후 relief 벡터 (EWMA 적용 결과).
    pub after: [f32; RELIEF_DIMS],
    /// 이번에 관측된 델타 (before_pressure - after_pressure).
    pub observed: [f32; RELIEF_DIMS],
    /// 업데이트 후 total observation count.
    pub observation_count: u32,
    /// ObservationContext가 기록된 이후 경과 시간 (초).
    pub age_s: f64,
}
