use super::Monitor;
use crate::config::MemoryMonitorConfig;
use crate::evaluator::{Direction, ThresholdEvaluator, Thresholds};
use crate::signal::{Level, SystemSignal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// Available memory percentage → [`Level`] (descending: lower is worse).
///
/// Pure, stateless threshold check using the default `MemoryMonitorConfig` thresholds
/// (warning=40%, critical=20%, emergency=10%). Hysteresis is not applied; for stateful
/// evaluation with hysteresis the production monitor uses [`ThresholdEvaluator`] internally.
///
/// Exposed so the simulator can delegate level decisions here instead of
/// duplicating the threshold constants.
pub fn memory_level_from_available_pct(available_pct: f64) -> Level {
    let cfg = MemoryMonitorConfig::default();
    if available_pct <= cfg.emergency_pct {
        Level::Emergency
    } else if available_pct <= cfg.critical_pct {
        Level::Critical
    } else if available_pct <= cfg.warning_pct {
        Level::Warning
    } else {
        Level::Normal
    }
}

pub struct MemoryMonitor {
    poll_interval: Duration,
    evaluator: ThresholdEvaluator,
    meminfo_path: String,
    #[allow(dead_code)] // PSI path reserved for future use
    psi_path: String,
    last_total: u64,
    last_available: u64,
}

impl MemoryMonitor {
    pub fn new(config: &MemoryMonitorConfig, default_poll_ms: u64) -> Self {
        Self {
            poll_interval: Duration::from_millis(
                config.poll_interval_ms.unwrap_or(default_poll_ms),
            ),
            evaluator: ThresholdEvaluator::new(
                Direction::Descending,
                Thresholds {
                    warning: config.warning_pct,
                    critical: config.critical_pct,
                    emergency: config.emergency_pct,
                    hysteresis: config.hysteresis_pct,
                },
            ),
            meminfo_path: "/proc/meminfo".into(),
            psi_path: "/proc/pressure/memory".into(),
            last_total: 0,
            last_available: 0,
        }
    }

    #[cfg(test)]
    fn with_paths(config: &MemoryMonitorConfig, meminfo_path: String, psi_path: String) -> Self {
        let mut m = Self::new(config, 1000);
        m.meminfo_path = meminfo_path;
        m.psi_path = psi_path;
        m
    }

    fn build_signal(&self) -> SystemSignal {
        let reclaim = match self.evaluator.level() {
            Level::Normal => 0,
            Level::Warning => (self.last_total as f64 * 0.05) as u64,
            Level::Critical => (self.last_total as f64 * 0.10) as u64,
            Level::Emergency => (self.last_total as f64 * 0.20) as u64,
        };
        SystemSignal::MemoryPressure {
            level: self.evaluator.level(),
            available_bytes: self.last_available,
            total_bytes: self.last_total,
            reclaim_target_bytes: reclaim,
        }
    }
}

impl Monitor for MemoryMonitor {
    fn run(
        &mut self,
        tx: mpsc::Sender<SystemSignal>,
        shutdown: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        log::info!(
            "[MemoryMonitor] Starting (interval={}ms)",
            self.poll_interval.as_millis()
        );

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match parse_meminfo(&self.meminfo_path) {
                Ok((total, available)) => {
                    self.last_total = total;
                    self.last_available = available;

                    let pct = if total == 0 {
                        100.0
                    } else {
                        (available as f64 / total as f64) * 100.0
                    };

                    self.evaluator.evaluate(pct);
                    if tx.send(self.build_signal()).is_err() {
                        break;
                    }
                }
                Err(e) => log::warn!("[MemoryMonitor] Read failed: {}", e),
            }

            std::thread::sleep(self.poll_interval);
        }

        log::info!("[MemoryMonitor] Stopped");
        Ok(())
    }

    fn initial_signal(&self) -> Option<SystemSignal> {
        Some(self.build_signal())
    }

    fn name(&self) -> &str {
        "MemoryMonitor"
    }
}

fn parse_meminfo(path: &str) -> anyhow::Result<(u64, u64)> {
    let content = std::fs::read_to_string(path)?;
    let mut total_kb = None;
    let mut available_kb = None;

    for line in content.lines() {
        if let Some(val) = line.strip_prefix("MemTotal:") {
            total_kb = Some(parse_kb_value(val)?);
        } else if let Some(val) = line.strip_prefix("MemAvailable:") {
            available_kb = Some(parse_kb_value(val)?);
        }
        if total_kb.is_some() && available_kb.is_some() {
            break;
        }
    }

    let total = total_kb.ok_or_else(|| anyhow::anyhow!("MemTotal not found"))?;
    let available = available_kb.ok_or_else(|| anyhow::anyhow!("MemAvailable not found"))?;

    Ok((total * 1024, available * 1024))
}

fn parse_kb_value(s: &str) -> anyhow::Result<u64> {
    let s = s.trim().trim_end_matches("kB").trim();
    Ok(s.parse()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn default_config() -> MemoryMonitorConfig {
        MemoryMonitorConfig::default()
    }

    #[test]
    fn parse_meminfo_valid() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(
            f,
            "MemTotal:       65280212 kB\nMemFree:        40000000 kB\nMemAvailable:   55553552 kB\n"
        )
        .unwrap();

        let (total, avail) = parse_meminfo(f.path().to_str().unwrap()).unwrap();
        assert_eq!(total, 65280212 * 1024);
        assert_eq!(avail, 55553552 * 1024);
    }

    #[test]
    fn monitor_builds_signal() {
        let mut f_mem = tempfile::NamedTempFile::new().unwrap();
        write!(
            f_mem,
            "MemTotal:       1000000 kB\nMemAvailable:   500000 kB\n"
        )
        .unwrap();

        let mut f_psi = tempfile::NamedTempFile::new().unwrap();
        writeln!(f_psi, "some avg10=0.00 avg60=0.00 avg300=0.00 total=0").unwrap();

        let config = default_config();
        let mut monitor = MemoryMonitor::with_paths(
            &config,
            f_mem.path().to_str().unwrap().to_string(),
            f_psi.path().to_str().unwrap().to_string(),
        );

        // Read meminfo to populate last_total/last_available
        let (total, available) = parse_meminfo(f_mem.path().to_str().unwrap()).unwrap();
        monitor.last_total = total;
        monitor.last_available = available;

        let sig = monitor.build_signal();
        assert_eq!(sig.level(), Level::Normal); // 50% available, threshold is 40%
    }

    #[test]
    fn monitor_escalation() {
        let config = default_config(); // warning=40%, critical=20%, emergency=10%
        let mut monitor = MemoryMonitor::with_paths(
            &config,
            String::new(), // not used in direct evaluate
            String::new(),
        );

        monitor.last_total = 1_000_000_000;

        // 35% available → Warning
        monitor.last_available = 350_000_000;
        let level = monitor.evaluator.evaluate(35.0);
        assert_eq!(level, Some(Level::Warning));

        let sig = monitor.build_signal();
        assert_eq!(sig.level(), Level::Warning);
        if let SystemSignal::MemoryPressure {
            reclaim_target_bytes,
            ..
        } = sig
        {
            assert_eq!(reclaim_target_bytes, 50_000_000); // 5% of 1GB
        }
    }

    #[test]
    fn monitor_reclaim_scales_with_level() {
        let config = default_config();
        let mut monitor = MemoryMonitor::with_paths(&config, String::new(), String::new());
        monitor.last_total = 1_000_000_000;

        // Critical
        monitor.evaluator.evaluate(15.0); // → Critical
        monitor.last_available = 150_000_000;
        if let SystemSignal::MemoryPressure {
            reclaim_target_bytes,
            ..
        } = monitor.build_signal()
        {
            assert_eq!(reclaim_target_bytes, 100_000_000); // 10% of 1GB
        }
    }
}
