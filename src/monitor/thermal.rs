use super::Monitor;
use crate::config::ThermalMonitorConfig;
use crate::evaluator::{Direction, ThresholdEvaluator, Thresholds};
use argus_shared::{Level, SystemSignal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// Temperature in degrees Celsius → [`Level`] (ascending: higher is worse).
///
/// Pure, stateless threshold check using the default `ThermalMonitorConfig` thresholds
/// (warning=60°C, critical=75°C, emergency=85°C). Hysteresis is not applied; for stateful
/// evaluation with hysteresis the production monitor uses [`ThresholdEvaluator`] internally.
///
/// Exposed so the simulator can delegate level decisions here instead of
/// duplicating the threshold constants.
pub fn thermal_level_from_temp_c(temp_c: f64) -> Level {
    let cfg = ThermalMonitorConfig::default();
    // Config stores millidegrees; convert to °C for comparison.
    let warning_c = cfg.warning_mc as f64 / 1000.0;
    let critical_c = cfg.critical_mc as f64 / 1000.0;
    let emergency_c = cfg.emergency_mc as f64 / 1000.0;

    if temp_c >= emergency_c {
        Level::Emergency
    } else if temp_c >= critical_c {
        Level::Critical
    } else if temp_c >= warning_c {
        Level::Warning
    } else {
        Level::Normal
    }
}

/// [`Level`] → throttle ratio applied by the thermal monitor in `build_signal()`.
///
/// Exposed so the simulator can produce consistent throttle_ratio values.
pub fn throttle_ratio_from_level(level: Level) -> f64 {
    match level {
        Level::Normal | Level::Warning => 1.0,
        Level::Critical => 0.7,
        Level::Emergency => 0.3,
    }
}

pub struct ThermalMonitor {
    poll_interval: Duration,
    evaluator: ThresholdEvaluator,
    thermal_base: String,
    zone_indices: Vec<u32>,
    zone_type_filter: Vec<String>,
    zones_discovered: bool,
    last_temp_mc: i32,
    last_throttling: bool,
}

impl ThermalMonitor {
    pub fn new(config: &ThermalMonitorConfig, default_poll_ms: u64) -> Self {
        Self {
            poll_interval: Duration::from_millis(
                config.poll_interval_ms.unwrap_or(default_poll_ms),
            ),
            evaluator: ThresholdEvaluator::new(
                Direction::Ascending,
                Thresholds {
                    warning: config.warning_mc as f64,
                    critical: config.critical_mc as f64,
                    emergency: config.emergency_mc as f64,
                    hysteresis: config.hysteresis_mc as f64,
                },
            ),
            thermal_base: "/sys/class/thermal".into(),
            zone_indices: Vec::new(),
            zone_type_filter: config.zone_types.clone(),
            zones_discovered: false,
            last_temp_mc: 25000,
            last_throttling: false,
        }
    }

    #[cfg(test)]
    fn with_base(config: &ThermalMonitorConfig, base: String) -> Self {
        let mut m = Self::new(config, 1000);
        m.thermal_base = base;
        m
    }

    fn discover_zones(&mut self) {
        let base_path = std::path::Path::new(&self.thermal_base);
        let mut all_zones = Vec::new();

        for i in 0..32 {
            let type_path = base_path.join(format!("thermal_zone{}/type", i));
            let temp_path = base_path.join(format!("thermal_zone{}/temp", i));

            if !temp_path.exists() {
                continue;
            }

            let zone_type = std::fs::read_to_string(&type_path)
                .map(|s| s.trim().to_string())
                .unwrap_or_default();

            all_zones.push((i as u32, zone_type));
        }

        if self.zone_type_filter.is_empty() {
            self.zone_indices = all_zones.iter().map(|(idx, _)| *idx).collect();
            log::info!(
                "[ThermalMonitor] Monitoring all {} zones",
                self.zone_indices.len()
            );
        } else {
            self.zone_indices = all_zones
                .iter()
                .filter(|(_, zt)| self.zone_type_filter.iter().any(|f| zt == f))
                .map(|(idx, _)| *idx)
                .collect();

            if self.zone_indices.is_empty() {
                log::warn!(
                    "[ThermalMonitor] No zones matched filter {:?}, using all",
                    self.zone_type_filter
                );
                self.zone_indices = all_zones.iter().map(|(idx, _)| *idx).collect();
            } else {
                log::info!(
                    "[ThermalMonitor] Matched {} zones for filter {:?}",
                    self.zone_indices.len(),
                    self.zone_type_filter
                );
            }
        }

        self.zones_discovered = true;
    }

    fn read_once(&mut self) -> anyhow::Result<()> {
        if !self.zones_discovered {
            self.discover_zones();
        }

        let (max_temp, throttling) = read_thermal_state(&self.thermal_base, &self.zone_indices)?;
        self.last_temp_mc = max_temp;
        self.last_throttling = throttling;
        Ok(())
    }

    fn build_signal(&self) -> SystemSignal {
        let throttle_ratio = throttle_ratio_from_level(self.evaluator.level());
        SystemSignal::ThermalAlert {
            level: self.evaluator.level(),
            temperature_mc: self.last_temp_mc,
            throttling_active: self.last_throttling,
            throttle_ratio,
        }
    }
}

impl Monitor for ThermalMonitor {
    fn run(
        &mut self,
        tx: mpsc::Sender<SystemSignal>,
        shutdown: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        log::info!(
            "[ThermalMonitor] Starting (interval={}ms)",
            self.poll_interval.as_millis()
        );

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match self.read_once() {
                Ok(()) => {
                    self.evaluator.evaluate(self.last_temp_mc as f64);
                    if tx.send(self.build_signal()).is_err() {
                        break;
                    }
                }
                Err(e) => log::warn!("[ThermalMonitor] Read failed: {}", e),
            }

            std::thread::sleep(self.poll_interval);
        }

        log::info!("[ThermalMonitor] Stopped");
        Ok(())
    }

    fn initial_signal(&self) -> Option<SystemSignal> {
        Some(self.build_signal())
    }

    fn name(&self) -> &str {
        "ThermalMonitor"
    }
}

fn read_thermal_state(base: &str, zone_indices: &[u32]) -> anyhow::Result<(i32, bool)> {
    let base_path = std::path::Path::new(base);

    let mut max_temp: Option<i32> = None;
    for &i in zone_indices {
        let temp_path = base_path.join(format!("thermal_zone{}/temp", i));
        match std::fs::read_to_string(&temp_path) {
            Ok(content) => {
                if let Ok(temp) = content.trim().parse::<i32>() {
                    max_temp = Some(max_temp.map_or(temp, |cur: i32| cur.max(temp)));
                }
            }
            Err(_) => continue,
        }
    }

    let max_temp = max_temp.ok_or_else(|| anyhow::anyhow!("No thermal zones found"))?;

    let mut throttling = false;
    for i in 0..64 {
        let cur_path = base_path.join(format!("cooling_device{}/cur_state", i));
        match std::fs::read_to_string(&cur_path) {
            Ok(content) => {
                if let Ok(state) = content.trim().parse::<u32>()
                    && state > 0
                {
                    throttling = true;
                    break;
                }
            }
            Err(_) => break,
        }
    }

    Ok((max_temp, throttling))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ThermalMonitorConfig {
        ThermalMonitorConfig::default()
    }

    fn setup_thermal_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();

        let zone0 = dir.path().join("thermal_zone0");
        std::fs::create_dir(&zone0).unwrap();
        std::fs::write(zone0.join("temp"), "45000\n").unwrap();
        std::fs::write(zone0.join("type"), "acpitz\n").unwrap();

        let zone1 = dir.path().join("thermal_zone1");
        std::fs::create_dir(&zone1).unwrap();
        std::fs::write(zone1.join("temp"), "62000\n").unwrap();
        std::fs::write(zone1.join("type"), "x86_pkg_temp\n").unwrap();

        let cool0 = dir.path().join("cooling_device0");
        std::fs::create_dir(&cool0).unwrap();
        std::fs::write(cool0.join("cur_state"), "0\n").unwrap();

        dir
    }

    #[test]
    fn reads_hottest_zone() {
        let dir = setup_thermal_dir();
        let (temp, throttling) = read_thermal_state(dir.path().to_str().unwrap(), &[0, 1]).unwrap();
        assert_eq!(temp, 62000);
        assert!(!throttling);
    }

    #[test]
    fn detects_throttling() {
        let dir = setup_thermal_dir();
        std::fs::write(dir.path().join("cooling_device0/cur_state"), "5\n").unwrap();

        let (_, throttling) = read_thermal_state(dir.path().to_str().unwrap(), &[0, 1]).unwrap();
        assert!(throttling);
    }

    #[test]
    fn monitor_zone_discovery() {
        let dir = setup_thermal_dir();
        let config = default_config();
        let mut monitor =
            ThermalMonitor::with_base(&config, dir.path().to_str().unwrap().to_string());

        monitor.read_once().unwrap();
        assert_eq!(monitor.last_temp_mc, 62000);
        assert!(!monitor.last_throttling);
    }

    #[test]
    fn monitor_zone_filter() {
        let dir = setup_thermal_dir();

        // Add WiFi zone at 70°C (should be excluded)
        let zone2 = dir.path().join("thermal_zone2");
        std::fs::create_dir(&zone2).unwrap();
        std::fs::write(zone2.join("temp"), "70000\n").unwrap();
        std::fs::write(zone2.join("type"), "iwlwifi_1\n").unwrap();

        let mut config = default_config();
        config.zone_types = vec!["x86_pkg_temp".to_string()];
        let mut monitor =
            ThermalMonitor::with_base(&config, dir.path().to_str().unwrap().to_string());

        monitor.read_once().unwrap();
        assert_eq!(monitor.last_temp_mc, 62000); // Only zone1, not zone2
    }

    #[test]
    fn monitor_throttle_ratio() {
        let config = default_config();
        let mut monitor = ThermalMonitor::with_base(&config, String::new());

        monitor.last_temp_mc = 76000;
        monitor.evaluator.evaluate(76000.0); // → Critical

        if let SystemSignal::ThermalAlert { throttle_ratio, .. } = monitor.build_signal() {
            assert!((throttle_ratio - 0.7).abs() < 0.01);
        } else {
            panic!("Expected ThermalAlert");
        }
    }

    #[test]
    fn monitor_fallback_on_no_match() {
        let dir = setup_thermal_dir();

        let mut config = default_config();
        config.zone_types = vec!["nonexistent_sensor".to_string()];
        let mut monitor =
            ThermalMonitor::with_base(&config, dir.path().to_str().unwrap().to_string());

        monitor.read_once().unwrap();
        assert_eq!(monitor.last_temp_mc, 62000); // Falls back to all zones
    }
}
