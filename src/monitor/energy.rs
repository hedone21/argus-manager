use super::Monitor;
use crate::config::EnergyMonitorConfig;
use crate::evaluator::{Direction, ThresholdEvaluator, Thresholds};
use crate::signal::{EnergyReason, Level, SystemSignal};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

pub struct EnergyMonitor {
    poll_interval: Duration,
    evaluator: ThresholdEvaluator,
    ignore_when_charging: bool,
    power_budgets: PowerBudgets,
    battery_path: Option<PathBuf>,
    last_charging: bool,
}

struct PowerBudgets {
    warning_mw: u32,
    critical_mw: u32,
    emergency_mw: u32,
}

impl EnergyMonitor {
    pub fn new(config: &EnergyMonitorConfig, default_poll_ms: u64) -> Self {
        let battery_path = find_battery_path("/sys/class/power_supply");
        if let Some(ref path) = battery_path {
            log::info!("[EnergyMonitor] Found battery at {}", path.display());
        } else {
            log::info!("[EnergyMonitor] No battery found, will report defaults");
        }

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
                    hysteresis: 2.0, // Fixed 2% hysteresis for battery
                },
            ),
            ignore_when_charging: config.ignore_when_charging,
            power_budgets: PowerBudgets {
                warning_mw: config.warning_power_budget_mw,
                critical_mw: config.critical_power_budget_mw,
                emergency_mw: config.emergency_power_budget_mw,
            },
            battery_path,
            last_charging: false,
        }
    }

    #[cfg(test)]
    fn with_path(config: &EnergyMonitorConfig, battery_path: Option<PathBuf>) -> Self {
        let mut m = Self::new_without_discovery(config);
        m.battery_path = battery_path;
        m
    }

    #[cfg(test)]
    fn new_without_discovery(config: &EnergyMonitorConfig) -> Self {
        Self {
            poll_interval: Duration::from_millis(config.poll_interval_ms.unwrap_or(1000)),
            evaluator: ThresholdEvaluator::new(
                Direction::Descending,
                Thresholds {
                    warning: config.warning_pct,
                    critical: config.critical_pct,
                    emergency: config.emergency_pct,
                    hysteresis: 2.0,
                },
            ),
            ignore_when_charging: config.ignore_when_charging,
            power_budgets: PowerBudgets {
                warning_mw: config.warning_power_budget_mw,
                critical_mw: config.critical_power_budget_mw,
                emergency_mw: config.emergency_power_budget_mw,
            },
            battery_path: None,
            last_charging: false,
        }
    }

    fn evaluate_energy(&mut self, battery_pct: Option<f64>, charging: bool) -> Option<Level> {
        self.last_charging = charging;

        if charging && self.ignore_when_charging {
            // Force Normal when charging
            let current = self.evaluator.level();
            if current != Level::Normal {
                // Reset evaluator by evaluating a very high value
                self.evaluator.evaluate(100.0);
                return Some(Level::Normal);
            }
            return None;
        }

        if let Some(pct) = battery_pct {
            self.evaluator.evaluate(pct)
        } else {
            None
        }
    }

    fn build_signal(&self) -> SystemSignal {
        let (reason, budget) = match self.evaluator.level() {
            Level::Normal => {
                if self.last_charging {
                    (EnergyReason::Charging, 0)
                } else {
                    (EnergyReason::None, 0)
                }
            }
            Level::Warning => (EnergyReason::BatteryLow, self.power_budgets.warning_mw),
            Level::Critical => (
                EnergyReason::BatteryCritical,
                self.power_budgets.critical_mw,
            ),
            Level::Emergency => (
                EnergyReason::BatteryCritical,
                self.power_budgets.emergency_mw,
            ),
        };
        SystemSignal::EnergyConstraint {
            level: self.evaluator.level(),
            reason,
            power_budget_mw: budget,
        }
    }
}

impl Monitor for EnergyMonitor {
    fn run(
        &mut self,
        tx: mpsc::Sender<SystemSignal>,
        shutdown: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        log::info!(
            "[EnergyMonitor] Starting (interval={}ms, battery={})",
            self.poll_interval.as_millis(),
            self.battery_path
                .as_ref()
                .map_or("none".to_string(), |p| p.display().to_string())
        );

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            let (battery_pct, charging, _power_mw) = match &self.battery_path {
                Some(path) => read_battery_state(path),
                None => (None, false, None),
            };

            self.evaluate_energy(battery_pct, charging);
            if tx.send(self.build_signal()).is_err() {
                break;
            }

            std::thread::sleep(self.poll_interval);
        }

        log::info!("[EnergyMonitor] Stopped");
        Ok(())
    }

    fn initial_signal(&self) -> Option<SystemSignal> {
        Some(self.build_signal())
    }

    fn name(&self) -> &str {
        "EnergyMonitor"
    }
}

fn find_battery_path(base: &str) -> Option<PathBuf> {
    let base_path = Path::new(base);
    let entries = std::fs::read_dir(base_path).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        let type_path = path.join("type");
        if let Ok(device_type) = std::fs::read_to_string(&type_path)
            && device_type.trim().eq_ignore_ascii_case("battery")
            && path.join("capacity").exists()
        {
            return Some(path);
        }
    }
    None
}

fn read_battery_state(path: &Path) -> (Option<f64>, bool, Option<u32>) {
    let capacity = std::fs::read_to_string(path.join("capacity"))
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok());

    let charging = std::fs::read_to_string(path.join("status"))
        .ok()
        .map(|s| {
            let status = s.trim().to_lowercase();
            status == "charging" || status == "full"
        })
        .unwrap_or(false);

    let power_mw = std::fs::read_to_string(path.join("power_now"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|uw| (uw / 1000) as u32);

    (capacity, charging, power_mw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> EnergyMonitorConfig {
        EnergyMonitorConfig::default()
    }

    fn setup_battery_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();

        let bat = dir.path().join("BAT0");
        std::fs::create_dir(&bat).unwrap();
        std::fs::write(bat.join("type"), "Battery\n").unwrap();
        std::fs::write(bat.join("capacity"), "72\n").unwrap();
        std::fs::write(bat.join("status"), "Discharging\n").unwrap();
        std::fs::write(bat.join("power_now"), "15000000\n").unwrap();

        dir
    }

    #[test]
    fn find_battery() {
        let dir = setup_battery_dir();
        let path = find_battery_path(dir.path().to_str().unwrap());
        assert!(path.is_some());
        assert!(path.unwrap().ends_with("BAT0"));
    }

    #[test]
    fn read_battery_discharging() {
        let dir = setup_battery_dir();
        let bat_path = dir.path().join("BAT0");
        let (cap, charging, power) = read_battery_state(&bat_path);

        assert_eq!(cap, Some(72.0));
        assert!(!charging);
        assert_eq!(power, Some(15000));
    }

    #[test]
    fn read_battery_charging() {
        let dir = setup_battery_dir();
        let bat_path = dir.path().join("BAT0");
        std::fs::write(bat_path.join("status"), "Charging\n").unwrap();

        let (_, charging, _) = read_battery_state(&bat_path);
        assert!(charging);
    }

    #[test]
    fn monitor_no_battery() {
        let config = default_config();
        let mut monitor = EnergyMonitor::with_path(&config, None);

        let changed = monitor.evaluate_energy(None, false);
        assert!(changed.is_none());
        assert_eq!(monitor.evaluator.level(), Level::Normal);
    }

    #[test]
    fn monitor_battery_depletion() {
        let config = default_config(); // warning=30%, critical=15%, emergency=5%
        let mut monitor = EnergyMonitor::with_path(&config, None);

        assert!(monitor.evaluate_energy(Some(50.0), false).is_none());
        assert_eq!(
            monitor.evaluate_energy(Some(25.0), false),
            Some(Level::Warning)
        );
        assert_eq!(
            monitor.evaluate_energy(Some(10.0), false),
            Some(Level::Critical)
        );
        assert_eq!(
            monitor.evaluate_energy(Some(3.0), false),
            Some(Level::Emergency)
        );
    }

    #[test]
    fn monitor_charging_overrides() {
        let config = default_config();
        let mut monitor = EnergyMonitor::with_path(&config, None);

        // Go to Warning
        monitor.evaluate_energy(Some(25.0), false);
        assert_eq!(monitor.evaluator.level(), Level::Warning);

        // Start charging → Normal
        let level = monitor.evaluate_energy(Some(25.0), true);
        assert_eq!(level, Some(Level::Normal));

        let sig = monitor.build_signal();
        if let SystemSignal::EnergyConstraint { reason, .. } = sig {
            assert_eq!(reason, EnergyReason::Charging);
        } else {
            panic!("Expected EnergyConstraint");
        }
    }

    #[test]
    fn monitor_with_battery() {
        let dir = setup_battery_dir();
        let config = default_config();
        let monitor = EnergyMonitor::with_path(&config, Some(dir.path().join("BAT0")));

        let sig = monitor.build_signal();
        assert_eq!(sig.level(), Level::Normal); // 72% is well above warning
    }
}
