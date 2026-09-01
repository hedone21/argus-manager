use crate::emitter::Emitter;
use crate::signal::SystemSignal;

const MANAGER_PATH: &str = "/org/llm/Manager1";
const MANAGER_IFACE: &str = "org.llm.Manager1";

/// D-Bus signal emitter for Linux System Bus.
///
/// Emits `org.llm.Manager1` signals matching the IPC specification.
pub struct DbusEmitter {
    conn: zbus::blocking::Connection,
}

impl DbusEmitter {
    /// Connect to the System Bus and register the well-known name.
    pub fn new() -> anyhow::Result<Self> {
        let conn = zbus::blocking::Connection::system()?;
        conn.request_name("org.llm.Manager1")?;
        log::info!("[DbusEmitter] Registered org.llm.Manager1 on System Bus");
        Ok(Self { conn })
    }

    fn emit_signal(&mut self, signal: &SystemSignal) -> anyhow::Result<()> {
        match signal {
            SystemSignal::MemoryPressure {
                level,
                available_bytes,
                total_bytes,
                reclaim_target_bytes,
            } => {
                let level_str = level_to_str(*level);
                self.conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "MemoryPressure",
                    &(
                        level_str,
                        *available_bytes,
                        *total_bytes,
                        *reclaim_target_bytes,
                    ),
                )?;
            }
            SystemSignal::ComputeGuidance {
                level,
                recommended_backend,
                reason,
                cpu_usage_pct,
                gpu_usage_pct,
            } => {
                let level_str = level_to_str(*level);
                let backend_str = backend_to_str(*recommended_backend);
                let reason_str = compute_reason_to_str(*reason);
                self.conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "ComputeGuidance",
                    &(
                        level_str,
                        backend_str,
                        reason_str,
                        *cpu_usage_pct,
                        *gpu_usage_pct,
                    ),
                )?;
            }
            SystemSignal::ThermalAlert {
                level,
                temperature_mc,
                throttling_active,
                throttle_ratio,
            } => {
                let level_str = level_to_str(*level);
                self.conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "ThermalAlert",
                    &(
                        level_str,
                        *temperature_mc,
                        *throttling_active,
                        *throttle_ratio,
                    ),
                )?;
            }
            SystemSignal::EnergyConstraint {
                level,
                reason,
                power_budget_mw,
            } => {
                let level_str = level_to_str(*level);
                let reason_str = energy_reason_to_str(*reason);
                self.conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "EnergyConstraint",
                    &(level_str, reason_str, *power_budget_mw),
                )?;
            }
        }
        Ok(())
    }
}

impl Emitter for DbusEmitter {
    fn emit(&mut self, signal: &SystemSignal) -> anyhow::Result<()> {
        log::debug!("[DbusEmitter] Emitting {:?}", signal);
        self.emit_signal(signal)
    }

    fn emit_initial(&mut self, signals: &[SystemSignal]) -> anyhow::Result<()> {
        log::info!(
            "[DbusEmitter] Emitting {} initial state signals",
            signals.len()
        );
        for signal in signals {
            self.emit_signal(signal)?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "DbusEmitter"
    }
}

fn level_to_str(level: crate::signal::Level) -> &'static str {
    match level {
        crate::signal::Level::Normal => "normal",
        crate::signal::Level::Warning => "warning",
        crate::signal::Level::Critical => "critical",
        crate::signal::Level::Emergency => "emergency",
    }
}

fn backend_to_str(b: crate::signal::RecommendedBackend) -> &'static str {
    match b {
        crate::signal::RecommendedBackend::Cpu => "cpu",
        crate::signal::RecommendedBackend::Gpu => "gpu",
        crate::signal::RecommendedBackend::Any => "any",
    }
}

fn compute_reason_to_str(r: crate::signal::ComputeReason) -> &'static str {
    match r {
        crate::signal::ComputeReason::CpuBottleneck => "cpu_bottleneck",
        crate::signal::ComputeReason::GpuBottleneck => "gpu_bottleneck",
        crate::signal::ComputeReason::CpuAvailable => "cpu_available",
        crate::signal::ComputeReason::GpuAvailable => "gpu_available",
        crate::signal::ComputeReason::BothLoaded => "both_loaded",
        crate::signal::ComputeReason::Balanced => "balanced",
    }
}

fn energy_reason_to_str(r: crate::signal::EnergyReason) -> &'static str {
    match r {
        crate::signal::EnergyReason::BatteryLow => "battery_low",
        crate::signal::EnergyReason::BatteryCritical => "battery_critical",
        crate::signal::EnergyReason::PowerLimit => "power_limit",
        crate::signal::EnergyReason::ThermalPower => "thermal_power",
        crate::signal::EnergyReason::Charging => "charging",
        crate::signal::EnergyReason::None => "none",
    }
}
