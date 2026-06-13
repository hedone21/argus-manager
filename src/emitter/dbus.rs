use crate::emitter::Emitter;
use argus_shared::SystemSignal;

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

fn level_to_str(level: argus_shared::Level) -> &'static str {
    match level {
        argus_shared::Level::Normal => "normal",
        argus_shared::Level::Warning => "warning",
        argus_shared::Level::Critical => "critical",
        argus_shared::Level::Emergency => "emergency",
    }
}

fn backend_to_str(b: argus_shared::RecommendedBackend) -> &'static str {
    match b {
        argus_shared::RecommendedBackend::Cpu => "cpu",
        argus_shared::RecommendedBackend::Gpu => "gpu",
        argus_shared::RecommendedBackend::Any => "any",
    }
}

fn compute_reason_to_str(r: argus_shared::ComputeReason) -> &'static str {
    match r {
        argus_shared::ComputeReason::CpuBottleneck => "cpu_bottleneck",
        argus_shared::ComputeReason::GpuBottleneck => "gpu_bottleneck",
        argus_shared::ComputeReason::CpuAvailable => "cpu_available",
        argus_shared::ComputeReason::GpuAvailable => "gpu_available",
        argus_shared::ComputeReason::BothLoaded => "both_loaded",
        argus_shared::ComputeReason::Balanced => "balanced",
    }
}

fn energy_reason_to_str(r: argus_shared::EnergyReason) -> &'static str {
    match r {
        argus_shared::EnergyReason::BatteryLow => "battery_low",
        argus_shared::EnergyReason::BatteryCritical => "battery_critical",
        argus_shared::EnergyReason::PowerLimit => "power_limit",
        argus_shared::EnergyReason::ThermalPower => "thermal_power",
        argus_shared::EnergyReason::Charging => "charging",
        argus_shared::EnergyReason::None => "none",
    }
}
