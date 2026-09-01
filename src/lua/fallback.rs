//! Fallback decider — used when the Lua VM itself is unusable.
//!
//! This is not a policy. It exists so that a broken script degrades to something
//! conservative rather than to nothing, and it deliberately has no thresholds of its own:
//! the levels it switches on were already computed by the monitors.

use crate::signal::{Level, SystemSignal};
use argus_shared::EngineCommand;

/// Compress the KV cache in proportion to memory pressure, and release when it clears.
///
/// Only the memory axis acts. The compute and thermal axes had commands of their own
/// (throttle, device switch) until the contract narrowed to a KV budget; with those gone
/// there is nothing honest for this fallback to do about them, and inventing a KV
/// compression in response to a hot SoC would be a policy decision — exactly what the
/// externalized script is for.
pub fn fallback_decide(signal: &SystemSignal) -> Vec<EngineCommand> {
    match signal {
        SystemSignal::MemoryPressure { level, .. } => match level {
            Level::Normal => vec![EngineCommand::RestoreDefaults],
            Level::Warning => vec![EngineCommand::KvCompress { budget: 0.85 }],
            Level::Critical => vec![EngineCommand::KvCompress { budget: 0.50 }],
            Level::Emergency => vec![EngineCommand::KvCompress { budget: 0.25 }],
        },
        SystemSignal::ThermalAlert { .. }
        | SystemSignal::ComputeGuidance { .. }
        | SystemSignal::EnergyConstraint { .. } => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(level: Level) -> SystemSignal {
        SystemSignal::MemoryPressure {
            level,
            available_bytes: 1,
            total_bytes: 100,
            reclaim_target_bytes: 0,
        }
    }

    #[test]
    fn memory_pressure_tightens_the_budget_monotonically() {
        let budget = |s| match fallback_decide(&s).first() {
            Some(EngineCommand::KvCompress { budget }) => Some(*budget),
            _ => None,
        };
        assert_eq!(budget(mem(Level::Normal)), None, "Normal releases instead");
        let w = budget(mem(Level::Warning)).unwrap();
        let c = budget(mem(Level::Critical)).unwrap();
        let e = budget(mem(Level::Emergency)).unwrap();
        assert!(w > c && c > e, "{w} > {c} > {e}");
        assert!(e > 0.0, "budget stays a usable fraction");
    }

    #[test]
    fn normal_memory_releases() {
        assert!(matches!(
            fallback_decide(&mem(Level::Normal))[..],
            [EngineCommand::RestoreDefaults]
        ));
    }

    /// The other axes have no KV-shaped answer, so the fallback says nothing rather than
    /// compressing the cache because the SoC is warm.
    #[test]
    fn non_memory_axes_emit_nothing() {
        let thermal = SystemSignal::ThermalAlert {
            level: Level::Emergency,
            temperature_mc: 90_000,
            throttling_active: true,
            throttle_ratio: 0.5,
        };
        assert!(fallback_decide(&thermal).is_empty());
    }
}
