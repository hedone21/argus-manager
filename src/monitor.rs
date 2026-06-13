pub mod compute;
pub mod energy;
pub mod external;
pub mod gpu_provider;
pub mod memory;
pub mod thermal;

// Re-export the pure level-decision helpers so the integration test simulator
// can import them via `argus_manager::monitor::*` without duplicating thresholds.
pub use compute::{compute_level_from_pcts, compute_recommendation};
pub use memory::memory_level_from_available_pct;
pub use thermal::{thermal_level_from_temp_c, throttle_ratio_from_level};

use argus_shared::SystemSignal;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;

/// A self-contained monitoring unit that collects data, evaluates thresholds,
/// and produces SystemSignals.
///
/// Each Monitor owns its domain completely: data source, evaluation logic,
/// and signal construction. The Manager simply acts as a signal bus.
///
/// # Adding a new monitor
///
/// 1. Create `monitor/your_monitor.rs`
/// 2. Implement `Monitor` trait
/// 3. Add config section to `config.rs`
/// 4. Register in `main.rs::build_monitors()`
pub trait Monitor: Send + 'static {
    /// Run the monitor in a dedicated thread.
    ///
    /// Should send `SystemSignal`s to `tx` when thresholds are crossed
    /// and return `Ok(())` when `shutdown` becomes true.
    fn run(
        &mut self,
        tx: mpsc::Sender<SystemSignal>,
        shutdown: Arc<AtomicBool>,
    ) -> anyhow::Result<()>;

    /// Initial state signal for new clients. Returns `None` if not applicable
    /// (e.g., ExternalMonitor has no initial state).
    fn initial_signal(&self) -> Option<SystemSignal>;

    fn name(&self) -> &str;
}
