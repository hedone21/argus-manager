//! Lua-scripted policy: the manager's decision layer.
//!
//! `LuaPolicy` normalizes the monitors' signals into per-axis pressure, applies
//! enter/exit hysteresis, and hands both to a `decide(ctx)` function in an operator-
//! supplied script. Whatever that returns becomes a directive.
//!
//! The decision logic is a script rather than Rust because the inputs are not portable:
//! which sensors exist, what counts as pressure on this SoC, and what QoS the co-running
//! app needs all change per platform and per scenario, while the manager binary should
//! not. Keeping the ruleset outside the binary is the second half of the narrow-contract
//! design — the first being that the contract itself names no KV technique.
//!
//! ## The script contract
//!
//! A script must define a global `decide(ctx)`. See [`crate::lua::context`] for `ctx`.
//! It may return `nil` (do nothing), one action table, or an array of them:
//!
//! ```lua
//! return { type = "kv.compress", budget = 0.5 }   -- retain 50% of uncompressed KV bytes
//! return { type = "restore_defaults" }            -- release what was applied
//! return { type = "suspend" } / { type = "resume" }
//! ```
//!
//! It may also define `POLICY_META = { name = ..., version = ... }`, which is logged.
//!
//! An earlier version of this file also ran a DPP argmax over ten named actions, an EWMA
//! relief table learned from observed effects, a LinUCB exploration bonus, and a QCF
//! round trip that asked the engine to score each candidate before choosing. All of it
//! ranked candidates. The contract now carries one command whose only payload is a
//! budget, so there is nothing to rank; the engine picks the technique, which is where
//! the knowledge to pick it lives.

use mlua::{Lua, Result as LuaResult, StdLib, Table, Value};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::clock::{Clock, SystemClock};
use crate::config::AdaptationConfig;
use crate::lua::context::build_ctx;
use crate::lua::fallback::fallback_decide;
use crate::monitor::compute::SharedGpuProvider;
use crate::pipeline::{PolicyStrategy, next_seq_id};
use crate::policy::common::state::{SignalState, TriggerEngine};
use crate::signal::SystemSignal;
use crate::types::OperatingMode;
use argus_shared::{EngineCommand, EngineDirective, EngineMessage, EngineStatus};

/// Consecutive whole-VM failures before the policy stops calling Lua for good.
const FALLBACK_ERROR_THRESHOLD: u32 = 3;

#[derive(Debug, Clone)]
pub struct PolicyMeta {
    pub name: String,
    pub version: String,
}

pub struct LuaPolicy {
    lua: Lua,
    script_path: std::path::PathBuf,
    policy_meta: Option<PolicyMeta>,

    signal_state: SignalState,
    trigger_engine: TriggerEngine,
    engine_state: Option<EngineStatus>,

    adaptation_config: AdaptationConfig,
    #[allow(dead_code)]
    clock: Arc<dyn Clock>,
    gpu_provider: SharedGpuProvider,

    consecutive_errors: u32,
    permanent_fallback: bool,
    /// Actions a script asked for that this manager cannot express. Surfaced through
    /// `inspect_state` because a script left behind by a contract change fails this way
    /// and no other: the VM runs, `decide` returns, and every entry is discarded.
    dropped_actions: u64,
}

impl std::fmt::Debug for LuaPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LuaPolicy")
            .field("script_path", &self.script_path)
            .field("policy_meta", &self.policy_meta)
            .field("permanent_fallback", &self.permanent_fallback)
            .finish()
    }
}

fn log_policy_meta(lua: &Lua) -> Option<PolicyMeta> {
    let meta: Table = lua.globals().get("POLICY_META").ok()?;
    let name: String = meta.get("name").ok()?;
    let version: String = meta.get("version").ok()?;
    log::info!("Lua policy: {} v{}", name, version);
    Some(PolicyMeta { name, version })
}

impl LuaPolicy {
    pub fn new(
        script_path: &str,
        config: AdaptationConfig,
        clock: Arc<dyn Clock>,
    ) -> anyhow::Result<Self> {
        Self::new_with_gpu(
            script_path,
            config,
            clock,
            crate::monitor::gpu_provider::shared_null(),
        )
    }

    /// Production path — shares the `ComputeMonitor`'s GPU provider so a tegrastats child
    /// is not spawned twice.
    pub fn new_with_gpu(
        script_path: &str,
        config: AdaptationConfig,
        clock: Arc<dyn Clock>,
        gpu_provider: SharedGpuProvider,
    ) -> anyhow::Result<Self> {
        // Sandbox: TABLE | STRING | MATH | IO. OS / PACKAGE / DEBUG are blocked
        // (MGR-049). `unsafe_new_with` because mlua classifies the subset as unsafe for
        // want of DEBUG. IO is deliberately allowed so a script can read platform sensors
        // the manager does not know about — which also means a policy script runs with
        // the manager's privileges. Do not point --policy-script at an untrusted file.
        let lua = unsafe {
            Lua::unsafe_new_with(
                StdLib::TABLE | StdLib::STRING | StdLib::MATH | StdLib::IO,
                mlua::LuaOptions::default(),
            )
        };
        let _ = lua.set_memory_limit(4 * 1024 * 1024);

        register_sys_helpers(&lua, Arc::clone(&gpu_provider))
            .map_err(|e| anyhow::anyhow!("Failed to register sys helpers: {}", e))?;

        load_script(&lua, script_path)?;
        let policy_meta = log_policy_meta(&lua);
        log::info!("LuaPolicy loaded from {}", script_path);

        Ok(Self {
            lua,
            script_path: script_path.into(),
            policy_meta,
            signal_state: SignalState::default(),
            trigger_engine: TriggerEngine::new(config.trigger.clone()),
            engine_state: None,
            adaptation_config: config,
            clock,
            gpu_provider,
            consecutive_errors: 0,
            permanent_fallback: false,
            dropped_actions: 0,
        })
    }

    pub fn with_system_clock(script_path: &str, config: AdaptationConfig) -> anyhow::Result<Self> {
        Self::new(script_path, config, Arc::new(SystemClock::new()))
    }

    pub fn with_system_clock_and_gpu(
        script_path: &str,
        config: AdaptationConfig,
        gpu_provider: SharedGpuProvider,
    ) -> anyhow::Result<Self> {
        Self::new_with_gpu(
            script_path,
            config,
            Arc::new(SystemClock::new()),
            gpu_provider,
        )
    }

    /// Read access to the VM — integration tests inspect globals through this.
    #[doc(hidden)]
    pub fn lua(&self) -> &Lua {
        &self.lua
    }

    /// Per-axis pressure from the latest readings.
    fn pressure(&self) -> crate::policy::common::state::Pressure6D {
        self.signal_state.pressure_with_thermal(
            self.adaptation_config.temp_safe_c,
            self.adaptation_config.temp_critical_c,
            self.trigger_engine.tbt_degradation_ratio(),
        )
    }

    /// Call `decide(ctx)`, falling back to [`fallback_decide`] on a VM-level failure.
    ///
    /// Three consecutive VM failures latch `permanent_fallback`. A script that merely
    /// returns actions this manager cannot express is NOT a VM failure — it is counted in
    /// `dropped_actions` and logged, because latching the fallback on it would hide a
    /// working script behind one bad branch.
    fn call_decide(&mut self, signal: &SystemSignal) -> Vec<EngineCommand> {
        if self.permanent_fallback {
            return fallback_decide(signal);
        }
        let pressure = self.pressure();
        let result = (|| -> LuaResult<Vec<EngineCommand>> {
            let ctx = build_ctx(
                &self.lua,
                &pressure,
                self.trigger_engine.state(),
                &self.signal_state,
                self.engine_state.as_ref(),
            )?;
            let decide: mlua::Function = self.lua.globals().get("decide")?;
            let value: Value = decide.call(ctx)?;
            Ok(parse_actions(value, &mut self.dropped_actions))
        })();

        match result {
            Ok(cmds) => {
                self.consecutive_errors = 0;
                cmds
            }
            Err(e) => {
                self.consecutive_errors += 1;
                log::error!(
                    "Lua decide() failed ({}/{}): {}",
                    self.consecutive_errors,
                    FALLBACK_ERROR_THRESHOLD,
                    e
                );
                if self.consecutive_errors >= FALLBACK_ERROR_THRESHOLD {
                    log::error!(
                        "Lua policy disabled after {} consecutive failures — using the \
                         built-in fallback for the rest of this session",
                        FALLBACK_ERROR_THRESHOLD
                    );
                    self.permanent_fallback = true;
                }
                fallback_decide(signal)
            }
        }
    }
}

fn load_script(lua: &Lua, script_path: &str) -> anyhow::Result<()> {
    let script = std::fs::read_to_string(script_path)
        .map_err(|e| anyhow::anyhow!("Failed to read Lua script {}: {}", script_path, e))?;
    lua.load(&script)
        .set_name(script_path)
        .exec()
        .map_err(|e| anyhow::anyhow!("Failed to evaluate Lua script {}: {}", script_path, e))?;
    let decide: Value = lua
        .globals()
        .get("decide")
        .map_err(|e| anyhow::anyhow!("Failed to get 'decide' global: {}", e))?;
    if !decide.is_function() {
        anyhow::bail!(
            "Lua script {} must define a global `decide(ctx)` function",
            script_path
        );
    }
    Ok(())
}

impl PolicyStrategy for LuaPolicy {
    fn process_signal(&mut self, signal: &SystemSignal) -> Option<EngineDirective> {
        match signal {
            SystemSignal::MemoryPressure {
                available_bytes,
                total_bytes,
                ..
            } => {
                self.signal_state
                    .update_memory(*available_bytes, *total_bytes);
                let p = if *total_bytes > 0 {
                    1.0 - (*available_bytes as f64 / *total_bytes as f64)
                } else {
                    0.0
                };
                self.trigger_engine.update_mem(p);
            }
            SystemSignal::ComputeGuidance {
                cpu_usage_pct,
                gpu_usage_pct,
                ..
            } => {
                self.signal_state
                    .update_compute(*cpu_usage_pct, *gpu_usage_pct);
            }
            SystemSignal::ThermalAlert {
                temperature_mc,
                throttling_active,
                ..
            } => {
                self.signal_state
                    .update_thermal(*temperature_mc, *throttling_active);
                let normalized = self.pressure().thermal as f64;
                self.trigger_engine.update_temp(normalized);
            }
            SystemSignal::EnergyConstraint { .. } => {}
        }

        let commands = self.call_decide(signal);
        if commands.is_empty() {
            return None;
        }
        Some(EngineDirective {
            seq_id: next_seq_id(),
            commands,
        })
    }

    fn update_engine_state(&mut self, msg: &EngineMessage) {
        match msg {
            EngineMessage::Heartbeat(status) => {
                self.trigger_engine.update_tbt_ms(status.tbt_ms as f64);
                self.engine_state = Some(status.clone());
            }
            EngineMessage::Response(resp) => {
                // `Rejected` is how the engine says what it cannot do — the contract has
                // no capability exchange — so it is worth a level a running operator sees.
                for r in &resp.results {
                    match r {
                        argus_shared::CommandResult::Rejected { reason } => log::warn!(
                            "engine rejected a command from directive {}: {}",
                            resp.seq_id,
                            reason
                        ),
                        argus_shared::CommandResult::Partial { achieved, reason } => log::info!(
                            "engine partially applied directive {} (achieved {:.3}): {}",
                            resp.seq_id,
                            achieved,
                            reason
                        ),
                        argus_shared::CommandResult::Ok => {}
                    }
                }
            }
        }
    }

    fn mode(&self) -> OperatingMode {
        let t = self.trigger_engine.state();
        // Memory is the axis with a hard floor under it — the LMK kills the process, so
        // crossing it is not the same kind of event as a warm SoC or a slow token.
        if t.mem_low {
            OperatingMode::Critical
        } else if t.temp_high || t.tbt_degraded {
            OperatingMode::Warning
        } else {
            OperatingMode::Normal
        }
    }

    fn inspect_state(&mut self, visitor: &mut dyn crate::pipeline::PolicyVisitor) {
        let p = self.pressure();
        visitor.record_f32("pressure_gpu", p.gpu);
        visitor.record_f32("pressure_cpu", p.cpu);
        visitor.record_f32("pressure_memory", p.memory);
        visitor.record_f32("pressure_thermal", p.thermal);
        visitor.record_f32("pressure_latency", p.latency);
        visitor.record_u64("dropped_actions", self.dropped_actions);
        if let Some(m) = &self.policy_meta {
            visitor.record_string("policy_name", &m.name);
            visitor.record_string("policy_version", &m.version);
        }
    }

    fn as_reloadable(&mut self) -> Option<&mut dyn crate::pipeline::ReloadablePolicy> {
        Some(self)
    }
}

impl crate::pipeline::ReloadablePolicy for LuaPolicy {
    /// Swap the script only if the new one loads cleanly — a fresh VM is built and
    /// discarded on failure, so a bad edit leaves the running policy untouched.
    fn reload_script(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF-8 script path"))?;
        let lua = unsafe {
            Lua::unsafe_new_with(
                StdLib::TABLE | StdLib::STRING | StdLib::MATH | StdLib::IO,
                mlua::LuaOptions::default(),
            )
        };
        let _ = lua.set_memory_limit(4 * 1024 * 1024);
        register_sys_helpers(&lua, Arc::clone(&self.gpu_provider))
            .map_err(|e| anyhow::anyhow!("Failed to register sys helpers: {}", e))?;
        load_script(&lua, path_str)?;

        self.policy_meta = log_policy_meta(&lua);
        self.lua = lua;
        self.script_path = path.to_path_buf();
        self.consecutive_errors = 0;
        self.permanent_fallback = false;
        log::info!("Lua policy reloaded from {}", path_str);
        Ok(())
    }

    fn script_path(&self) -> Option<&std::path::Path> {
        Some(&self.script_path)
    }
}

/// Turn `decide`'s return value into commands.
///
/// `nil` and an empty table both mean "no action". A single action table is accepted as
/// well as an array of them. An entry this manager cannot express is dropped with a log
/// line and counted — see `dropped_actions` on why that is not a VM failure.
fn parse_actions(value: Value, dropped: &mut u64) -> Vec<EngineCommand> {
    let table = match value {
        Value::Nil => return Vec::new(),
        Value::Table(t) => t,
        other => {
            log::error!("decide() must return nil or a table, got {:?}", other);
            *dropped += 1;
            return Vec::new();
        }
    };

    // A single action table has a `type`; an array of them does not.
    if table.contains_key("type").unwrap_or(false) {
        return match parse_single_action(&table) {
            Ok(cmd) => vec![cmd],
            Err(e) => {
                log::error!("decide() returned an unusable action: {}", e);
                *dropped += 1;
                Vec::new()
            }
        };
    }

    let mut commands = Vec::new();
    for entry in table.sequence_values::<Table>() {
        match entry.and_then(|t| parse_single_action(&t)) {
            Ok(cmd) => commands.push(cmd),
            Err(e) => {
                log::error!("decide() returned an unusable action: {}", e);
                *dropped += 1;
            }
        }
    }
    commands
}

fn parse_single_action(entry: &Table) -> LuaResult<EngineCommand> {
    let action_type: String = entry.get("type")?;
    match action_type.as_str() {
        "kv.compress" => {
            let budget: f32 = entry.get("budget")?;
            // Validated here rather than by the engine because a non-finite float cannot
            // reach the engine to be rejected: serde_json writes NaN and infinity as
            // `null`, `null` does not deserialize into `f32`, and the whole frame is
            // dropped without a response. The realistic producer is a script dividing by
            // `kv_cache_budget_bytes` before the first heartbeat.
            if !(budget.is_finite() && budget > 0.0 && budget <= 1.0) {
                return Err(mlua::Error::runtime(format!(
                    "kv.compress budget must be finite and in (0.0, 1.0], got {budget}"
                )));
            }
            Ok(EngineCommand::KvCompress { budget })
        }
        "restore_defaults" => Ok(EngineCommand::RestoreDefaults),
        "suspend" => Ok(EngineCommand::Suspend),
        "resume" => Ok(EngineCommand::Resume),
        unknown => Err(mlua::Error::runtime(format!(
            "unknown action type '{unknown}' — the contract carries kv.compress, \
             restore_defaults, suspend and resume"
        ))),
    }
}

fn register_sys_helpers(lua: &Lua, gpu_provider: SharedGpuProvider) -> LuaResult<()> {
    let sys = lua.create_table()?;

    // sys.read(path) -> string
    sys.set(
        "read",
        lua.create_function(|_, path: String| -> LuaResult<String> {
            Ok(std::fs::read_to_string(&path)
                .map(|s| s.trim().to_string())
                .unwrap_or_default())
        })?,
    )?;

    // sys.meminfo() -> {total, available, free} (KB)
    sys.set(
        "meminfo",
        lua.create_function(|lua_inner, ()| -> LuaResult<Table> {
            let tbl = lua_inner.create_table()?;
            let (total, available, free) = read_meminfo();
            tbl.set("total", total)?;
            tbl.set("available", available)?;
            tbl.set("free", free)?;
            Ok(tbl)
        })?,
    )?;

    // sys.thermal(zone) -> float (degrees C)
    sys.set(
        "thermal",
        lua.create_function(|_, zone: u32| -> LuaResult<f64> {
            let path = format!("/sys/class/thermal/thermal_zone{}/temp", zone);
            let temp = std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok())
                .map(|mc| mc / 1000.0)
                .unwrap_or(-1.0);
            Ok(temp)
        })?,
    )?;

    // sys.gpu_busy() -> int (0-100, -1 if unavailable)
    let provider_busy = Arc::clone(&gpu_provider);
    sys.set(
        "gpu_busy",
        lua.create_function(move |_, ()| -> LuaResult<i64> {
            let pct = provider_busy
                .lock()
                .ok()
                .and_then(|mut g| g.util_pct())
                .map(|v| v.round() as i64)
                .unwrap_or(-1);
            Ok(pct)
        })?,
    )?;

    // sys.gpu_freq() -> int (Hz, -1 if unavailable)
    let provider_freq = Arc::clone(&gpu_provider);
    sys.set(
        "gpu_freq",
        lua.create_function(move |_, ()| -> LuaResult<i64> {
            let freq = provider_freq
                .lock()
                .ok()
                .and_then(|mut g| g.freq_hz())
                .map(|v| v as i64)
                .unwrap_or(-1);
            Ok(freq)
        })?,
    )?;

    // sys.cpu_freq(cpu_index) -> int (KHz)
    sys.set(
        "cpu_freq",
        lua.create_function(|_, cpu: u32| -> LuaResult<i64> {
            let path = format!(
                "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq",
                cpu
            );
            let freq = std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| s.trim().parse::<i64>().ok())
                .unwrap_or(-1);
            Ok(freq)
        })?,
    )?;

    // sys.foreground_fps(pkg) -> float|nil
    //
    // Measures the foreground frame rate of `pkg` by reading SurfaceFlinger
    // frame counters via `dumpsys SurfaceFlinger`. First call returns nil
    // (only stores baseline). Subsequent calls return FPS since last call.
    // Returns nil if the package surface is not found, or if called too soon
    // (< 100 ms since last call).
    //
    // NOTE: `dumpsys` is an Android command and is not available on the host.
    // The function will return nil gracefully when the command is unavailable.
    {
        struct FpsState {
            prev_frame: u64,
            prev_time: Instant,
            initialized: bool,
        }

        let fps_state = Arc::new(Mutex::new(FpsState {
            prev_frame: 0,
            prev_time: Instant::now(),
            initialized: false,
        }));

        sys.set(
            "foreground_fps",
            lua.create_function(move |_, pkg: String| -> LuaResult<Option<f32>> {
                let output = match std::process::Command::new("dumpsys")
                    .args(["SurfaceFlinger"])
                    .output()
                {
                    Ok(o) => o,
                    Err(_) => return Ok(None), // dumpsys not available (host dev)
                };

                let text = String::from_utf8_lossy(&output.stdout);

                // Find "SurfaceView[{pkg}" marker that has a "frame=" in its
                // next few lines.  SurfaceFlinger may list multiple layers for the
                // same package (e.g. Background layer without frame counter and
                // BLAST layer with frame counter).  We iterate all occurrences.
                let marker = format!("SurfaceView[{}", pkg);
                let mut frame_count: Option<u64> = None;
                let mut search_from = 0usize;
                while let Some(rel) = text[search_from..].find(&marker) {
                    let pos = search_from + rel;
                    let after = &text[pos..];
                    frame_count = after.lines().take(5).find_map(|line| {
                        let idx = line.find("frame=")?;
                        let rest = &line[idx + 6..];
                        let num_str: String =
                            rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                        num_str.parse().ok()
                    });
                    if frame_count.is_some() {
                        break; // Found a layer with frame counter
                    }
                    search_from = pos + marker.len();
                }

                let cur_frame = match frame_count {
                    Some(f) => f,
                    None => return Ok(None),
                };

                let mut state = fps_state.lock().unwrap();
                if !state.initialized {
                    state.prev_frame = cur_frame;
                    state.prev_time = Instant::now();
                    state.initialized = true;
                    return Ok(None); // First call: no delta yet
                }

                let now = Instant::now();
                let dt = now.duration_since(state.prev_time).as_secs_f32();
                if dt < 0.1 {
                    return Ok(None); // Too soon, skip
                }

                let delta_frames = cur_frame.saturating_sub(state.prev_frame);
                let fps = delta_frames as f32 / dt;

                state.prev_frame = cur_frame;
                state.prev_time = now;

                Ok(Some(fps))
            })?,
        )?;
    }

    lua.globals().set("sys", sys)?;
    Ok(())
}

/// Parse `/proc/meminfo` and return (total_kb, available_kb, free_kb).
fn read_meminfo() -> (u64, u64, u64) {
    let content = match std::fs::read_to_string("/proc/meminfo") {
        Ok(c) => c,
        Err(_) => return (0, 0, 0),
    };

    let mut total: u64 = 0;
    let mut available: u64 = 0;
    let mut free: u64 = 0;

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = parse_meminfo_value(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = parse_meminfo_value(rest);
        } else if let Some(rest) = line.strip_prefix("MemFree:") {
            free = parse_meminfo_value(rest);
        }
    }

    (total, available, free)
}

/// Parse a meminfo line value like "  12345 kB" into u64.
fn parse_meminfo_value(s: &str) -> u64 {
    s.split_whitespace()
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}
