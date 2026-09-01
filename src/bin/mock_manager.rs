//! Mock Manager binary for protocol validation and E2E testing.
//!
//! Supports three transport modes:
//!
//! 1. **Unix socket (default)**: Listens on a Unix socket, accepts an Engine
//!    connection, receives Capability + Heartbeats, and sends Directive commands.
//!    Supports single-command and scenario replay modes.
//!
//! 2. **TCP socket (`--tcp`)**: Same protocol over TCP. Useful on Android where
//!    Unix domain socket bind may fail with Permission denied.
//!
//! 3. **D-Bus (legacy, `--dbus`)**: Emits D-Bus signals on the System Bus.
//!    Requires the `dbus` cargo feature.
//!
//! # Usage
//!
//! ```bash
//! # Unix socket — single command
//! cargo run -p argus_manager --no-default-features --bin mock_manager -- \
//!     --command KvEvictSliding --keep-ratio 0.7
//!
//! # TCP socket — single command
//! cargo run -p argus_manager --no-default-features --bin mock_manager -- \
//!     --tcp 127.0.0.1:9999 --command KvEvictSliding --keep-ratio 0.7
//!
//! # Unix socket — scenario replay
//! cargo run -p argus_manager --no-default-features --bin mock_manager -- \
//!     --scenario scenario.json
//!
//! # D-Bus (legacy) — requires `dbus` feature
//! cargo run -p argus_manager --bin mock_manager -- \
//!     --dbus --signal MemoryPressure --level critical
//! ```

use std::io::{Read, Write};
use std::net::TcpListener;
#[cfg(unix)]
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, bail};
use clap::Parser;
use serde::Deserialize;

use argus_shared::{EngineCommand, EngineDirective, EngineMessage, ManagerMessage};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "mock_manager",
    about = "Mock Manager for Engine protocol validation and E2E testing"
)]
struct Args {
    // ── Socket transport ──
    /// Unix socket path to listen on (default, ignored when --tcp is set).
    #[arg(long, default_value = "/tmp/argus_manager.sock")]
    socket: String,

    /// TCP address to listen on (e.g. 127.0.0.1:9999).
    /// When set, TCP is used instead of Unix socket.
    #[arg(long)]
    tcp: Option<String>,

    /// Command to send (KvEvictSliding, KvEvictH2o, KvStreaming, KvMergeD2o,
    /// Throttle, SetTargetTbt, SwitchHw, KvQuantDynamic, LayerSkip,
    /// SetPartitionRatio, SetPrefillPolicy, SwapWeights, RecallWeights,
    /// Suspend, Resume, RestoreDefaults, RequestQcf).
    #[arg(long)]
    command: Option<String>,

    /// Scenario JSON file to replay (Unix socket mode).
    #[arg(long)]
    scenario: Option<PathBuf>,

    /// Seconds to wait for Heartbeats before sending Directive.
    #[arg(long, default_value_t = 2)]
    wait_secs: u64,

    // ── Command parameters ──
    /// keep_ratio for KvEvictSliding, KvEvictH2o, KvMergeD2o.
    #[arg(long)]
    keep_ratio: Option<f32>,

    /// sink_size for KvStreaming.
    #[arg(long)]
    sink_size: Option<usize>,

    /// window_size for KvStreaming.
    #[arg(long)]
    window_size: Option<usize>,

    /// delay_ms for Throttle.
    #[arg(long)]
    delay_ms: Option<u64>,

    /// device for SwitchHw.
    #[arg(long)]
    device: Option<String>,

    /// target_bits for KvQuantDynamic.
    #[arg(long)]
    target_bits: Option<u8>,

    /// skip_ratio for LayerSkip.
    #[arg(long)]
    skip_ratio: Option<f32>,

    /// target_ms for SetTargetTbt.
    #[arg(long)]
    target_ms: Option<u64>,

    /// ratio for SetPartitionRatio (0.0~1.0).
    #[arg(long)]
    ratio: Option<f32>,

    /// chunk_size for SetPrefillPolicy.
    #[arg(long)]
    chunk_size: Option<usize>,

    /// yield_ms for SetPrefillPolicy.
    #[arg(long)]
    yield_ms: Option<u32>,

    /// cpu_chunk_size for SetPrefillPolicy.
    #[arg(long)]
    cpu_chunk_size: Option<usize>,

    /// target_dtype for SwapWeights (currently only "q4_0" is executable;
    /// f16/f32/q8_0 are reserved wire-format variants).
    #[arg(long)]
    target_dtype: Option<String>,

    // ── D-Bus mode (legacy) ──
    /// Use D-Bus transport instead of Unix socket.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    dbus: bool,

    /// D-Bus signal to emit (MemoryPressure, ComputeGuidance, ThermalAlert, EnergyConstraint).
    #[cfg(feature = "dbus")]
    #[arg(long)]
    signal: Option<String>,

    /// Signal level for D-Bus mode.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    level: Option<String>,

    /// D-Bus: available_bytes for MemoryPressure.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    available_bytes: Option<u64>,

    /// D-Bus: reclaim_target for MemoryPressure.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    reclaim_target: Option<u64>,

    /// D-Bus: recommended_backend for ComputeGuidance.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    recommended_backend: Option<String>,

    /// D-Bus: reason for ComputeGuidance/EnergyConstraint.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    reason: Option<String>,

    /// D-Bus: cpu_usage for ComputeGuidance.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    cpu_usage: Option<f64>,

    /// D-Bus: gpu_usage for ComputeGuidance.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    gpu_usage: Option<f64>,

    /// D-Bus: temperature_mc for ThermalAlert.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    temperature_mc: Option<i32>,

    /// D-Bus: throttling_active for ThermalAlert.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    throttling_active: Option<bool>,

    /// D-Bus: throttle_ratio for ThermalAlert.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    throttle_ratio: Option<f64>,

    /// D-Bus: power_budget_mw for EnergyConstraint.
    #[cfg(feature = "dbus")]
    #[arg(long)]
    power_budget_mw: Option<u32>,
}

// ── Wire format helpers ─────────────────────────────────────────────────────

/// Serialise `msg` as length-prefixed JSON and write to `stream`.
fn send_message(stream: &mut (impl Read + Write), msg: &ManagerMessage) -> anyhow::Result<()> {
    let json = serde_json::to_vec(msg).context("serialise ManagerMessage")?;
    let len = (json.len() as u32).to_be_bytes();
    stream.write_all(&len).context("write length prefix")?;
    stream.write_all(&json).context("write JSON payload")?;
    stream.flush().context("flush stream")?;
    Ok(())
}

/// Try to read one `EngineMessage` from `stream`.
///
/// Returns `Ok(None)` on read timeout / would-block (non-blocking read).
fn recv_message(stream: &mut (impl Read + Write)) -> anyhow::Result<Option<EngineMessage>> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e)
            if e.kind() == std::io::ErrorKind::WouldBlock
                || e.kind() == std::io::ErrorKind::TimedOut =>
        {
            return Ok(None);
        }
        Err(e) => return Err(e).context("read length prefix"),
    }

    let payload_len = u32::from_be_bytes(len_buf) as usize;
    let mut json_buf = vec![0u8; payload_len];
    stream
        .read_exact(&mut json_buf)
        .context("read JSON payload")?;

    let msg: EngineMessage =
        serde_json::from_slice(&json_buf).context("deserialise EngineMessage")?;
    Ok(Some(msg))
}

// ── Scenario types ──────────────────────────────────────────────────────────

/// Command-based scenario file format for Unix socket mode.
#[derive(Debug, Deserialize)]
struct CommandScenario {
    name: String,
    #[serde(default)]
    description: Option<String>,
    commands: Vec<ScenarioCommand>,
}

/// A single command entry in a scenario file.
///
/// The per-technique parameter fields went with the commands that took them; a
/// compression's only payload is its budget, which `keep_ratio` carries.
#[derive(Debug, Deserialize)]
struct ScenarioCommand {
    delay_ms: u64,
    command: String,
    #[serde(default)]
    keep_ratio: Option<f32>,
}

// ── Command construction ────────────────────────────────────────────────────

/// Parameters for building an EngineCommand from CLI or scenario input.
struct CommandParams<'a> {
    name: &'a str,
    keep_ratio: Option<f32>,
}

fn build_command(params: &CommandParams<'_>) -> anyhow::Result<EngineCommand> {
    match params.name {
        // `--keep-ratio` keeps its name: it is the retained fraction, which is exactly
        // what a budget is. It is now a fraction of uncompressed KV *bytes*, not tokens.
        "KvCompress" => {
            let budget = params
                .keep_ratio
                .context("--keep-ratio required for KvCompress")?;
            Ok(EngineCommand::KvCompress { budget })
        }
        "RestoreDefaults" => Ok(EngineCommand::RestoreDefaults),
        "Suspend" => Ok(EngineCommand::Suspend),
        "Resume" => Ok(EngineCommand::Resume),
        other => bail!(
            "unknown command '{}' — the contract carries KvCompress, RestoreDefaults, \
             Suspend and Resume",
            other
        ),
    }
}

// ── Protocol invariant validation (TOOL-048) ────────────────────────────────

fn validate_response(
    seq_id: u64,
    response: &argus_shared::CommandResponse,
    num_commands: usize,
) -> bool {
    let mut valid = true;

    // INV-023: seq_id must match
    if response.seq_id != seq_id {
        eprintln!(
            "[PROTOCOL VIOLATION] INV-023: seq_id mismatch: sent {} but received {}",
            seq_id, response.seq_id
        );
        valid = false;
    }

    // INV-024: results.len() must equal commands.len()
    if response.results.len() != num_commands {
        eprintln!(
            "[PROTOCOL VIOLATION] INV-024: results count mismatch: sent {} commands but received {} results",
            num_commands,
            response.results.len()
        );
        valid = false;
    }

    valid
}

// ── Socket mode (Unix / TCP) ────────────────────────────────────────────────

/// Accept a connection via TCP.
fn accept_tcp(addr: &str) -> anyhow::Result<std::net::TcpStream> {
    let listener = TcpListener::bind(addr).with_context(|| format!("TCP bind to {}", addr))?;
    println!("[MockManager] Listening on TCP {}...", addr);
    let (stream, peer) = listener.accept().context("TCP accept")?;
    println!("[MockManager] Engine connected from {}", peer);
    stream
        .set_read_timeout(Some(Duration::from_millis(200)))
        .context("set_read_timeout")?;
    Ok(stream)
}

/// Accept a connection via Unix domain socket.
#[cfg(unix)]
fn accept_unix(path: &str) -> anyhow::Result<std::os::unix::net::UnixStream> {
    let _ = std::fs::remove_file(path);
    let listener = UnixListener::bind(path).with_context(|| format!("Unix bind to {}", path))?;
    println!("[MockManager] Listening on {}...", path);
    let (stream, _addr) = listener.accept().context("Unix accept")?;
    stream
        .set_read_timeout(Some(Duration::from_millis(200)))
        .context("set_read_timeout")?;
    Ok(stream)
}

fn run_socket_mode(args: &Args) -> anyhow::Result<()> {
    if let Some(ref addr) = args.tcp {
        let mut stream = accept_tcp(addr)?;
        run_protocol(args, &mut stream)
    } else {
        #[cfg(unix)]
        {
            let mut stream = accept_unix(&args.socket)?;
            run_protocol(args, &mut stream)
        }
        #[cfg(not(unix))]
        {
            bail!("Unix sockets not supported on this platform; use --tcp instead");
        }
    }
}

fn run_protocol(args: &Args, stream: &mut (impl Read + Write)) -> anyhow::Result<()> {
    // No capability handshake — the contract has none. The first thing an engine says
    // is a heartbeat.
    println!("[MockManager] Engine connected, waiting for heartbeats...");

    // Step 2: Wait for Heartbeats
    let wait_until = std::time::Instant::now() + Duration::from_secs(args.wait_secs);
    let mut heartbeat_count = 0u32;
    while std::time::Instant::now() < wait_until {
        match recv_message(stream)? {
            Some(EngineMessage::Heartbeat(status)) => {
                heartbeat_count += 1;
                println!(
                    "[MockManager] Heartbeat #{}: kv={}/{}B, tokens={}, tbt={:.1}ms, phase={:?}, state={:?}",
                    heartbeat_count,
                    status.kv_cache_bytes,
                    status.kv_cache_budget_bytes,
                    status.kv_cache_tokens,
                    status.tbt_ms,
                    status.phase,
                    status.state,
                );
            }
            Some(other) => {
                println!(
                    "[MockManager] Unexpected message during wait: {:?}",
                    std::mem::discriminant(&other)
                );
            }
            None => {
                // Timeout, continue waiting
            }
        }
    }
    println!(
        "[MockManager] Received {} heartbeats during {}s wait",
        heartbeat_count, args.wait_secs
    );

    // Step 3: Send directive(s)
    if let Some(scenario_path) = &args.scenario {
        run_scenario(stream, scenario_path)?;
    } else if let Some(cmd_name) = &args.command {
        run_single_command(args, stream, cmd_name)?;
    } else {
        #[cfg(feature = "dbus")]
        if !args.dbus {
            bail!("Either --command or --scenario must be specified for socket mode");
        }
        #[cfg(not(feature = "dbus"))]
        bail!("Either --command or --scenario must be specified");
    }

    // Step 4: Receive a few more heartbeats to observe effect
    println!("[MockManager] Observing post-directive heartbeats...");
    let observe_until = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < observe_until {
        match recv_message(stream)? {
            Some(EngineMessage::Heartbeat(status)) => {
                heartbeat_count += 1;
                println!(
                    "[MockManager] Heartbeat #{}: kv={}/{}B, tokens={}, tbt={:.1}ms",
                    heartbeat_count,
                    status.kv_cache_bytes,
                    status.kv_cache_budget_bytes,
                    status.kv_cache_tokens,
                    status.tbt_ms,
                );
            }
            Some(_) => {}
            None => {}
        }
    }

    println!("[MockManager] Done.");
    Ok(())
}

/// Outcome of a `recv_until` selector: either the extracted target value, or
/// the original message returned for logging.
///
/// `Skip` is intentionally larger than `Match`; this type only exists briefly
/// inside the selector loop, so the size asymmetry is fine.
#[allow(clippy::large_enum_variant)]
enum Selected<T> {
    Match(T),
    Skip(EngineMessage),
}

/// Receive messages until `select` returns `Match`, logging and discarding any
/// `Skip` variants. Returns `Ok(None)` on timeout.
///
/// Engine's MessageLoop sends Heartbeats and other side messages asynchronously,
/// so one or more may arrive between our Directive and its Response. This helper
/// centralises the skip-and-log logic so each caller only specifies which variant
/// it actually wants.
fn recv_until<T, F>(
    stream: &mut (impl Read + Write),
    timeout: Duration,
    waiting_for: &str,
    mut select: F,
) -> anyhow::Result<Option<T>>
where
    F: FnMut(EngineMessage) -> Selected<T>,
{
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        match recv_message(stream)? {
            Some(msg) => match select(msg) {
                Selected::Match(value) => return Ok(Some(value)),
                Selected::Skip(skipped) => log_skipped(&skipped, waiting_for),
            },
            None => {
                // read timeout / would-block, retry
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
    Ok(None)
}

fn log_skipped(msg: &EngineMessage, waiting_for: &str) {
    match msg {
        EngineMessage::Heartbeat(status) => {
            println!(
                "[MockManager] (skipping heartbeat while waiting for {}: kv={}/{}B, tokens={})",
                waiting_for,
                status.kv_cache_bytes,
                status.kv_cache_budget_bytes,
                status.kv_cache_tokens,
            );
        }
        EngineMessage::Response(resp) => {
            println!(
                "[MockManager] (unexpected Response seq_id={} while waiting for {}, skipping)",
                resp.seq_id, waiting_for,
            );
        }
    }
}

/// Receive a `CommandResponse`, skipping any interleaved side messages.
fn recv_response_skip_heartbeats(
    stream: &mut (impl Read + Write),
    timeout: Duration,
) -> anyhow::Result<Option<argus_shared::CommandResponse>> {
    recv_until(stream, timeout, "Response", |msg| match msg {
        EngineMessage::Response(resp) => Selected::Match(resp),
        other => Selected::Skip(other),
    })
}

fn run_single_command(
    args: &Args,
    stream: &mut (impl Read + Write),
    cmd_name: &str,
) -> anyhow::Result<()> {
    let cmd = build_command(&CommandParams {
        name: cmd_name,
        keep_ratio: args.keep_ratio,
    })?;

    let seq_id = 1u64;

    let directive = ManagerMessage::Directive(EngineDirective {
        seq_id,
        commands: vec![cmd],
    });

    send_message(stream, &directive)?;
    println!(
        "[MockManager] Sent: Directive seq_id={} [{}]",
        seq_id, cmd_name
    );

    // Wait for Response (skip interleaved Heartbeats)
    match recv_response_skip_heartbeats(stream, Duration::from_secs(5))? {
        Some(resp) => {
            validate_response(seq_id, &resp, 1);
            println!(
                "[MockManager] Response seq_id={}: {:?}",
                resp.seq_id, resp.results
            );
        }
        None => {
            println!(
                "[MockManager] Timed out waiting for Response (seq_id={})",
                seq_id
            );
        }
    }

    Ok(())
}

fn run_scenario(stream: &mut (impl Read + Write), path: &PathBuf) -> anyhow::Result<()> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("read scenario file: {:?}", path))?;
    let scenario: CommandScenario =
        serde_json::from_str(&content).context("parse scenario JSON")?;

    println!(
        "[MockManager] Playing scenario: {} ({} commands)",
        scenario.name,
        scenario.commands.len()
    );
    if let Some(desc) = &scenario.description {
        println!("  {}", desc);
    }

    let mut seq_id = 0u64;

    for (i, entry) in scenario.commands.iter().enumerate() {
        if entry.delay_ms > 0 {
            println!("  Waiting {}ms...", entry.delay_ms);

            // Drain heartbeats during wait
            let wait_until = std::time::Instant::now() + Duration::from_millis(entry.delay_ms);
            while std::time::Instant::now() < wait_until {
                match recv_message(stream) {
                    Ok(Some(EngineMessage::Heartbeat(status))) => {
                        println!(
                            "  (heartbeat: kv={}/{}B, tokens={})",
                            status.kv_cache_bytes,
                            status.kv_cache_budget_bytes,
                            status.kv_cache_tokens
                        );
                    }
                    _ => {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            }
        }

        let cmd = build_command(&CommandParams {
            name: &entry.command,
            keep_ratio: entry.keep_ratio, // reuse delay_ms for target_ms in scenario
        })?;

        seq_id += 1;

        let directive = ManagerMessage::Directive(EngineDirective {
            seq_id,
            commands: vec![cmd],
        });

        send_message(stream, &directive)?;
        println!(
            "  [{}/{}] Sent: {} (seq_id={})",
            i + 1,
            scenario.commands.len(),
            entry.command,
            seq_id
        );

        // Wait for Response (skip interleaved Heartbeats)
        match recv_response_skip_heartbeats(stream, Duration::from_secs(5))? {
            Some(resp) => {
                validate_response(seq_id, &resp, 1);
                println!("  Response: {:?}", resp.results);
            }
            None => {
                println!("  Timed out waiting for Response (seq_id={})", seq_id);
            }
        }
    }

    println!("[MockManager] Scenario complete.");
    Ok(())
}

// ── D-Bus mode (legacy) ─────────────────────────────────────────────────────

#[cfg(feature = "dbus")]
mod dbus_mode {
    use super::*;

    const MANAGER_NAME: &str = "org.llm.Manager1";
    const MANAGER_PATH: &str = "/org/llm/Manager1";
    const MANAGER_IFACE: &str = "org.llm.Manager1";

    /// D-Bus scenario file format.
    #[derive(Debug, Deserialize)]
    pub struct DbusScenario {
        pub name: String,
        #[serde(default)]
        pub description: Option<String>,
        pub signals: Vec<DbusScenarioSignal>,
    }

    #[derive(Debug, Deserialize)]
    pub struct DbusScenarioSignal {
        pub delay_ms: u64,
        pub signal: String,
        pub level: String,
        #[serde(default)]
        pub available_bytes: Option<u64>,
        #[serde(default)]
        pub reclaim_target_bytes: Option<u64>,
        #[serde(default)]
        pub recommended_backend: Option<String>,
        #[serde(default)]
        pub reason: Option<String>,
        #[serde(default)]
        pub cpu_usage_pct: Option<f64>,
        #[serde(default)]
        pub gpu_usage_pct: Option<f64>,
        #[serde(default)]
        pub temperature_mc: Option<i32>,
        #[serde(default)]
        pub throttling_active: Option<bool>,
        #[serde(default)]
        pub throttle_ratio: Option<f64>,
        #[serde(default)]
        pub power_budget_mw: Option<u32>,
    }

    pub fn run_dbus_mode(args: &Args) -> anyhow::Result<()> {
        let conn = zbus::blocking::Connection::system()?;
        conn.request_name(MANAGER_NAME)?;
        println!("Registered as {} on System Bus", MANAGER_NAME);

        if let Some(scenario_path) = &args.scenario {
            run_dbus_scenario(&conn, scenario_path)?;
        } else if let Some(signal_name) = &args.signal {
            emit_single(&conn, args, signal_name)?;
        } else {
            anyhow::bail!("D-Bus mode requires --signal or --scenario");
        }

        Ok(())
    }

    fn run_dbus_scenario(conn: &zbus::blocking::Connection, path: &PathBuf) -> anyhow::Result<()> {
        let content = std::fs::read_to_string(path)?;
        let scenario: DbusScenario = serde_json::from_str(&content)?;

        println!(
            "Playing scenario: {} ({} signals)",
            scenario.name,
            scenario.signals.len()
        );
        if let Some(desc) = &scenario.description {
            println!("  {}", desc);
        }

        for (i, entry) in scenario.signals.iter().enumerate() {
            if entry.delay_ms > 0 {
                println!("  Waiting {}ms...", entry.delay_ms);
                std::thread::sleep(Duration::from_millis(entry.delay_ms));
            }

            emit_scenario_signal(conn, entry)?;
            println!(
                "  [{}/{}] Emitted {} (level={})",
                i + 1,
                scenario.signals.len(),
                entry.signal,
                entry.level
            );
        }

        println!("Scenario complete.");
        Ok(())
    }

    fn emit_scenario_signal(
        conn: &zbus::blocking::Connection,
        entry: &DbusScenarioSignal,
    ) -> anyhow::Result<()> {
        match entry.signal.as_str() {
            "MemoryPressure" => {
                let body = (
                    &entry.level,
                    entry.available_bytes.unwrap_or(0),
                    entry.reclaim_target_bytes.unwrap_or(0),
                );
                conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "MemoryPressure",
                    &body,
                )?;
            }
            "ComputeGuidance" => {
                let body = (
                    &entry.level,
                    entry.recommended_backend.as_deref().unwrap_or("any"),
                    entry.reason.as_deref().unwrap_or("balanced"),
                    entry.cpu_usage_pct.unwrap_or(0.0),
                    entry.gpu_usage_pct.unwrap_or(0.0),
                );
                conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "ComputeGuidance",
                    &body,
                )?;
            }
            "ThermalAlert" => {
                let body = (
                    &entry.level,
                    entry.temperature_mc.unwrap_or(25000),
                    entry.throttling_active.unwrap_or(false),
                    entry.throttle_ratio.unwrap_or(1.0),
                );
                conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "ThermalAlert",
                    &body,
                )?;
            }
            "EnergyConstraint" => {
                let body = (
                    &entry.level,
                    entry.reason.as_deref().unwrap_or("none"),
                    entry.power_budget_mw.unwrap_or(0),
                );
                conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "EnergyConstraint",
                    &body,
                )?;
            }
            other => {
                anyhow::bail!("Unknown signal type: {}", other);
            }
        }
        Ok(())
    }

    fn emit_single(
        conn: &zbus::blocking::Connection,
        args: &Args,
        signal_name: &str,
    ) -> anyhow::Result<()> {
        let level = args.level.as_deref().unwrap_or("normal");

        match signal_name {
            "MemoryPressure" => {
                let body = (
                    level,
                    args.available_bytes.unwrap_or(0),
                    args.reclaim_target.unwrap_or(0),
                );
                conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "MemoryPressure",
                    &body,
                )?;
            }
            "ComputeGuidance" => {
                let body = (
                    level,
                    args.recommended_backend.as_deref().unwrap_or("any"),
                    args.reason.as_deref().unwrap_or("balanced"),
                    args.cpu_usage.unwrap_or(0.0),
                    args.gpu_usage.unwrap_or(0.0),
                );
                conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "ComputeGuidance",
                    &body,
                )?;
            }
            "ThermalAlert" => {
                let body = (
                    level,
                    args.temperature_mc.unwrap_or(25000),
                    args.throttling_active.unwrap_or(false),
                    args.throttle_ratio.unwrap_or(1.0),
                );
                conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "ThermalAlert",
                    &body,
                )?;
            }
            "EnergyConstraint" => {
                let body = (
                    level,
                    args.reason.as_deref().unwrap_or("none"),
                    args.power_budget_mw.unwrap_or(0),
                );
                conn.emit_signal(
                    Option::<&str>::None,
                    MANAGER_PATH,
                    MANAGER_IFACE,
                    "EnergyConstraint",
                    &body,
                )?;
            }
            other => {
                anyhow::bail!("Unknown signal: {}", other);
            }
        }

        println!("Emitted {} (level={})", signal_name, level);
        Ok(())
    }
}

// ── Entry point ─────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    #[cfg(feature = "dbus")]
    if args.dbus {
        return dbus_mode::run_dbus_mode(&args);
    }

    run_socket_mode(&args)
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The four commands the contract carries, and a refusal for anything else.
    #[test]
    fn build_command_covers_the_contract() {
        let cmd = build_command(&CommandParams {
            keep_ratio: Some(0.7),
            ..params("KvCompress")
        })
        .unwrap();
        assert!(
            matches!(cmd, EngineCommand::KvCompress { budget } if (budget - 0.7).abs() < f32::EPSILON)
        );

        for (name, want) in [
            ("RestoreDefaults", EngineCommand::RestoreDefaults),
            ("Suspend", EngineCommand::Suspend),
            ("Resume", EngineCommand::Resume),
        ] {
            assert_eq!(build_command(&params(name)).unwrap(), want);
        }

        assert!(
            build_command(&params("KvEvictH2o")).is_err(),
            "a technique name is not a command any more"
        );
    }
    use argus_shared::{CommandResponse, CommandResult, EngineState, EngineStatus, Phase};
    use std::os::unix::net::{UnixListener, UnixStream};

    fn tmp_sock() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mock_manager_test.sock");
        (dir, path)
    }

    // Helper: send an EngineMessage over a stream (engine side)
    fn engine_send(stream: &mut UnixStream, msg: &EngineMessage) {
        let json = serde_json::to_vec(msg).unwrap();
        let len = (json.len() as u32).to_be_bytes();
        stream.write_all(&len).unwrap();
        stream.write_all(&json).unwrap();
        stream.flush().unwrap();
    }

    fn make_heartbeat_status() -> EngineStatus {
        EngineStatus {
            kv_cache_bytes: 1024,
            kv_cache_budget_bytes: 4096,
            kv_cache_tokens: 32,
            tbt_ms: 12.5,
            phase: Phase::Decode,
            state: EngineState::Running,
        }
    }

    // ── Wire format round-trip tests ─────────────────────────────────────────

    #[test]
    fn send_message_writes_length_prefixed_json() {
        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        let directive = ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::Suspend],
        });

        send_message(&mut client, &directive).unwrap();

        // Read on server side
        let mut len_buf = [0u8; 4];
        server.read_exact(&mut len_buf).unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut json_buf = vec![0u8; len];
        server.read_exact(&mut json_buf).unwrap();

        let msg: ManagerMessage = serde_json::from_slice(&json_buf).unwrap();
        match msg {
            ManagerMessage::Directive(d) => {
                assert_eq!(d.seq_id, 1);
                assert!(matches!(d.commands[0], EngineCommand::Suspend));
            }
        }
    }

    #[test]
    fn recv_message_returns_none_on_timeout() {
        let (_dir, sock_path) = tmp_sock();
        let _listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(20)))
            .unwrap();

        let result = recv_message(&mut client).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn recv_message_parses_engine_message() {
        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        engine_send(
            &mut server,
            &EngineMessage::Heartbeat(EngineStatus {
                kv_cache_bytes: 1024,
                kv_cache_budget_bytes: 4096,
                kv_cache_tokens: 32,
                tbt_ms: 12.5,
                phase: Phase::Decode,
                state: EngineState::Running,
            }),
        );

        client
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        let msg = recv_message(&mut client).unwrap().unwrap();
        assert!(matches!(
            msg,
            EngineMessage::Heartbeat(EngineStatus {
                kv_cache_bytes: 1024,
                kv_cache_budget_bytes: 4096,
                kv_cache_tokens: 32,
                tbt_ms: 12.5,
                phase: Phase::Decode,
                state: EngineState::Running,
            })
        ));
    }

    // ── build_command tests ──────────────────────────────────────────────────

    fn params(name: &str) -> CommandParams<'_> {
        CommandParams {
            name,
            keep_ratio: None,
        }
    }

    // ── parse_dtype_tag tests ────────────────────────────────────────────────

    // ── validate_response tests ──────────────────────────────────────────────

    #[test]
    fn validate_response_ok() {
        let resp = CommandResponse {
            seq_id: 1,
            results: vec![CommandResult::Ok],
        };
        assert!(validate_response(1, &resp, 1));
    }

    #[test]
    fn validate_response_seq_id_mismatch() {
        let resp = CommandResponse {
            seq_id: 2,
            results: vec![CommandResult::Ok],
        };
        assert!(!validate_response(1, &resp, 1));
    }

    #[test]
    fn validate_response_results_count_mismatch() {
        let resp = CommandResponse {
            seq_id: 1,
            results: vec![CommandResult::Ok, CommandResult::Ok],
        };
        assert!(!validate_response(1, &resp, 1));
    }

    // ── Scenario deserialization tests ────────────────────────────────────────

    #[test]
    fn command_scenario_deserialize() {
        let json = r#"{
            "name": "test_scenario",
            "description": "A test",
            "commands": [
                { "delay_ms": 1000, "command": "KvEvictSliding", "keep_ratio": 0.8 },
                { "delay_ms": 500, "command": "RestoreDefaults" },
                { "delay_ms": 0, "command": "RequestQcf" }
            ]
        }"#;
        let scenario: CommandScenario = serde_json::from_str(json).unwrap();
        assert_eq!(scenario.name, "test_scenario");
        assert_eq!(scenario.commands.len(), 3);
        assert_eq!(scenario.commands[0].command, "KvEvictSliding");
        assert!((scenario.commands[0].keep_ratio.unwrap() - 0.8).abs() < f32::EPSILON);
        assert_eq!(scenario.commands[1].command, "RestoreDefaults");
        assert!(scenario.commands[1].keep_ratio.is_none());
    }

    // ── TCP transport tests ─────────────────────────────────────────────────

    #[test]
    fn send_recv_over_tcp_stream() {
        use std::net::{TcpListener, TcpStream};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        // Engine sends Capability over TCP
        let cap_msg = EngineMessage::Heartbeat(EngineStatus {
            kv_cache_bytes: 1024,
            kv_cache_budget_bytes: 4096,
            kv_cache_tokens: 32,
            tbt_ms: 12.5,
            phase: Phase::Decode,
            state: EngineState::Running,
        });
        let json = serde_json::to_vec(&cap_msg).unwrap();
        let len = (json.len() as u32).to_be_bytes();
        client.write_all(&len).unwrap();
        client.write_all(&json).unwrap();
        client.flush().unwrap();

        // Manager side receives it via recv_message (generic over TcpStream)
        server
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        let received = recv_message(&mut server).unwrap().unwrap();
        assert!(matches!(
            received,
            EngineMessage::Heartbeat(EngineStatus {
                kv_cache_bytes: 1024,
                kv_cache_budget_bytes: 4096,
                kv_cache_tokens: 32,
                tbt_ms: 12.5,
                phase: Phase::Decode,
                state: EngineState::Running,
            })
        ));

        // Manager side sends Directive over TCP via send_message
        let directive = ManagerMessage::Directive(EngineDirective {
            seq_id: 42,
            commands: vec![EngineCommand::Resume],
        });
        send_message(&mut server, &directive).unwrap();

        // Engine side reads raw length-prefixed JSON and parses as ManagerMessage
        let mut len_buf = [0u8; 4];
        client.read_exact(&mut len_buf).unwrap();
        let payload_len = u32::from_be_bytes(len_buf) as usize;
        let mut json_buf = vec![0u8; payload_len];
        client.read_exact(&mut json_buf).unwrap();
        let msg: ManagerMessage = serde_json::from_slice(&json_buf).unwrap();
        match msg {
            ManagerMessage::Directive(d) => {
                assert_eq!(d.seq_id, 42);
                assert!(matches!(d.commands[0], EngineCommand::Resume));
            }
        }
    }

    #[test]
    fn recv_message_tcp_returns_none_on_timeout() {
        use std::net::{TcpListener, TcpStream};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let mut client = TcpStream::connect(addr).unwrap();
        let (_server, _) = listener.accept().unwrap();

        client
            .set_read_timeout(Some(Duration::from_millis(20)))
            .unwrap();
        let result = recv_message(&mut client).unwrap();
        assert!(result.is_none());
    }

    // ── recv_response_skip_heartbeats tests ─────────────────────────────────

    #[test]
    fn recv_response_skip_heartbeats_skips_heartbeats() {
        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        server
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();

        // Engine sends: Heartbeat, Heartbeat, Response
        engine_send(
            &mut client,
            &EngineMessage::Heartbeat(make_heartbeat_status()),
        );
        engine_send(
            &mut client,
            &EngineMessage::Heartbeat(make_heartbeat_status()),
        );
        engine_send(
            &mut client,
            &EngineMessage::Response(CommandResponse {
                seq_id: 1,
                results: vec![CommandResult::Ok],
            }),
        );

        let resp = recv_response_skip_heartbeats(&mut server, Duration::from_secs(2))
            .unwrap()
            .expect("should receive Response");
        assert_eq!(resp.seq_id, 1);
        assert_eq!(resp.results.len(), 1);
    }

    #[test]
    fn recv_response_skip_heartbeats_returns_none_on_timeout() {
        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        server
            .set_read_timeout(Some(Duration::from_millis(50)))
            .unwrap();

        // Engine sends only heartbeats, no Response
        engine_send(
            &mut client,
            &EngineMessage::Heartbeat(make_heartbeat_status()),
        );

        let result =
            recv_response_skip_heartbeats(&mut server, Duration::from_millis(200)).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn recv_response_skip_heartbeats_immediate_response() {
        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        server
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();

        // Engine sends Response immediately (no heartbeats in between)
        engine_send(
            &mut client,
            &EngineMessage::Response(CommandResponse {
                seq_id: 42,
                results: vec![CommandResult::Ok, CommandResult::Ok],
            }),
        );

        let resp = recv_response_skip_heartbeats(&mut server, Duration::from_secs(2))
            .unwrap()
            .expect("should receive Response");
        assert_eq!(resp.seq_id, 42);
        assert_eq!(resp.results.len(), 2);
    }

    // ── recv_qcf_skip_heartbeats tests ──────────────────────────────────────
}
