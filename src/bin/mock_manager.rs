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

use argus_shared::{DtypeTag, EngineCommand, EngineDirective, EngineMessage, ManagerMessage};

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
#[derive(Debug, Deserialize)]
struct ScenarioCommand {
    delay_ms: u64,
    command: String,
    #[serde(default)]
    keep_ratio: Option<f32>,
    #[serde(default)]
    sink_size: Option<usize>,
    #[serde(default)]
    window_size: Option<usize>,
    #[serde(default)]
    delay_ms_param: Option<u64>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    target_bits: Option<u8>,
    #[serde(default)]
    skip_ratio: Option<f32>,
    #[serde(default)]
    ratio: Option<f32>,
    #[serde(default)]
    chunk_size: Option<usize>,
    #[serde(default)]
    yield_ms: Option<u32>,
    #[serde(default)]
    cpu_chunk_size: Option<usize>,
    #[serde(default)]
    target_dtype: Option<String>,
}

// ── Command construction ────────────────────────────────────────────────────

/// Parameters for building an EngineCommand from CLI or scenario input.
struct CommandParams<'a> {
    name: &'a str,
    keep_ratio: Option<f32>,
    sink_size: Option<usize>,
    window_size: Option<usize>,
    delay_ms: Option<u64>,
    device: Option<&'a str>,
    target_bits: Option<u8>,
    skip_ratio: Option<f32>,
    target_ms: Option<u64>,
    ratio: Option<f32>,
    chunk_size: Option<usize>,
    yield_ms: Option<u32>,
    cpu_chunk_size: Option<usize>,
    target_dtype: Option<&'a str>,
}

/// Parse a `--target-dtype` string into a `DtypeTag`.
fn parse_dtype_tag(s: &str) -> anyhow::Result<DtypeTag> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "q4_0" => DtypeTag::Q4_0,
        "f16" => DtypeTag::F16,
        "f32" => DtypeTag::F32,
        "q8_0" => DtypeTag::Q8_0,
        other => bail!(
            "unsupported --target-dtype '{}', expected q4_0/f16/f32/q8_0",
            other
        ),
    })
}

fn build_command(params: &CommandParams<'_>) -> anyhow::Result<EngineCommand> {
    match params.name {
        "KvEvictSliding" => {
            let ratio = params
                .keep_ratio
                .context("--keep-ratio required for KvEvictSliding")?;
            Ok(EngineCommand::KvEvictSliding { keep_ratio: ratio })
        }
        "KvEvictH2o" => {
            let ratio = params
                .keep_ratio
                .context("--keep-ratio required for KvEvictH2o")?;
            Ok(EngineCommand::KvEvictH2o { keep_ratio: ratio })
        }
        "KvStreaming" => {
            let sink = params
                .sink_size
                .context("--sink-size required for KvStreaming")?;
            let window = params
                .window_size
                .context("--window-size required for KvStreaming")?;
            Ok(EngineCommand::KvStreaming {
                sink_size: sink,
                window_size: window,
            })
        }
        "KvMergeD2o" => {
            let ratio = params
                .keep_ratio
                .context("--keep-ratio required for KvMergeD2o")?;
            Ok(EngineCommand::KvMergeD2o { keep_ratio: ratio })
        }
        "Throttle" => {
            let ms = params
                .delay_ms
                .context("--delay-ms required for Throttle")?;
            Ok(EngineCommand::Throttle { delay_ms: ms })
        }
        "SwitchHw" => {
            let dev = params.device.context("--device required for SwitchHw")?;
            Ok(EngineCommand::SwitchHw {
                device: dev.to_string(),
            })
        }
        "KvQuantDynamic" => {
            let bits = params
                .target_bits
                .context("--target-bits required for KvQuantDynamic")?;
            Ok(EngineCommand::KvQuantDynamic { target_bits: bits })
        }
        "LayerSkip" => {
            let ratio = params
                .skip_ratio
                .context("--skip-ratio required for LayerSkip")?;
            Ok(EngineCommand::LayerSkip { skip_ratio: ratio })
        }
        "SetTargetTbt" => {
            let ms = params
                .target_ms
                .context("--target-ms required for SetTargetTbt")?;
            Ok(EngineCommand::SetTargetTbt { target_ms: ms })
        }
        "SetPartitionRatio" => {
            let ratio = params
                .ratio
                .context("--ratio required for SetPartitionRatio")?;
            Ok(EngineCommand::SetPartitionRatio { ratio })
        }
        "KvOffload" => {
            let ratio = params.ratio.context("--ratio required for KvOffload")?;
            Ok(EngineCommand::KvOffload { ratio })
        }
        "SetPrefillPolicy" => Ok(EngineCommand::SetPrefillPolicy {
            chunk_size: params.chunk_size,
            yield_ms: params.yield_ms,
            cpu_chunk_size: params.cpu_chunk_size,
        }),
        "SwapWeights" => {
            let ratio = params.ratio.context("--ratio required for SwapWeights")?;
            let dtype_str = params
                .target_dtype
                .context("--target-dtype required for SwapWeights")?;
            let target_dtype = parse_dtype_tag(dtype_str)?;
            Ok(EngineCommand::SwapWeights {
                ratio,
                target_dtype,
            })
        }
        "RecallWeights" => {
            let ratio = params.ratio.context("--ratio required for RecallWeights")?;
            Ok(EngineCommand::RecallWeights { ratio })
        }
        "Suspend" => Ok(EngineCommand::Suspend),
        "Resume" => Ok(EngineCommand::Resume),
        "RestoreDefaults" => Ok(EngineCommand::RestoreDefaults),
        "RequestQcf" => Ok(EngineCommand::RequestQcf),
        other => bail!("Unknown command: {}", other),
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
    // Step 1: Receive Capability
    println!("[MockManager] Engine connected, waiting for Capability...");
    let capability = recv_blocking_with_timeout(stream, Duration::from_secs(5))?;
    match &capability {
        EngineMessage::Capability(cap) => {
            println!(
                "[MockManager] Engine Capability: devices={:?}, active={}, max_kv={}, layers={}",
                cap.available_devices, cap.active_device, cap.max_kv_tokens, cap.num_layers
            );
        }
        other => {
            bail!(
                "Expected Capability as first message, got: {:?}",
                std::mem::discriminant(other)
            );
        }
    }

    // Step 2: Wait for Heartbeats
    let wait_until = std::time::Instant::now() + Duration::from_secs(args.wait_secs);
    let mut heartbeat_count = 0u32;
    while std::time::Instant::now() < wait_until {
        match recv_message(stream)? {
            Some(EngineMessage::Heartbeat(status)) => {
                heartbeat_count += 1;
                println!(
                    "[MockManager] Heartbeat #{}: device={}, kv_util={:.3}, tokens={}, state={:?}, \
                     active_actions={:?}, kv_dtype={}",
                    heartbeat_count,
                    status.active_device,
                    status.kv_cache_utilization,
                    status.tokens_generated,
                    status.state,
                    status.active_actions,
                    status.kv_dtype,
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
                    "[MockManager] Heartbeat #{}: device={}, kv_util={:.3}, active_actions={:?}, \
                     kv_dtype={}",
                    heartbeat_count,
                    status.active_device,
                    status.kv_cache_utilization,
                    status.active_actions,
                    status.kv_dtype,
                );
            }
            Some(_) => {}
            None => {}
        }
    }

    println!("[MockManager] Done.");
    Ok(())
}

/// Receive a message with a hard timeout, retrying on WouldBlock.
fn recv_blocking_with_timeout(
    stream: &mut (impl Read + Write),
    timeout: Duration,
) -> anyhow::Result<EngineMessage> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if std::time::Instant::now() >= deadline {
            bail!("Timed out waiting for message ({:?})", timeout);
        }
        match recv_message(stream)? {
            Some(msg) => return Ok(msg),
            None => {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
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
                "[MockManager] (skipping heartbeat while waiting for {}: \
                 kv_util={:.3}, tokens={})",
                waiting_for, status.kv_cache_utilization, status.tokens_generated,
            );
        }
        EngineMessage::WeightSwapReport(r) => {
            println!(
                "[MockManager] (skipping WeightSwapReport while waiting for {}: \
                 {} layers, freed={}B)",
                waiting_for,
                r.layers_swapped.len(),
                r.freed_bytes,
            );
        }
        EngineMessage::Capability(_) => {
            println!("[MockManager] (unexpected Capability after handshake, skipping)");
        }
        EngineMessage::Response(resp) => {
            println!(
                "[MockManager] (unexpected Response seq_id={} while waiting for {}, skipping)",
                resp.seq_id, waiting_for,
            );
        }
        EngineMessage::QcfEstimate(_) => {
            println!(
                "[MockManager] (unexpected QcfEstimate while waiting for {}, skipping)",
                waiting_for,
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

/// Receive a `QcfEstimate`, skipping any interleaved side messages.
fn recv_qcf_skip_heartbeats(
    stream: &mut (impl Read + Write),
    timeout: Duration,
) -> anyhow::Result<Option<argus_shared::QcfEstimate>> {
    recv_until(stream, timeout, "QcfEstimate", |msg| match msg {
        EngineMessage::QcfEstimate(est) => Selected::Match(est),
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
        sink_size: args.sink_size,
        window_size: args.window_size,
        delay_ms: args.delay_ms,
        device: args.device.as_deref(),
        target_bits: args.target_bits,
        skip_ratio: args.skip_ratio,
        target_ms: args.target_ms,
        ratio: args.ratio,
        chunk_size: args.chunk_size,
        yield_ms: args.yield_ms,
        cpu_chunk_size: args.cpu_chunk_size,
        target_dtype: args.target_dtype.as_deref(),
    })?;

    let is_request_qcf = matches!(cmd, EngineCommand::RequestQcf);
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

    // TOOL-038: If RequestQcf, wait for QcfEstimate (skip interleaved Heartbeats)
    if is_request_qcf {
        match recv_qcf_skip_heartbeats(stream, Duration::from_secs(5))? {
            Some(qcf) => {
                println!(
                    "[MockManager] QcfEstimate received: {} entries",
                    qcf.estimates.len()
                );
                for (action, cost) in &qcf.estimates {
                    println!("  {}: {:.4}", action, cost);
                }
            }
            None => {
                println!("[MockManager] Timed out waiting for QcfEstimate");
            }
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
                            "  (heartbeat: kv_util={:.3}, actions={:?})",
                            status.kv_cache_utilization, status.active_actions
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
            keep_ratio: entry.keep_ratio,
            sink_size: entry.sink_size,
            window_size: entry.window_size,
            delay_ms: entry.delay_ms_param,
            device: entry.device.as_deref(),
            target_bits: entry.target_bits,
            skip_ratio: entry.skip_ratio,
            target_ms: entry.delay_ms_param, // reuse delay_ms for target_ms in scenario
            ratio: entry.ratio,
            chunk_size: entry.chunk_size,
            yield_ms: entry.yield_ms,
            cpu_chunk_size: entry.cpu_chunk_size,
            target_dtype: entry.target_dtype.as_deref(),
        })?;

        let is_request_qcf = matches!(cmd, EngineCommand::RequestQcf);
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

        if is_request_qcf {
            match recv_qcf_skip_heartbeats(stream, Duration::from_secs(5))? {
                Some(qcf) => {
                    println!("  QcfEstimate: {} entries", qcf.estimates.len());
                    for (action, cost) in &qcf.estimates {
                        println!("  {}: {:.4}", action, cost);
                    }
                }
                None => {
                    println!("  Timed out waiting for QcfEstimate");
                }
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
    use argus_shared::{
        CommandResponse, CommandResult, EngineCapability, EngineState, EngineStatus, ResourceLevel,
    };
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

    fn make_capability() -> EngineCapability {
        EngineCapability {
            available_devices: vec!["cpu".into(), "opencl".into()],
            active_device: "opencl".into(),
            max_kv_tokens: 2048,
            bytes_per_kv_token: 256,
            num_layers: 16,
            ..Default::default()
        }
    }

    fn make_heartbeat_status() -> EngineStatus {
        EngineStatus {
            active_device: "opencl".to_string(),
            compute_level: ResourceLevel::Normal,
            actual_throughput: 15.0,
            memory_level: ResourceLevel::Normal,
            kv_cache_bytes: 128_000,
            kv_cache_tokens: 500,
            kv_cache_utilization: 0.5,
            memory_lossless_min: 1.0,
            memory_lossy_min: 0.01,
            state: EngineState::Running,
            tokens_generated: 10,
            available_actions: vec!["throttle".into(), "kv.evict_sliding".into()],
            active_actions: vec![],
            eviction_policy: "none".into(),
            kv_dtype: "f16".into(),
            skip_ratio: 0.0,
            phase: String::new(),
            prefill_pos: 0,
            prefill_total: 0,
            partition_ratio: 0.0,
            self_cpu_pct: 0.0,
            self_gpu_pct: 0.0,
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

        engine_send(&mut server, &EngineMessage::Capability(make_capability()));

        client
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        let msg = recv_message(&mut client).unwrap().unwrap();
        assert!(matches!(msg, EngineMessage::Capability(_)));
    }

    // ── build_command tests ──────────────────────────────────────────────────

    fn params(name: &str) -> CommandParams<'_> {
        CommandParams {
            name,
            keep_ratio: None,
            sink_size: None,
            window_size: None,
            delay_ms: None,
            device: None,
            target_bits: None,
            skip_ratio: None,
            target_ms: None,
            ratio: None,
            chunk_size: None,
            yield_ms: None,
            cpu_chunk_size: None,
            target_dtype: None,
        }
    }

    #[test]
    fn build_command_kv_evict_sliding() {
        let cmd = build_command(&CommandParams {
            keep_ratio: Some(0.7),
            ..params("KvEvictSliding")
        })
        .unwrap();
        match cmd {
            EngineCommand::KvEvictSliding { keep_ratio } => {
                assert!((keep_ratio - 0.7).abs() < f32::EPSILON);
            }
            _ => panic!("Expected KvEvictSliding"),
        }
    }

    #[test]
    fn build_command_kv_evict_sliding_missing_ratio() {
        let result = build_command(&params("KvEvictSliding"));
        assert!(result.is_err());
    }

    #[test]
    fn build_command_kv_streaming() {
        let cmd = build_command(&CommandParams {
            sink_size: Some(4),
            window_size: Some(256),
            ..params("KvStreaming")
        })
        .unwrap();
        match cmd {
            EngineCommand::KvStreaming {
                sink_size,
                window_size,
            } => {
                assert_eq!(sink_size, 4);
                assert_eq!(window_size, 256);
            }
            _ => panic!("Expected KvStreaming"),
        }
    }

    #[test]
    fn build_command_throttle() {
        let cmd = build_command(&CommandParams {
            delay_ms: Some(50),
            ..params("Throttle")
        })
        .unwrap();
        match cmd {
            EngineCommand::Throttle { delay_ms } => assert_eq!(delay_ms, 50),
            _ => panic!("Expected Throttle"),
        }
    }

    #[test]
    fn build_command_switch_hw() {
        let cmd = build_command(&CommandParams {
            device: Some("cpu"),
            ..params("SwitchHw")
        })
        .unwrap();
        match cmd {
            EngineCommand::SwitchHw { device } => assert_eq!(device, "cpu"),
            _ => panic!("Expected SwitchHw"),
        }
    }

    #[test]
    fn build_command_layer_skip() {
        let cmd = build_command(&CommandParams {
            skip_ratio: Some(0.3),
            ..params("LayerSkip")
        })
        .unwrap();
        match cmd {
            EngineCommand::LayerSkip { skip_ratio } => {
                assert!((skip_ratio - 0.3).abs() < f32::EPSILON);
            }
            _ => panic!("Expected LayerSkip"),
        }
    }

    #[test]
    fn build_command_parameterless_variants() {
        assert!(matches!(
            build_command(&params("Suspend")).unwrap(),
            EngineCommand::Suspend
        ));
        assert!(matches!(
            build_command(&params("Resume")).unwrap(),
            EngineCommand::Resume
        ));
        assert!(matches!(
            build_command(&params("RestoreDefaults")).unwrap(),
            EngineCommand::RestoreDefaults
        ));
        assert!(matches!(
            build_command(&params("RequestQcf")).unwrap(),
            EngineCommand::RequestQcf
        ));
    }

    #[test]
    fn build_command_unknown() {
        let result = build_command(&params("FooBar"));
        assert!(result.is_err());
    }

    #[test]
    fn build_command_set_partition_ratio() {
        let cmd = build_command(&CommandParams {
            ratio: Some(0.5),
            ..params("SetPartitionRatio")
        })
        .unwrap();
        match cmd {
            EngineCommand::SetPartitionRatio { ratio } => {
                assert!((ratio - 0.5).abs() < f32::EPSILON);
            }
            _ => panic!("Expected SetPartitionRatio"),
        }
    }

    #[test]
    fn build_command_set_partition_ratio_missing_ratio() {
        let result = build_command(&params("SetPartitionRatio"));
        assert!(result.is_err());
    }

    #[test]
    fn build_command_set_prefill_policy_full() {
        let cmd = build_command(&CommandParams {
            chunk_size: Some(48),
            yield_ms: Some(10),
            cpu_chunk_size: Some(16),
            ..params("SetPrefillPolicy")
        })
        .unwrap();
        match cmd {
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                assert_eq!(chunk_size, Some(48));
                assert_eq!(yield_ms, Some(10));
                assert_eq!(cpu_chunk_size, Some(16));
            }
            _ => panic!("Expected SetPrefillPolicy"),
        }
    }

    #[test]
    fn build_command_set_prefill_policy_partial() {
        let cmd = build_command(&CommandParams {
            chunk_size: Some(64),
            ..params("SetPrefillPolicy")
        })
        .unwrap();
        match cmd {
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                assert_eq!(chunk_size, Some(64));
                assert!(yield_ms.is_none());
                assert!(cpu_chunk_size.is_none());
            }
            _ => panic!("Expected SetPrefillPolicy"),
        }
    }

    #[test]
    fn build_command_set_prefill_policy_empty() {
        // All None is valid — engine keeps current values
        let cmd = build_command(&params("SetPrefillPolicy")).unwrap();
        match cmd {
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                assert!(chunk_size.is_none());
                assert!(yield_ms.is_none());
                assert!(cpu_chunk_size.is_none());
            }
            _ => panic!("Expected SetPrefillPolicy"),
        }
    }

    #[test]
    fn build_command_swap_weights() {
        let cmd = build_command(&CommandParams {
            ratio: Some(0.5),
            target_dtype: Some("q4_0"),
            ..params("SwapWeights")
        })
        .unwrap();
        match cmd {
            EngineCommand::SwapWeights {
                ratio,
                target_dtype,
            } => {
                assert!((ratio - 0.5).abs() < f32::EPSILON);
                assert_eq!(target_dtype, DtypeTag::Q4_0);
            }
            _ => panic!("Expected SwapWeights"),
        }
    }

    #[test]
    fn build_command_swap_weights_uppercase_dtype() {
        let cmd = build_command(&CommandParams {
            ratio: Some(0.25),
            target_dtype: Some("F16"),
            ..params("SwapWeights")
        })
        .unwrap();
        match cmd {
            EngineCommand::SwapWeights { target_dtype, .. } => {
                assert_eq!(target_dtype, DtypeTag::F16);
            }
            _ => panic!("Expected SwapWeights"),
        }
    }

    #[test]
    fn build_command_swap_weights_missing_ratio() {
        let result = build_command(&CommandParams {
            target_dtype: Some("q4_0"),
            ..params("SwapWeights")
        });
        assert!(result.is_err());
    }

    #[test]
    fn build_command_swap_weights_missing_dtype() {
        let result = build_command(&CommandParams {
            ratio: Some(0.5),
            ..params("SwapWeights")
        });
        assert!(result.is_err());
    }

    #[test]
    fn build_command_swap_weights_invalid_dtype() {
        let result = build_command(&CommandParams {
            ratio: Some(0.5),
            target_dtype: Some("bf16"),
            ..params("SwapWeights")
        });
        assert!(result.is_err());
    }

    // ── parse_dtype_tag tests ────────────────────────────────────────────────

    #[test]
    fn parse_dtype_tag_known_variants() {
        assert_eq!(parse_dtype_tag("q4_0").unwrap(), DtypeTag::Q4_0);
        assert_eq!(parse_dtype_tag("Q4_0").unwrap(), DtypeTag::Q4_0);
        assert_eq!(parse_dtype_tag("f16").unwrap(), DtypeTag::F16);
        assert_eq!(parse_dtype_tag("F32").unwrap(), DtypeTag::F32);
        assert_eq!(parse_dtype_tag("q8_0").unwrap(), DtypeTag::Q8_0);
    }

    #[test]
    fn parse_dtype_tag_rejects_unknown() {
        assert!(parse_dtype_tag("bf16").is_err());
        assert!(parse_dtype_tag("").is_err());
    }

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

    #[test]
    fn command_scenario_deserialize_new_commands() {
        let json = r#"{
            "name": "partition_and_prefill",
            "commands": [
                { "delay_ms": 5000, "command": "SetPartitionRatio", "ratio": 0.5 },
                { "delay_ms": 3000, "command": "SetPrefillPolicy", "chunk_size": 48, "yield_ms": 10, "cpu_chunk_size": 16 },
                { "delay_ms": 1000, "command": "SetPrefillPolicy", "chunk_size": 64 }
            ]
        }"#;
        let scenario: CommandScenario = serde_json::from_str(json).unwrap();
        assert_eq!(scenario.name, "partition_and_prefill");
        assert_eq!(scenario.commands.len(), 3);
        assert_eq!(scenario.commands[0].command, "SetPartitionRatio");
        assert!((scenario.commands[0].ratio.unwrap() - 0.5).abs() < f32::EPSILON);
        assert_eq!(scenario.commands[1].command, "SetPrefillPolicy");
        assert_eq!(scenario.commands[1].chunk_size, Some(48));
        assert_eq!(scenario.commands[1].yield_ms, Some(10));
        assert_eq!(scenario.commands[1].cpu_chunk_size, Some(16));
        // partial: only chunk_size
        assert_eq!(scenario.commands[2].chunk_size, Some(64));
        assert!(scenario.commands[2].yield_ms.is_none());
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
        let cap_msg = EngineMessage::Capability(make_capability());
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
        assert!(matches!(received, EngineMessage::Capability(_)));

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
    fn recv_response_skip_heartbeats_skips_weight_swap_report() {
        use argus_shared::WeightSwapReport;

        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        server
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();

        // Engine sends a WeightSwapReport (side message) before the Response.
        // recv_until must skip it and still return the Response.
        engine_send(
            &mut client,
            &EngineMessage::WeightSwapReport(WeightSwapReport {
                layers_swapped: vec![],
                freed_bytes: 0,
                latency_ms: 1,
                qcf_swap_actual: 0.0,
            }),
        );
        engine_send(
            &mut client,
            &EngineMessage::Response(CommandResponse {
                seq_id: 7,
                results: vec![CommandResult::Ok],
            }),
        );

        let resp = recv_response_skip_heartbeats(&mut server, Duration::from_secs(2))
            .unwrap()
            .expect("should receive Response");
        assert_eq!(resp.seq_id, 7);
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

    #[test]
    fn recv_qcf_skip_heartbeats_skips_heartbeats() {
        use std::collections::HashMap;

        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        server
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();

        // Engine sends: Heartbeat, QcfEstimate
        engine_send(
            &mut client,
            &EngineMessage::Heartbeat(make_heartbeat_status()),
        );
        let mut estimates = HashMap::new();
        estimates.insert("kv.evict_h2o".to_string(), 0.15f32);
        engine_send(
            &mut client,
            &EngineMessage::QcfEstimate(argus_shared::QcfEstimate {
                estimates,
                layer_swap: None,
            }),
        );

        let qcf = recv_qcf_skip_heartbeats(&mut server, Duration::from_secs(2))
            .unwrap()
            .expect("should receive QcfEstimate");
        assert_eq!(qcf.estimates.len(), 1);
        assert!((qcf.estimates["kv.evict_h2o"] - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn recv_qcf_skip_heartbeats_returns_none_on_timeout() {
        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        server
            .set_read_timeout(Some(Duration::from_millis(50)))
            .unwrap();

        // Engine sends only heartbeat
        engine_send(
            &mut client,
            &EngineMessage::Heartbeat(make_heartbeat_status()),
        );

        let result = recv_qcf_skip_heartbeats(&mut server, Duration::from_millis(200)).unwrap();
        assert!(result.is_none());
    }
}
