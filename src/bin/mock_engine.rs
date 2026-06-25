//! Mock Engine binary for protocol validation and E2E testing.
//!
//! Connects to Manager's Unix socket, sends Capability + periodic Heartbeat,
//! and receives Directive messages. Logs every received directive and sends
//! back a CommandResponse. Updates internal state (kv_occupancy, active_device)
//! according to received commands so the simulation is observable.
//!
//! # Purpose
//!
//! Verifies that Manager's PolicyPipeline + UnixSocketEmitter correctly
//! serialises and transmits ManagerMessage::Directive JSON over the wire.
//!
//! # Usage
//!
//! ```bash
//! # Terminal 1 — Manager
//! RUST_LOG=info cargo run -p argus_manager -- \
//!     --transport unix:/tmp/argus_manager.sock
//!
//! # Terminal 2 — Mock Engine
//! RUST_LOG=info cargo run -p argus_manager --bin mock_engine -- \
//!     --socket /tmp/argus_manager.sock \
//!     --heartbeat-ms 100 \
//!     --kv-occupancy 0.5 \
//!     --duration-secs 30
//! ```

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use anyhow::Context;
use argus_shared::{
    CommandResponse, CommandResult, EngineCapability, EngineCommand, EngineDirective,
    EngineMessage, EngineState, EngineStatus, ManagerMessage, QcfEstimate, ResourceLevel,
};
use clap::Parser;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "mock_engine",
    about = "Mock Engine for Manager protocol validation and E2E testing"
)]
struct Args {
    /// Unix socket path that Manager is listening on (ignored when --tcp is set).
    #[arg(long, default_value = "/tmp/argus_manager.sock")]
    socket: String,

    /// TCP address to connect to (e.g. 127.0.0.1:9999).
    /// When set, TCP is used instead of Unix socket.
    #[arg(long)]
    tcp: Option<String>,

    /// Heartbeat send interval in milliseconds.
    #[arg(long, default_value_t = 100)]
    heartbeat_ms: u64,

    /// Simulated KV cache occupancy (0.0–1.0).
    #[arg(long, default_value_t = 0.5)]
    kv_occupancy: f32,

    /// Active compute device to report ("cpu" or "opencl").
    #[arg(long, default_value = "opencl")]
    device: String,

    /// How long to run before exiting (seconds).
    #[arg(long, default_value_t = 30)]
    duration_secs: u64,
}

// ── Wire format helpers ──────────────────────────────────────────────────────

/// Serialise `msg` as length-prefixed JSON and write to `stream`.
///
/// Wire format: `[4-byte BE u32 length][UTF-8 JSON]`
/// This matches the format used by `UnixSocketEmitter` on the Manager side.
fn send_message(stream: &mut (impl Read + Write), msg: &EngineMessage) -> anyhow::Result<()> {
    let json = serde_json::to_vec(msg).context("serialise EngineMessage")?;
    let len = (json.len() as u32).to_be_bytes();
    stream.write_all(&len).context("write length prefix")?;
    stream.write_all(&json).context("write JSON payload")?;
    stream.flush().context("flush stream")?;
    Ok(())
}

/// Try to read one `ManagerMessage` from `stream`.
///
/// Returns `Ok(None)` on read timeout / would-block (non-blocking read).
/// Returns `Err` on unrecoverable I/O or JSON parse errors.
fn recv_message(stream: &mut (impl Read + Write)) -> anyhow::Result<Option<ManagerMessage>> {
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

    let msg: ManagerMessage =
        serde_json::from_slice(&json_buf).context("deserialise ManagerMessage")?;
    Ok(Some(msg))
}

// ── State ────────────────────────────────────────────────────────────────────

/// All action identifiers the engine can support.
const ALL_AVAILABLE_ACTIONS: &[&str] = &[
    "switch_hw",
    "throttle",
    "kv.evict_sliding",
    "kv.evict_h2o",
    "kv.evict_streaming",
    "kv.merge_d2o",
    "kv.quant_dynamic",
    "weight.skip",
];

/// Mutable engine state that updates in response to received Directives.
struct EngineState_ {
    kv_occupancy: f32,
    active_device: String,
    throttle_delay_ms: u64,
    eviction_policy: String,
    skip_ratio: f32,
    state: EngineState,
    tokens_generated: usize,
    active_actions: Vec<String>,
    available_actions: Vec<String>,
}

impl EngineState_ {
    fn new(kv_occupancy: f32, device: String) -> Self {
        Self {
            kv_occupancy,
            active_device: device,
            throttle_delay_ms: 0,
            eviction_policy: "none".to_string(),
            skip_ratio: 0.0,
            state: EngineState::Running,
            tokens_generated: 0,
            active_actions: vec![],
            available_actions: ALL_AVAILABLE_ACTIONS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    /// Add an action to active_actions if not already present.
    fn activate_action(&mut self, action: &str) {
        if !self.active_actions.iter().any(|a| a == action) {
            self.active_actions.push(action.to_string());
        }
    }

    /// Apply a single `EngineCommand` and return a human-readable description
    /// of what changed.
    fn apply(&mut self, cmd: &EngineCommand) -> CommandResult {
        match cmd {
            EngineCommand::KvEvictSliding { keep_ratio } => {
                let before = self.kv_occupancy;
                self.kv_occupancy = (self.kv_occupancy * keep_ratio).clamp(0.01, 1.0);
                self.eviction_policy = "sliding".to_string();
                self.activate_action("kv.evict_sliding");
                println!(
                    "  → KvEvictSliding: kv_occupancy {:.3} → {:.3} (keep_ratio={:.2})",
                    before, self.kv_occupancy, keep_ratio
                );
                CommandResult::Ok
            }
            EngineCommand::KvEvictH2o { keep_ratio } => {
                let before = self.kv_occupancy;
                self.kv_occupancy = (self.kv_occupancy * keep_ratio).clamp(0.01, 1.0);
                self.eviction_policy = "h2o".to_string();
                self.activate_action("kv.evict_h2o");
                println!(
                    "  → KvEvictH2o: kv_occupancy {:.3} → {:.3} (keep_ratio={:.2})",
                    before, self.kv_occupancy, keep_ratio
                );
                CommandResult::Ok
            }
            EngineCommand::KvStreaming {
                sink_size,
                window_size,
            } => {
                self.eviction_policy = "streaming".to_string();
                self.activate_action("kv.evict_streaming");
                println!(
                    "  → KvStreaming: sink_size={} window_size={}",
                    sink_size, window_size
                );
                CommandResult::Ok
            }
            EngineCommand::KvMergeD2o { keep_ratio } => {
                let before = self.kv_occupancy;
                self.kv_occupancy = (self.kv_occupancy * keep_ratio).clamp(0.01, 1.0);
                self.eviction_policy = "d2o".to_string();
                self.activate_action("kv.merge_d2o");
                println!(
                    "  → KvMergeD2o: kv_occupancy {:.3} → {:.3} (keep_ratio={:.2})",
                    before, self.kv_occupancy, keep_ratio
                );
                CommandResult::Ok
            }
            EngineCommand::KvQuantDynamic { target_bits } => {
                self.activate_action("kv.quant_dynamic");
                println!("  → KvQuantDynamic: target_bits={}", target_bits);
                CommandResult::Ok
            }
            EngineCommand::KvReencodeFormat { format } => {
                self.activate_action("kv.reencode_format");
                println!("  → KvReencodeFormat: format={}", format);
                CommandResult::Ok
            }
            EngineCommand::Throttle { delay_ms } => {
                self.throttle_delay_ms = *delay_ms;
                self.activate_action("throttle");
                println!("  → Throttle: delay_ms={}", delay_ms);
                CommandResult::Ok
            }
            EngineCommand::SetTargetTbt { target_ms } => {
                self.activate_action("target_tbt");
                println!("  → SetTargetTbt: target_ms={}", target_ms);
                CommandResult::Ok
            }
            EngineCommand::LayerSkip { skip_ratio } => {
                self.skip_ratio = *skip_ratio;
                self.activate_action("weight.skip");
                println!("  → LayerSkip: skip_ratio={:.2}", skip_ratio);
                CommandResult::Ok
            }
            EngineCommand::SwitchHw { device } => {
                println!("  → SwitchHw: {} → {}", self.active_device, device);
                self.active_device = device.clone();
                self.activate_action("switch_hw");
                CommandResult::Ok
            }
            EngineCommand::PrepareComputeUnit { device } => {
                println!("  → PrepareComputeUnit: {}", device);
                CommandResult::Ok
            }
            EngineCommand::RestoreDefaults => {
                self.throttle_delay_ms = 0;
                self.skip_ratio = 0.0;
                self.eviction_policy = "none".to_string();
                self.active_actions.clear();
                println!("  → RestoreDefaults");
                CommandResult::Ok
            }
            EngineCommand::Suspend => {
                println!("  → Suspend");
                self.state = EngineState::Suspended;
                CommandResult::Ok
            }
            EngineCommand::Resume => {
                println!("  → Resume");
                self.state = EngineState::Running;
                CommandResult::Ok
            }
            EngineCommand::RequestQcf => {
                println!("  → RequestQcf (returning Ok + QcfEstimate)");
                CommandResult::Ok
            }
            EngineCommand::SetPartitionRatio { ratio } => {
                println!("  → SetPartitionRatio({})", ratio);
                CommandResult::Ok
            }
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                println!(
                    "  → SetPrefillPolicy(chunk={:?}, yield={:?}, cpu_chunk={:?})",
                    chunk_size, yield_ms, cpu_chunk_size
                );
                CommandResult::Ok
            }
            EngineCommand::KvOffload { ratio } => {
                println!("  → KvOffload(ratio={})", ratio);
                CommandResult::Ok
            }
            EngineCommand::SwapWeights {
                ratio,
                target_dtype,
            } => {
                // Mock engine: acknowledge but do not actually swap weights.
                println!(
                    "  → SwapWeights(ratio={:.2}, dtype={:?})",
                    ratio, target_dtype
                );
                CommandResult::Ok
            }
            EngineCommand::RecallWeights { ratio } => {
                // Mock engine: acknowledge but do not actually recall weights.
                println!("  → RecallWeights(ratio={:.2})", ratio);
                CommandResult::Ok
            }
        }
    }

    /// Build the current `EngineStatus` heartbeat from internal state.
    fn status(&self) -> EngineStatus {
        const MAX_KV_TOKENS: usize = 2048;
        const BYTES_PER_TOKEN: u64 = 256;
        let kv_tokens = (self.kv_occupancy * MAX_KV_TOKENS as f32) as usize;
        EngineStatus {
            active_device: self.active_device.clone(),
            compute_level: ResourceLevel::Normal,
            actual_throughput: 15.0,
            memory_level: ResourceLevel::Normal,
            kv_cache_bytes: kv_tokens as u64 * BYTES_PER_TOKEN,
            kv_cache_tokens: kv_tokens,
            kv_cache_utilization: self.kv_occupancy,
            memory_lossless_min: 1.0,
            memory_lossy_min: 0.01,
            state: self.state,
            tokens_generated: self.tokens_generated,
            available_actions: self.available_actions.clone(),
            active_actions: self.active_actions.clone(),
            eviction_policy: self.eviction_policy.clone(),
            kv_dtype: "f16".to_string(),
            skip_ratio: self.skip_ratio,
            phase: String::new(),
            prefill_pos: 0,
            prefill_total: 0,
            partition_ratio: 0.0,
            self_cpu_pct: 0.0,
            self_gpu_pct: 0.0,
        }
    }
}

// ── Entry point ──────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    if let Some(ref addr) = args.tcp {
        println!("[MockEngine] Connecting via TCP to {}", addr);
        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("TCP connect to {}", addr))?;
        stream
            .set_read_timeout(Some(Duration::from_millis(50)))
            .context("set_read_timeout")?;
        run_protocol(&args, &mut stream)
    } else {
        #[cfg(unix)]
        {
            println!("[MockEngine] Connecting to {}", args.socket);
            let mut stream = UnixStream::connect(&args.socket)
                .with_context(|| format!("connect to {}", args.socket))?;
            stream
                .set_read_timeout(Some(Duration::from_millis(50)))
                .context("set_read_timeout")?;
            run_protocol(&args, &mut stream)
        }
        #[cfg(not(unix))]
        {
            anyhow::bail!("Unix sockets not supported on this platform; use --tcp instead");
        }
    }
}

fn run_protocol(args: &Args, stream: &mut (impl Read + Write)) -> anyhow::Result<()> {
    // ── Step 1: Capability ────────────────────────────────────────────────────
    let capability = EngineCapability {
        available_devices: vec!["cpu".to_string(), "opencl".to_string()],
        active_device: args.device.clone(),
        max_kv_tokens: 2048,
        bytes_per_kv_token: 256,
        num_layers: 16,
        available_actions: ALL_AVAILABLE_ACTIONS
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    send_message(stream, &EngineMessage::Capability(capability)).context("send Capability")?;
    println!("[MockEngine] Sent Capability (device={})", args.device);

    // ── Step 2: Main loop ─────────────────────────────────────────────────────
    let run_duration = Duration::from_secs(args.duration_secs);
    let heartbeat_interval = Duration::from_millis(args.heartbeat_ms);

    let mut engine = EngineState_::new(args.kv_occupancy, args.device.clone());
    let start = Instant::now();
    let mut last_heartbeat = Instant::now();
    let mut directives_received: u32 = 0;
    let mut heartbeats_sent: u32 = 0;

    println!(
        "[MockEngine] Running for {}s (heartbeat={}ms, kv_occupancy={:.2})",
        args.duration_secs, args.heartbeat_ms, args.kv_occupancy
    );

    while start.elapsed() < run_duration {
        // ── Heartbeat ─────────────────────────────────────────────────────────
        if last_heartbeat.elapsed() >= heartbeat_interval {
            engine.tokens_generated += 1; // simulate token generation
            let status = engine.status();
            match send_message(stream, &EngineMessage::Heartbeat(status)) {
                Ok(()) => {
                    heartbeats_sent += 1;
                    log::debug!("[MockEngine] Heartbeat #{} sent", heartbeats_sent);
                }
                Err(e) => {
                    eprintln!("[MockEngine] Heartbeat send error: {} — exiting", e);
                    break;
                }
            }
            last_heartbeat = Instant::now();
        }

        // ── Receive Directive (non-blocking) ──────────────────────────────────
        match recv_message(stream) {
            Ok(Some(ManagerMessage::Directive(directive))) => {
                handle_directive(&directive, &mut engine, &mut directives_received, stream);
            }
            Ok(None) => {
                // Timeout — no message yet; loop back
            }
            Err(e) => {
                eprintln!("[MockEngine] Read error: {} — exiting", e);
                break;
            }
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\n[MockEngine] ── Summary ────────────────────────────");
    println!(
        "  Elapsed:             {:.1}s",
        start.elapsed().as_secs_f32()
    );
    println!("  Heartbeats sent:     {}", heartbeats_sent);
    println!("  Directives received: {}", directives_received);
    println!("  Final kv_occupancy:  {:.3}", engine.kv_occupancy);
    println!("  Final device:        {}", engine.active_device);
    println!("  Final throttle_ms:   {}", engine.throttle_delay_ms);
    println!("  Final eviction:      {}", engine.eviction_policy);
    println!("  Final skip_ratio:    {:.2}", engine.skip_ratio);
    println!("  Final state:         {:?}", engine.state);
    println!("────────────────────────────────────────────────────");

    Ok(())
}

// ── Directive handler ────────────────────────────────────────────────────────

fn handle_directive(
    directive: &EngineDirective,
    engine: &mut EngineState_,
    count: &mut u32,
    stream: &mut (impl Read + Write),
) {
    *count += 1;
    println!(
        "\n[MockEngine] Directive #{} seq={} ({} commands)",
        count,
        directive.seq_id,
        directive.commands.len()
    );

    let results: Vec<CommandResult> = directive
        .commands
        .iter()
        .enumerate()
        .map(|(i, cmd)| {
            println!("  [{}] {:?}", i, cmd);
            engine.apply(cmd)
        })
        .collect();

    let response = CommandResponse {
        seq_id: directive.seq_id,
        results,
    };

    if let Err(e) = send_message(stream, &EngineMessage::Response(response)) {
        eprintln!(
            "[MockEngine] Failed to send Response for seq={}: {}",
            directive.seq_id, e
        );
        return;
    }
    println!("[MockEngine] Response sent for seq={}", directive.seq_id);

    // TOOL-019: If any command was RequestQcf, send a separate QcfEstimate message
    let has_request_qcf = directive
        .commands
        .iter()
        .any(|c| matches!(c, EngineCommand::RequestQcf));
    if has_request_qcf {
        let mut estimates = HashMap::new();
        estimates.insert("kv.evict_sliding".to_string(), 0.05_f32);
        estimates.insert("kv.evict_h2o".to_string(), 0.12);
        estimates.insert("kv.merge_d2o".to_string(), 0.08);
        estimates.insert("kv.quant_dynamic".to_string(), 0.15);
        estimates.insert("weight.skip".to_string(), 0.20);
        let qcf = QcfEstimate {
            estimates,
            layer_swap: None,
        };
        if let Err(e) = send_message(stream, &EngineMessage::QcfEstimate(qcf)) {
            eprintln!(
                "[MockEngine] Failed to send QcfEstimate for seq={}: {}",
                directive.seq_id, e
            );
        } else {
            println!("[MockEngine] QcfEstimate sent for seq={}", directive.seq_id);
        }
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::os::unix::net::UnixListener;

    fn tmp_sock() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mock_engine_test.sock");
        (dir, path)
    }

    // ── send_message / recv_message round-trip ────────────────────────────────

    #[test]
    fn roundtrip_capability_over_socket() {
        let (_dir, sock_path) = tmp_sock();

        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();

        // Server side: accept and read raw bytes
        let (mut server, _) = listener.accept().unwrap();

        let cap = EngineCapability {
            available_devices: vec!["cpu".into(), "opencl".into()],
            active_device: "opencl".into(),
            max_kv_tokens: 2048,
            bytes_per_kv_token: 256,
            num_layers: 16,
            ..Default::default()
        };

        send_message(&mut client, &EngineMessage::Capability(cap.clone())).unwrap();

        // Read length prefix
        let mut len_buf = [0u8; 4];
        server.read_exact(&mut len_buf).unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;

        let mut json_buf = vec![0u8; len];
        server.read_exact(&mut json_buf).unwrap();

        let msg: EngineMessage = serde_json::from_slice(&json_buf).unwrap();
        match msg {
            EngineMessage::Capability(c) => {
                assert_eq!(c.active_device, "opencl");
                assert_eq!(c.max_kv_tokens, 2048);
                assert_eq!(c.num_layers, 16);
            }
            _ => panic!("Expected Capability"),
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

        // Nothing to receive — should return Ok(None)
        let result = recv_message(&mut client).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn recv_message_parses_directive() {
        use argus_shared::{EngineDirective, ManagerMessage};

        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        // Server sends a directive
        let directive = ManagerMessage::Directive(EngineDirective {
            seq_id: 7,
            commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }],
        });
        let json = serde_json::to_vec(&directive).unwrap();
        let len = (json.len() as u32).to_be_bytes();
        server.write_all(&len).unwrap();
        server.write_all(&json).unwrap();
        server.flush().unwrap();

        client
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        let msg = recv_message(&mut client).unwrap().unwrap();
        match msg {
            ManagerMessage::Directive(d) => {
                assert_eq!(d.seq_id, 7);
                assert_eq!(d.commands.len(), 1);
            }
        }
    }

    // ── EngineState_ ─────────────────────────────────────────────────────────

    #[test]
    fn apply_kv_evict_sliding_reduces_kv_occupancy() {
        let mut s = EngineState_::new(0.8, "opencl".into());
        let cmd = EngineCommand::KvEvictSliding { keep_ratio: 0.5 };
        let result = s.apply(&cmd);
        assert!(matches!(result, CommandResult::Ok));
        // 0.8 * 0.5 = 0.4
        assert!((s.kv_occupancy - 0.4).abs() < 1e-5);
        assert_eq!(s.eviction_policy, "sliding");
    }

    #[test]
    fn apply_kv_evict_h2o_reduces_kv_occupancy() {
        let mut s = EngineState_::new(0.8, "opencl".into());
        let cmd = EngineCommand::KvEvictH2o { keep_ratio: 0.6 };
        let result = s.apply(&cmd);
        assert!(matches!(result, CommandResult::Ok));
        // 0.8 * 0.6 = 0.48
        assert!((s.kv_occupancy - 0.48).abs() < 1e-5);
        assert_eq!(s.eviction_policy, "h2o");
    }

    #[test]
    fn apply_kv_evict_clamps_to_minimum() {
        let mut s = EngineState_::new(0.01, "cpu".into());
        let cmd = EngineCommand::KvEvictSliding { keep_ratio: 0.0 };
        s.apply(&cmd);
        // Should be clamped to 0.01
        assert!(s.kv_occupancy >= 0.01);
    }

    #[test]
    fn apply_switch_hw_changes_device() {
        let mut s = EngineState_::new(0.5, "opencl".into());
        let cmd = EngineCommand::SwitchHw {
            device: "cpu".into(),
        };
        let result = s.apply(&cmd);
        assert!(matches!(result, CommandResult::Ok));
        assert_eq!(s.active_device, "cpu");
    }

    #[test]
    fn apply_throttle_sets_delay() {
        let mut s = EngineState_::new(0.5, "cpu".into());
        let cmd = EngineCommand::Throttle { delay_ms: 50 };
        let result = s.apply(&cmd);
        assert!(matches!(result, CommandResult::Ok));
        assert_eq!(s.throttle_delay_ms, 50);
    }

    #[test]
    fn apply_layer_skip_sets_ratio() {
        let mut s = EngineState_::new(0.5, "cpu".into());
        let cmd = EngineCommand::LayerSkip { skip_ratio: 0.25 };
        let result = s.apply(&cmd);
        assert!(matches!(result, CommandResult::Ok));
        assert!((s.skip_ratio - 0.25).abs() < 1e-5);
    }

    #[test]
    fn apply_restore_defaults_resets_state() {
        let mut s = EngineState_::new(0.5, "cpu".into());
        s.throttle_delay_ms = 50;
        s.skip_ratio = 0.25;
        s.eviction_policy = "h2o".to_string();

        let result = s.apply(&EngineCommand::RestoreDefaults);
        assert!(matches!(result, CommandResult::Ok));
        assert_eq!(s.throttle_delay_ms, 0);
        assert!((s.skip_ratio).abs() < 1e-5);
        assert_eq!(s.eviction_policy, "none");
    }

    #[test]
    fn apply_suspend_changes_state() {
        let mut s = EngineState_::new(0.5, "cpu".into());
        s.apply(&EngineCommand::Suspend);
        assert_eq!(s.state, EngineState::Suspended);

        s.apply(&EngineCommand::Resume);
        assert_eq!(s.state, EngineState::Running);
    }

    #[test]
    fn apply_prepare_compute_unit_is_noop() {
        let mut s = EngineState_::new(0.5, "cpu".into());
        let cmd = EngineCommand::PrepareComputeUnit {
            device: "opencl".into(),
        };
        let result = s.apply(&cmd);
        assert!(matches!(result, CommandResult::Ok));
        // device should NOT change — only prepare
        assert_eq!(s.active_device, "cpu");
    }

    #[test]
    fn status_reflects_current_state() {
        let s = EngineState_::new(0.6, "opencl".into());
        let status = s.status();
        assert_eq!(status.active_device, "opencl");
        assert!((status.kv_cache_utilization - 0.6).abs() < 1e-5);
        // kv_cache_tokens should equal floor(0.6 * 2048) = 1228
        assert_eq!(status.kv_cache_tokens, (0.6 * 2048.0_f32) as usize);
        assert_eq!(status.kv_cache_bytes, status.kv_cache_tokens as u64 * 256);
        // available_actions should contain all supported actions
        assert_eq!(status.available_actions.len(), ALL_AVAILABLE_ACTIONS.len());
        assert!(status.active_actions.is_empty());
    }

    #[test]
    fn apply_tracks_active_actions() {
        let mut s = EngineState_::new(0.5, "cpu".into());
        assert!(s.active_actions.is_empty());

        s.apply(&EngineCommand::KvEvictSliding { keep_ratio: 0.8 });
        assert!(s.active_actions.contains(&"kv.evict_sliding".to_string()));

        s.apply(&EngineCommand::Throttle { delay_ms: 50 });
        assert!(s.active_actions.contains(&"throttle".to_string()));
        assert_eq!(s.active_actions.len(), 2);

        // Duplicate action should not add twice
        s.apply(&EngineCommand::Throttle { delay_ms: 100 });
        assert_eq!(
            s.active_actions.iter().filter(|a| *a == "throttle").count(),
            1
        );
    }

    #[test]
    fn restore_defaults_clears_active_actions() {
        let mut s = EngineState_::new(0.5, "cpu".into());
        s.apply(&EngineCommand::KvEvictH2o { keep_ratio: 0.5 });
        s.apply(&EngineCommand::LayerSkip { skip_ratio: 0.3 });
        assert_eq!(s.active_actions.len(), 2);

        s.apply(&EngineCommand::RestoreDefaults);
        assert!(s.active_actions.is_empty());
    }

    #[test]
    fn all_command_types_track_active_actions() {
        let mut s = EngineState_::new(0.5, "cpu".into());
        s.apply(&EngineCommand::KvEvictSliding { keep_ratio: 0.8 });
        s.apply(&EngineCommand::KvEvictH2o { keep_ratio: 0.7 });
        s.apply(&EngineCommand::KvStreaming {
            sink_size: 4,
            window_size: 256,
        });
        s.apply(&EngineCommand::KvMergeD2o { keep_ratio: 0.75 });
        s.apply(&EngineCommand::KvQuantDynamic { target_bits: 4 });
        s.apply(&EngineCommand::Throttle { delay_ms: 50 });
        s.apply(&EngineCommand::LayerSkip { skip_ratio: 0.3 });
        s.apply(&EngineCommand::SwitchHw {
            device: "opencl".into(),
        });

        assert!(s.active_actions.contains(&"kv.evict_sliding".to_string()));
        assert!(s.active_actions.contains(&"kv.evict_h2o".to_string()));
        assert!(s.active_actions.contains(&"kv.evict_streaming".to_string()));
        assert!(s.active_actions.contains(&"kv.merge_d2o".to_string()));
        assert!(s.active_actions.contains(&"kv.quant_dynamic".to_string()));
        assert!(s.active_actions.contains(&"throttle".to_string()));
        assert!(s.active_actions.contains(&"weight.skip".to_string()));
        assert!(s.active_actions.contains(&"switch_hw".to_string()));
        assert_eq!(s.active_actions.len(), 8);
    }

    #[test]
    fn handle_directive_sends_qcf_estimate_on_request_qcf() {
        use argus_shared::EngineDirective;
        use std::io::Read;

        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        let directive = EngineDirective {
            seq_id: 10,
            commands: vec![EngineCommand::RequestQcf],
        };

        let mut engine = EngineState_::new(0.5, "opencl".into());
        let mut count = 0u32;
        handle_directive(&directive, &mut engine, &mut count, &mut client);

        // First message: Response
        let mut len_buf = [0u8; 4];
        server.read_exact(&mut len_buf).unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut json_buf = vec![0u8; len];
        server.read_exact(&mut json_buf).unwrap();
        let msg: EngineMessage = serde_json::from_slice(&json_buf).unwrap();
        assert!(matches!(msg, EngineMessage::Response(_)));

        // Second message: QcfEstimate
        server.read_exact(&mut len_buf).unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut json_buf = vec![0u8; len];
        server.read_exact(&mut json_buf).unwrap();
        let msg: EngineMessage = serde_json::from_slice(&json_buf).unwrap();
        match msg {
            EngineMessage::QcfEstimate(qcf) => {
                assert!(!qcf.estimates.is_empty());
                assert!(qcf.estimates.contains_key("kv.evict_h2o"));
                assert!(qcf.estimates.contains_key("kv.evict_sliding"));
            }
            _ => panic!("Expected QcfEstimate"),
        }
    }

    // ── handle_directive ─────────────────────────────────────────────────────

    #[test]
    fn handle_directive_sends_response_with_matching_seq_id() {
        use argus_shared::EngineDirective;
        use std::io::Read;

        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        let directive = EngineDirective {
            seq_id: 42,
            commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }],
        };

        let mut engine = EngineState_::new(0.9, "opencl".into());
        let mut count = 0u32;
        handle_directive(&directive, &mut engine, &mut count, &mut client);

        // Read response from server side
        let mut len_buf = [0u8; 4];
        server.read_exact(&mut len_buf).unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut json_buf = vec![0u8; len];
        server.read_exact(&mut json_buf).unwrap();

        let msg: EngineMessage = serde_json::from_slice(&json_buf).unwrap();
        match msg {
            EngineMessage::Response(resp) => {
                assert_eq!(resp.seq_id, 42);
                assert_eq!(resp.results.len(), 1);
                assert!(matches!(resp.results[0], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }

        assert_eq!(count, 1);
        // kv_occupancy should have dropped: 0.9 * 0.5 = 0.45
        assert!((engine.kv_occupancy - 0.45).abs() < 1e-5);
    }

    #[test]
    fn handle_directive_increments_count() {
        use argus_shared::EngineDirective;

        let (_dir, sock_path) = tmp_sock();
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (_server, _) = listener.accept().unwrap();

        let directive = EngineDirective {
            seq_id: 1,
            commands: vec![],
        };

        let mut engine = EngineState_::new(0.5, "cpu".into());
        let mut count = 0u32;
        handle_directive(&directive, &mut engine, &mut count, &mut client);
        handle_directive(&directive, &mut engine, &mut count, &mut client);

        assert_eq!(count, 2);
    }
}
