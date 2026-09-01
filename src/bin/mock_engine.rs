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

use std::io::{Read, Write};
use std::net::TcpStream;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use anyhow::Context;
use argus_shared::{
    CommandResponse, CommandResult, EngineCommand, EngineDirective, EngineMessage, EngineState,
    EngineStatus, ManagerMessage, Phase,
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
}

/// Smallest occupancy delta the mock will act on — the real engine has an equivalent
/// token floor (`MIN_EVICT_TOKENS`).
const MIN_COMPRESSION: f32 = 0.03;

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
        }
    }

    /// Apply a single `EngineCommand` and return a human-readable description
    /// of what changed.
    fn apply(&mut self, cmd: &EngineCommand) -> CommandResult {
        match cmd {
            EngineCommand::KvCompress { budget } => {
                // Mirrors the real engine's floor: it declines rather than shave off a
                // handful of tokens, and says so with `Partial` instead of a bare `Ok`.
                let before = self.kv_occupancy;
                let target = (before * budget).clamp(0.01, 1.0);
                if before - target < MIN_COMPRESSION {
                    return CommandResult::Partial {
                        achieved: 1.0,
                        reason: "compression declined: too little would be reclaimed".to_string(),
                    };
                }
                self.kv_occupancy = target;
                CommandResult::Ok
            }
            EngineCommand::RestoreDefaults => {
                self.active_actions.clear();
                self.skip_ratio = 0.0;
                self.state = EngineState::Running;
                CommandResult::Ok
            }
            EngineCommand::Suspend => {
                self.state = EngineState::Suspended;
                CommandResult::Ok
            }
            EngineCommand::Resume => {
                self.state = EngineState::Running;
                CommandResult::Ok
            }
        }
    }

    /// Build the current `EngineStatus` heartbeat from internal state.
    fn status(&self) -> EngineStatus {
        EngineStatus {
            kv_cache_bytes: 1024,
            kv_cache_budget_bytes: 4096,
            kv_cache_tokens: 32,
            tbt_ms: 12.5,
            phase: Phase::Decode,
            state: EngineState::Running,
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
    // No capability report: the contract has none. What this engine can do is answered
    // per command, as `Rejected`.

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
    /// Length-prefixed framing round-trip for the message an engine actually sends first.
    #[test]
    fn heartbeat_round_trips_over_the_socket() {
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("mock_engine_frame.sock");
        let listener = UnixListener::bind(&sock_path).unwrap();
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        let hb = EngineMessage::Heartbeat(EngineStatus {
            kv_cache_bytes: 1024,
            kv_cache_budget_bytes: 4096,
            kv_cache_tokens: 32,
            tbt_ms: 12.5,
            phase: Phase::Decode,
            state: EngineState::Running,
        });
        send_message(&mut client, &hb).unwrap();

        let mut len_buf = [0u8; 4];
        server.read_exact(&mut len_buf).unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut body = vec![0u8; len];
        server.read_exact(&mut body).unwrap();
        match serde_json::from_slice::<EngineMessage>(&body).unwrap() {
            EngineMessage::Heartbeat(s) => {
                assert_eq!(s.kv_cache_bytes, 1024);
                assert_eq!(s.kv_cache_budget_bytes, 4096);
            }
            other => panic!("expected heartbeat, got {other:?}"),
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
            commands: vec![EngineCommand::KvCompress { budget: 0.5 }],
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
    fn apply_kv_evict_clamps_to_minimum() {
        let mut s = EngineState_::new(0.01, "cpu".into());
        let cmd = EngineCommand::KvCompress { budget: 0.5 };
        s.apply(&cmd);
        // Should be clamped to 0.01
        assert!(s.kv_occupancy >= 0.01);
    }

    /// A compression below the floor is answered `Partial`, not `Ok` — the mock mirrors
    /// the real engine, which declines rather than shave off a handful of tokens.
    #[test]
    fn tiny_compression_is_declined_as_partial() {
        let mut s = EngineState_::new(0.8, "opencl".into());
        let before = s.kv_occupancy;
        let result = s.apply(&EngineCommand::KvCompress { budget: 0.99 });
        assert!(
            matches!(result, CommandResult::Partial { .. }),
            "got {result:?}"
        );
        assert_eq!(s.kv_occupancy, before, "declined means untouched");
    }

    #[test]
    fn compression_shrinks_the_cache() {
        let mut s = EngineState_::new(0.8, "opencl".into());
        assert!(matches!(
            s.apply(&EngineCommand::KvCompress { budget: 0.5 }),
            CommandResult::Ok
        ));
        assert!((s.kv_occupancy - 0.4).abs() < 1e-6, "{}", s.kv_occupancy);
    }

    #[test]
    fn apply_suspend_changes_state() {
        let mut s = EngineState_::new(0.5, "cpu".into());
        s.apply(&EngineCommand::Suspend);
        assert_eq!(s.state, EngineState::Suspended);

        s.apply(&EngineCommand::Resume);
        assert_eq!(s.state, EngineState::Running);
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
            commands: vec![EngineCommand::KvCompress { budget: 0.5 }],
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
