use argus_manager::channel::EngineReceiver;
use argus_manager::channel::TcpChannel;
use argus_manager::channel::unix_socket::UnixSocketChannel;
use argus_manager::config::Config;
use argus_manager::emitter::Emitter;
use argus_manager::monitor::Monitor;
use argus_manager::monitor::compute::{ComputeMonitor, SharedGpuProvider, resolve_backend};
use argus_manager::monitor::energy::EnergyMonitor;
use argus_manager::monitor::external::ExternalMonitor;
use argus_manager::monitor::gpu_provider::build_provider;
use argus_manager::monitor::memory::MemoryMonitor;
use argus_manager::monitor::thermal::ThermalMonitor;
use argus_manager::pipeline::{DirectiveDeduplicator, PolicyStrategy};
use argus_manager::signal::SystemSignal;
use argus_shared::EngineMessage;
use clap::Parser;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// Transport 핸들. unix socket / tcp는 양방향, dbus는 emit-only.
enum TransportHandle {
    /// Unix socket — Emitter + EngineReceiver 겸용.
    Unix(UnixSocketChannel),
    /// TCP socket — Emitter + EngineReceiver 겸용.
    Tcp(TcpChannel),
    /// D-Bus 또는 기타 단방향 emitter.
    EmitterOnly(Box<dyn Emitter>),
}

impl TransportHandle {
    fn emitter(&mut self) -> &mut dyn Emitter {
        match self {
            Self::Unix(ch) => ch,
            Self::Tcp(ch) => ch,
            Self::EmitterOnly(em) => em.as_mut(),
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Unix(ch) => ch.name(),
            Self::Tcp(ch) => ch.name(),
            Self::EmitterOnly(em) => em.name(),
        }
    }

    /// Engine으로부터 메시지를 non-blocking으로 수신한다.
    fn try_recv_engine_message(&mut self) -> Option<EngineMessage> {
        match self {
            Self::Unix(ch) => ch.try_recv().ok().flatten(),
            Self::Tcp(ch) => ch.try_recv().ok().flatten(),
            Self::EmitterOnly(_) => None,
        }
    }

    /// Whether this transport can carry engine messages back.
    ///
    /// An emitter-only transport cannot, and the policy is heartbeat-driven: a KV budget
    /// is a fraction of `kv_cache_budget_bytes`, which only the engine knows. On such a
    /// transport the manager would run forever, log a healthy monitor loop, and never
    /// emit a directive — indistinguishable from an idle system. Startup refuses instead.
    fn is_bidirectional(&self) -> bool {
        !matches!(self, Self::EmitterOnly(_))
    }
}

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// SIGHUP 수신 시 true로 설정 — 메인 루프에서 검사하여 hot-reload를 실행한다.
#[cfg(unix)]
static SIGHUP_RELOAD: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_signal(_: libc::c_int) {
    SHUTDOWN.store(true, Ordering::Relaxed);
}

#[cfg(unix)]
extern "C" fn handle_sighup(_: libc::c_int) {
    SIGHUP_RELOAD.store(true, Ordering::Relaxed);
}

#[derive(Parser)]
#[command(
    about = "LLM Resource Manager — monitors system resources and emits directives to LLM engine"
)]
struct Args {
    /// Path to TOML configuration file.
    #[arg(short, long, default_value = "/etc/llm-manager/config.toml")]
    config: std::path::PathBuf,

    /// Transport: "unix:<socket_path>" or "tcp:<host:port>".
    ///
    /// "dbus" is emit-only and is refused at startup — see `TransportHandle::is_bidirectional`.
    #[arg(short, long, default_value = "unix:/tmp/argus_manager.sock")]
    transport: String,

    /// Timeout in seconds to wait for LLM client (unix socket only).
    #[arg(long, default_value_t = 60)]
    client_timeout: u64,

    /// Path to policy configuration TOML.
    /// When omitted, built-in defaults are used.
    #[arg(long)]
    policy_config: Option<std::path::PathBuf>,

    /// Path to a Lua policy script.
    /// When specified, the Lua script replaces the built-in HierarchicalPolicy.
    /// Requires the `lua` feature to be enabled at compile time.
    #[arg(long)]
    policy_script: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            handle_signal as *const () as libc::sighandler_t,
        );
        #[cfg(unix)]
        libc::signal(
            libc::SIGHUP,
            handle_sighup as *const () as libc::sighandler_t,
        );
    }

    let config = if args.config.exists() {
        log::info!("Loading config from {}", args.config.display());
        Config::from_file(&args.config)?
    } else {
        log::info!("Config not found, using defaults");
        Config::default()
    };

    let default_poll = config.manager.poll_interval_ms;
    log::info!("LLM Manager starting (poll_interval={}ms)", default_poll);

    let shutdown = Arc::new(AtomicBool::new(false));

    // Create transport (unix: 양방향, dbus: 단방향)
    let mut transport = create_transport(&args, &shutdown)?;
    log::info!("Transport: {}", transport.name());
    if !transport.is_bidirectional() {
        anyhow::bail!(
            "transport '{}' cannot receive engine messages, and every policy decision \
             needs the heartbeat (a KV budget is a fraction of kv_cache_budget_bytes, \
             which only the engine reports). Use --transport unix:<path> or tcp:<addr>.",
            transport.name()
        );
    }

    // GPU telemetry provider — ComputeMonitor와 LuaPolicy가 공유한다.
    let gpu_provider: SharedGpuProvider = {
        let compute_cfg = config.compute.clone().unwrap_or_default();
        let backend = resolve_backend(&compute_cfg);
        let provider = build_provider(&backend);
        log::info!(
            "GPU telemetry backend: {} ({:?})",
            provider.describe(),
            backend
        );
        Arc::new(std::sync::Mutex::new(provider))
    };

    // Build monitors
    let monitors = build_monitors(&config, Arc::clone(&gpu_provider));
    log::info!("Monitors: {}", monitors.len());

    // Collect initial state from monitors
    let initial_signals: Vec<SystemSignal> =
        monitors.iter().filter_map(|m| m.initial_signal()).collect();

    // Spawn monitor threads
    let (tx, rx) = mpsc::channel::<SystemSignal>();
    let handles = spawn_monitors(monitors, tx, shutdown.clone());
    log::info!("Started {} monitor threads", handles.len());

    // ── Policy 초기화 ─────────────────────────────────────────────────────────

    let mut policy: Box<dyn PolicyStrategy> =
        create_policy(&args, &config, Arc::clone(&gpu_provider))?;

    // Emit initial state
    for signal in &initial_signals {
        if let Some(directive) = policy.process_signal(signal) {
            log::info!(
                "Initial directive seq={}: {} commands",
                directive.seq_id,
                directive.commands.len()
            );
            transport.emitter().emit_directive(&directive)?;
        }
    }

    // ── Main loop ─────────────────────────────────────────────────────────────
    log::info!("Entering main loop");
    let start = std::time::Instant::now();
    let mut dedup = DirectiveDeduplicator::with_cooldown(config.adaptation.dedup_cooldown_secs);
    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            shutdown.store(true, Ordering::Relaxed);
            break;
        }

        #[cfg(unix)]
        if SIGHUP_RELOAD.swap(false, Ordering::Relaxed) {
            if let Some(reloadable) = policy.as_reloadable() {
                if let Some(path) = reloadable.script_path() {
                    let path = path.to_path_buf();
                    log::info!(
                        "SIGHUP received — reloading policy script: {}",
                        path.display()
                    );
                    if let Err(e) = reloadable.reload_script(&path) {
                        log::error!("Policy hot-reload failed: {}", e);
                    } else {
                        log::info!("Policy hot-reload succeeded");
                    }
                } else {
                    log::warn!("SIGHUP received but policy does not have a script path");
                }
            } else {
                log::warn!("SIGHUP received but policy does not support hot-reload");
            }
        }

        // ── Engine message 수신 (unix transport일 때만 유효) ──────────────
        while let Some(msg) = transport.try_recv_engine_message() {
            match &msg {
                EngineMessage::Heartbeat(status) => {
                    log::debug!(
                        "Engine heartbeat: kv={}/{}B tokens={} tbt={:.1}ms phase={:?}",
                        status.kv_cache_bytes,
                        status.kv_cache_budget_bytes,
                        status.kv_cache_tokens,
                        status.tbt_ms,
                        status.phase,
                    );
                }
                EngineMessage::Response(resp) => {
                    log::debug!(
                        "Engine response seq={}: {} results",
                        resp.seq_id,
                        resp.results.len()
                    );
                }
            }
            // The policy logs Rejected and Partial itself — those are the only signal it
            // gets about what the engine can actually do.
            policy.update_engine_state(&msg);
        }

        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(signal) => {
                log::info!("Signal: {:?}", signal);

                if let Some(directive) = policy.process_signal(&signal) {
                    if let Some(directive) = dedup.process(directive, start.elapsed().as_secs_f64())
                    {
                        log::info!(
                            "Directive seq={}: {} commands [mode={:?}]",
                            directive.seq_id,
                            directive.commands.len(),
                            policy.mode()
                        );
                        if let Err(e) = transport.emitter().emit_directive(&directive) {
                            log::error!("Emit directive failed: {}", e);
                        }
                    } else {
                        policy.cancel_last_observation();
                        log::debug!("Directive suppressed (duplicate), observation cancelled");
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                log::warn!("All monitors disconnected");
                break;
            }
        }
    }

    log::info!("Shutting down...");
    shutdown.store(true, Ordering::Relaxed);

    // Relief model 저장
    policy.save_model();

    for handle in handles {
        let _ = handle.join();
    }
    log::info!("LLM Manager stopped");

    Ok(())
}

/// `--policy-script`가 지정되면 LuaPolicy를, 아니면 HierarchicalPolicy를 생성한다.
fn create_policy(
    args: &Args,
    config: &Config,
    gpu_provider: SharedGpuProvider,
) -> anyhow::Result<Box<dyn PolicyStrategy>> {
    // Lua policy script 지정 시
    if let Some(ref script_path) = args.policy_script {
        return create_lua_policy(script_path, config, gpu_provider);
    }
    let _ = (gpu_provider, config);
    // The Lua script IS the policy — there is no compiled-in alternative to fall back to.
    {
        anyhow::bail!("--policy-script is required: the decision logic lives in the script");
    }

    // 기본: HierarchicalPolicy (hierarchical feature 활성 시)
}

#[cfg(feature = "lua")]
fn create_lua_policy(
    script_path: &std::path::Path,
    config: &Config,
    gpu_provider: SharedGpuProvider,
) -> anyhow::Result<Box<dyn PolicyStrategy>> {
    let path_str = script_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 in policy script path"))?;
    let policy = argus_manager::lua_policy::LuaPolicy::with_system_clock_and_gpu(
        path_str,
        config.adaptation.clone(),
        gpu_provider,
    )?;
    log::info!("LuaPolicy initialized from {}", path_str);
    Ok(Box::new(policy))
}

#[cfg(not(feature = "lua"))]
fn create_lua_policy(
    _script_path: &std::path::Path,
    _config: &Config,
    _gpu_provider: SharedGpuProvider,
) -> anyhow::Result<Box<dyn PolicyStrategy>> {
    anyhow::bail!(
        "--policy-script requires the 'lua' feature (compile with: cargo build --features lua)"
    )
}

fn create_transport(args: &Args, shutdown: &Arc<AtomicBool>) -> anyhow::Result<TransportHandle> {
    if let Some(path) = args.transport.strip_prefix("unix:") {
        let mut channel = UnixSocketChannel::new(std::path::Path::new(path))?;
        log::info!(
            "Waiting for client on {} (timeout={}s)...",
            path,
            args.client_timeout
        );
        if !channel.wait_for_client(Duration::from_secs(args.client_timeout), shutdown) {
            if shutdown.load(Ordering::Relaxed) {
                anyhow::bail!("Shutdown during client wait");
            }
            log::warn!("No client connected within timeout, proceeding anyway");
        }
        Ok(TransportHandle::Unix(channel))
    } else if let Some(addr) = args.transport.strip_prefix("tcp:") {
        let mut channel = TcpChannel::new(addr)?;
        log::info!(
            "TCP transport: waiting for client on {} (timeout={}s)...",
            addr,
            args.client_timeout
        );
        if !channel.wait_for_client(Duration::from_secs(args.client_timeout), shutdown) {
            if shutdown.load(Ordering::Relaxed) {
                anyhow::bail!("Shutdown during client wait");
            }
            anyhow::bail!("TCP: no client connected within {}s", args.client_timeout);
        }
        Ok(TransportHandle::Tcp(channel))
    } else if args.transport == "dbus" {
        Ok(TransportHandle::EmitterOnly(create_dbus_emitter()?))
    } else {
        anyhow::bail!(
            "Unknown transport: {}. Use 'dbus', 'unix:<path>', or 'tcp:<host:port>'",
            args.transport
        );
    }
}

#[cfg(feature = "dbus")]
fn create_dbus_emitter() -> anyhow::Result<Box<dyn Emitter>> {
    let emitter = argus_manager::emitter::dbus::DbusEmitter::new()?;
    Ok(Box::new(emitter))
}

#[cfg(not(feature = "dbus"))]
fn create_dbus_emitter() -> anyhow::Result<Box<dyn Emitter>> {
    anyhow::bail!("Transport 'dbus' requires the 'dbus' feature (compiled without it)")
}

fn build_monitors(config: &Config, gpu_provider: SharedGpuProvider) -> Vec<Box<dyn Monitor>> {
    let default_poll = config.manager.poll_interval_ms;
    let mut monitors: Vec<Box<dyn Monitor>> = Vec::new();

    if config.memory.as_ref().is_none_or(|c| c.enabled) {
        let c = config.memory.clone().unwrap_or_default();
        monitors.push(Box::new(MemoryMonitor::new(&c, default_poll)));
    }

    if config.thermal.as_ref().is_none_or(|c| c.enabled) {
        let c = config.thermal.clone().unwrap_or_default();
        monitors.push(Box::new(ThermalMonitor::new(&c, default_poll)));
    }

    if config.compute.as_ref().is_none_or(|c| c.enabled) {
        let c = config.compute.clone().unwrap_or_default();
        monitors.push(Box::new(ComputeMonitor::new(
            &c,
            default_poll,
            gpu_provider.clone(),
        )));
    }

    if config.energy.as_ref().is_none_or(|c| c.enabled) {
        let c = config.energy.clone().unwrap_or_default();
        monitors.push(Box::new(EnergyMonitor::new(&c, default_poll)));
    }

    if config.external.as_ref().is_some_and(|c| c.enabled) {
        let c = config.external.as_ref().unwrap();
        monitors.push(Box::new(ExternalMonitor::new(c)));
    }

    monitors
}

fn spawn_monitors(
    monitors: Vec<Box<dyn Monitor>>,
    tx: mpsc::Sender<SystemSignal>,
    shutdown: Arc<AtomicBool>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut handles = Vec::new();

    for mut monitor in monitors {
        let tx = tx.clone();
        let shutdown = shutdown.clone();
        let name = monitor.name().to_string();

        let handle = std::thread::Builder::new()
            .name(name.clone())
            .spawn(move || {
                if let Err(e) = monitor.run(tx, shutdown) {
                    log::error!("[{}] Monitor error: {}", name, e);
                }
            })
            .expect("Failed to spawn monitor thread");

        handles.push(handle);
    }

    drop(tx);
    handles
}
