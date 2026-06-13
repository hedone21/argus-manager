use super::Monitor;
use crate::config::ExternalMonitorConfig;
use argus_shared::SystemSignal;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::os::unix::net::UnixListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// Signal injection monitor for research and testing.
///
/// Reads SystemSignal JSON from an external source (Unix socket or stdin)
/// and forwards them directly to the emitter — no threshold evaluation.
pub struct ExternalMonitor {
    transport: ExternalTransport,
}

pub enum ExternalTransport {
    /// Listen on a Unix socket for incoming signals.
    /// Wire format: one JSON object per line (JSON Lines).
    UnixSocket(String),
    /// Listen on a TCP socket for incoming signals.
    /// Wire format: one JSON object per line (JSON Lines).
    Tcp(String),
    /// Read from stdin (for pipe mode).
    Stdin,
}

impl ExternalMonitor {
    pub fn new(config: &ExternalMonitorConfig) -> Self {
        let transport = if config.transport == "stdin" || config.transport == "-" {
            ExternalTransport::Stdin
        } else if let Some(path) = config.transport.strip_prefix("unix:") {
            ExternalTransport::UnixSocket(path.to_string())
        } else if let Some(addr) = config.transport.strip_prefix("tcp:") {
            ExternalTransport::Tcp(addr.to_string())
        } else {
            log::warn!(
                "[ExternalMonitor] Unknown transport '{}', defaulting to stdin",
                config.transport
            );
            ExternalTransport::Stdin
        };

        Self { transport }
    }

    fn run_with_reader(
        &self,
        reader: impl BufRead,
        tx: &mpsc::Sender<SystemSignal>,
        shutdown: &Arc<AtomicBool>,
    ) {
        for line in reader.lines() {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match line {
                Ok(text) => {
                    let text = text.trim().to_string();
                    if text.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<SystemSignal>(&text) {
                        Ok(signal) => {
                            log::info!("[ExternalMonitor] Injected: {:?}", signal);
                            if tx.send(signal).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            log::warn!("[ExternalMonitor] Parse error: {} (input: {})", e, text);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("[ExternalMonitor] Read error: {}", e);
                    break;
                }
            }
        }
    }

    fn run_unix_socket(
        &self,
        path: &str,
        tx: &mpsc::Sender<SystemSignal>,
        shutdown: &Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        // Remove stale socket
        let _ = std::fs::remove_file(path);

        let listener = UnixListener::bind(path)?;
        listener.set_nonblocking(true)?;
        log::info!("[ExternalMonitor] Listening on {}", path);

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match listener.accept() {
                Ok((stream, _)) => {
                    log::info!("[ExternalMonitor] Client connected");
                    stream.set_nonblocking(false)?;
                    let reader = BufReader::new(stream);
                    self.run_with_reader(reader, tx, shutdown);
                    log::info!("[ExternalMonitor] Client disconnected, waiting for next...");
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => {
                    log::error!("[ExternalMonitor] Accept error: {}", e);
                    break;
                }
            }
        }

        let _ = std::fs::remove_file(path);
        Ok(())
    }

    fn run_tcp_socket(
        &self,
        addr: &str,
        tx: &mpsc::Sender<SystemSignal>,
        shutdown: &Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        log::info!("[ExternalMonitor] TCP listening on {}", addr);

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match listener.accept() {
                Ok((stream, peer)) => {
                    log::info!("[ExternalMonitor] TCP client connected from {}", peer);
                    stream.set_nonblocking(false)?;
                    let reader = BufReader::new(stream);
                    self.run_with_reader(reader, tx, shutdown);
                    log::info!("[ExternalMonitor] TCP client disconnected, waiting for next...");
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => {
                    log::error!("[ExternalMonitor] TCP accept error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }
}

impl Monitor for ExternalMonitor {
    fn run(
        &mut self,
        tx: mpsc::Sender<SystemSignal>,
        shutdown: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        log::info!("[ExternalMonitor] Starting");

        match &self.transport {
            ExternalTransport::Stdin => {
                let stdin = std::io::stdin();
                let reader = BufReader::new(stdin.lock());
                self.run_with_reader(reader, &tx, &shutdown);
            }
            ExternalTransport::UnixSocket(path) => {
                let path = path.clone();
                self.run_unix_socket(&path, &tx, &shutdown)?;
            }
            ExternalTransport::Tcp(addr) => {
                let addr = addr.clone();
                self.run_tcp_socket(&addr, &tx, &shutdown)?;
            }
        }

        log::info!("[ExternalMonitor] Stopped");
        Ok(())
    }

    fn initial_signal(&self) -> Option<SystemSignal> {
        None // External monitor has no initial state
    }

    fn name(&self) -> &str {
        "ExternalMonitor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_shared::Level;
    use std::io::Write;
    use std::os::unix::net::UnixStream;

    #[test]
    fn parse_valid_signal() {
        let json = r#"{"thermal_alert":{"level":"critical","temperature_mc":80000,"throttling_active":true,"throttle_ratio":0.5}}"#;
        let signal: SystemSignal = serde_json::from_str(json).unwrap();
        assert_eq!(signal.level(), Level::Critical);
    }

    #[test]
    fn parse_memory_signal() {
        let json = r#"{"memory_pressure":{"level":"warning","available_bytes":100000000,"total_bytes":1000000000,"reclaim_target_bytes":50000000}}"#;
        let signal: SystemSignal = serde_json::from_str(json).unwrap();
        assert_eq!(signal.level(), Level::Warning);
    }

    #[test]
    fn unix_socket_injection() {
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("test_ext.sock");
        let sock_path_str = sock_path.to_str().unwrap().to_string();

        let config = ExternalMonitorConfig {
            enabled: true,
            transport: format!("unix:{}", sock_path_str),
        };
        let mut monitor = ExternalMonitor::new(&config);

        let (tx, rx) = mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        // Run monitor in background
        let handle = std::thread::spawn(move || {
            let _ = monitor.run(tx, shutdown_clone);
        });

        // Wait for socket to be ready
        std::thread::sleep(Duration::from_millis(200));

        // Connect and send a signal
        let mut client = UnixStream::connect(&sock_path).unwrap();
        let json = r#"{"thermal_alert":{"level":"warning","temperature_mc":65000,"throttling_active":false,"throttle_ratio":1.0}}"#;
        writeln!(client, "{}", json).unwrap();
        client.flush().unwrap();

        // Receive signal
        let signal = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(signal.level(), Level::Warning);

        // Cleanup
        shutdown.store(true, Ordering::Relaxed);
        drop(client);
        let _ = handle.join();
    }

    #[test]
    fn skips_invalid_lines() {
        let input = "not json\n{\"memory_pressure\":{\"level\":\"normal\",\"available_bytes\":500000000,\"total_bytes\":2000000000,\"reclaim_target_bytes\":0}}\n";
        let reader = BufReader::new(input.as_bytes());

        let config = ExternalMonitorConfig {
            enabled: true,
            transport: "stdin".into(),
        };
        let monitor = ExternalMonitor::new(&config);

        let (tx, rx) = mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));

        monitor.run_with_reader(reader, &tx, &shutdown);

        // Should have received 1 valid signal (skipped the invalid one)
        let signal = rx.try_recv().unwrap();
        assert_eq!(signal.level(), Level::Normal);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn initial_signal_is_none() {
        let config = ExternalMonitorConfig {
            enabled: true,
            transport: "stdin".into(),
        };
        let monitor = ExternalMonitor::new(&config);
        assert!(monitor.initial_signal().is_none());
    }
}
