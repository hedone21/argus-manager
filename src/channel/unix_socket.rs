use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use argus_shared::{EngineDirective, EngineMessage, ManagerMessage, SystemSignal};

use crate::channel::EngineReceiver;
use crate::emitter::Emitter;

// ── RAII 핸들 ──────────────────────────────────────────────────────────────

/// Reader thread를 RAII로 관리한다.
///
/// Drop 시 handle을 join하지 않는다. reader thread는 stream EOF 시 자연 종료한다.
struct ReaderHandle {
    handle: Option<JoinHandle<()>>,
}

impl Drop for ReaderHandle {
    fn drop(&mut self) {
        // handle은 의도적으로 join하지 않는다 (main loop 지연 방지).
        // reader thread는 stream이 닫힐 때 EOF로 자연 종료한다.
        let _ = self.handle.take();
    }
}

// ── 연결 상태 ──────────────────────────────────────────────────────────────

/// 연결 상태를 명시적으로 모델링한다.
///
/// Writer 에러와 Reader EOF 모두 Disconnected로 수렴하여
/// 비대칭 연결 끊김 처리를 단순화한다.
enum ConnectionState {
    /// 소켓을 bind했지만 클라이언트 미연결.
    Listening,
    /// 클라이언트 연결됨.
    Connected {
        writer: UnixStream,
        /// reader thread가 파싱한 메시지를 push하는 채널의 수신단.
        inbox: mpsc::Receiver<EngineMessage>,
        /// reader thread RAII 핸들.
        _reader: ReaderHandle,
    },
    /// 연결이 끊어짐 — 다음 emit() 호출 시 재연결 시도.
    Disconnected,
}

// ── UnixSocketChannel ─────────────────────────────────────────────────────

/// 양방향 Unix socket 채널.
///
/// Manager → Engine (write) 방향과 Engine → Manager (read) 방향을
/// 하나의 struct에서 관리한다.
///
/// # 상태 전이
///
/// ```text
/// Listening    ──(accept)───────────→ Connected
/// Connected    ──(write err)─────────→ Disconnected
/// Connected    ──(inbox Disconnected)→ Disconnected
/// Disconnected ──(try_accept)────────→ Connected  (다음 emit() 호출 시)
/// ```
pub struct UnixSocketChannel {
    socket_path: PathBuf,
    /// listener는 항상 보유 (재연결에 재사용).
    listener: UnixListener,
    state: ConnectionState,
}

impl UnixSocketChannel {
    /// 주어진 경로에서 listen하는 채널을 생성한다.
    pub fn new(socket_path: &Path) -> anyhow::Result<Self> {
        // 기존 소켓 파일 제거
        let _ = std::fs::remove_file(socket_path);

        let listener = UnixListener::bind(socket_path)?;
        listener.set_nonblocking(true)?;
        log::info!("[UnixSocketChannel] Listening on {}", socket_path.display());

        Ok(Self {
            socket_path: socket_path.to_path_buf(),
            listener,
            state: ConnectionState::Listening,
        })
    }

    /// 클라이언트가 연결될 때까지 대기한다 (blocking with timeout).
    ///
    /// 연결에 성공하면 `true`, 타임아웃 또는 shutdown이면 `false`를 반환한다.
    pub fn wait_for_client(&mut self, timeout: Duration, shutdown: &Arc<AtomicBool>) -> bool {
        let start = std::time::Instant::now();
        loop {
            if shutdown.load(Ordering::Relaxed) {
                return false;
            }
            match self.listener.accept() {
                Ok((stream, _)) => {
                    log::info!("[UnixSocketChannel] Client connected");
                    self.transition_to_connected(stream);
                    return true;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if start.elapsed() >= timeout {
                        return false;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => {
                    log::error!("[UnixSocketChannel] Accept error: {}", e);
                    return false;
                }
            }
        }
    }

    /// Disconnected 상태에서 non-blocking으로 재연결을 시도한다.
    fn ensure_connected(&mut self) {
        if !matches!(self.state, ConnectionState::Disconnected) {
            return;
        }
        match self.listener.accept() {
            Ok((stream, _)) => {
                log::info!("[UnixSocketChannel] Client reconnected");
                self.transition_to_connected(stream);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // 연결 대기 중 — 정상
            }
            Err(e) => {
                log::debug!("[UnixSocketChannel] try_accept error: {}", e);
            }
        }
    }

    /// 연결된 stream으로 Connected 상태로 전이한다.
    fn transition_to_connected(&mut self, stream: UnixStream) {
        stream.set_nonblocking(false).ok();

        let writer = match stream.try_clone() {
            Ok(w) => w,
            Err(e) => {
                log::error!("[UnixSocketChannel] try_clone failed: {}", e);
                self.state = ConnectionState::Disconnected;
                return;
            }
        };

        let (inbox_tx, inbox_rx) = mpsc::sync_channel(64);
        let reader_handle = spawn_reader(stream, inbox_tx);

        self.state = ConnectionState::Connected {
            writer,
            inbox: inbox_rx,
            _reader: reader_handle,
        };
    }

    /// writer에 `ManagerMessage`를 length-prefixed JSON으로 전송한다.
    fn write_manager_message(&mut self, msg: &ManagerMessage) -> anyhow::Result<()> {
        if let ConnectionState::Connected { writer, .. } = &mut self.state {
            let json = serde_json::to_vec(msg)?;
            let len = (json.len() as u32).to_be_bytes();
            writer.write_all(&len)?;
            writer.write_all(&json)?;
            writer.flush()?;
            Ok(())
        } else {
            anyhow::bail!("No client connected")
        }
    }

    /// writer에 `SystemSignal`을 length-prefixed JSON으로 전송한다.
    fn write_signal(&mut self, signal: &SystemSignal) -> anyhow::Result<()> {
        if let ConnectionState::Connected { writer, .. } = &mut self.state {
            let json = serde_json::to_vec(signal)?;
            let len = (json.len() as u32).to_be_bytes();
            writer.write_all(&len)?;
            writer.write_all(&json)?;
            writer.flush()?;
            Ok(())
        } else {
            anyhow::bail!("No client connected")
        }
    }
}

impl Drop for UnixSocketChannel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

// ── Emitter 구현 ──────────────────────────────────────────────────────────

impl Emitter for UnixSocketChannel {
    fn emit(&mut self, signal: &SystemSignal) -> anyhow::Result<()> {
        self.ensure_connected();
        if !matches!(self.state, ConnectionState::Connected { .. }) {
            return Ok(()); // 클라이언트 없음 — 치명적이지 않음
        }
        if let Err(e) = self.write_signal(signal) {
            log::warn!("[UnixSocketChannel] Write error (signal): {}", e);
            self.state = ConnectionState::Disconnected;
        }
        Ok(())
    }

    fn emit_initial(&mut self, signals: &[SystemSignal]) -> anyhow::Result<()> {
        for signal in signals {
            self.emit(signal)?;
        }
        Ok(())
    }

    fn emit_directive(&mut self, directive: &EngineDirective) -> anyhow::Result<()> {
        self.ensure_connected();
        if !matches!(self.state, ConnectionState::Connected { .. }) {
            return Ok(()); // 클라이언트 없음 — 치명적이지 않음
        }
        let msg = ManagerMessage::Directive(directive.clone());
        if let Err(e) = self.write_manager_message(&msg) {
            log::warn!(
                "[UnixSocketChannel] Write error (directive seq={}): {}",
                directive.seq_id,
                e
            );
            self.state = ConnectionState::Disconnected;
        } else {
            log::debug!(
                "[UnixSocketChannel] Sent directive seq={}",
                directive.seq_id
            );
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "UnixSocketChannel"
    }
}

// ── EngineReceiver 구현 ───────────────────────────────────────────────────

impl EngineReceiver for UnixSocketChannel {
    fn try_recv(&mut self) -> anyhow::Result<Option<EngineMessage>> {
        if let ConnectionState::Connected { inbox, .. } = &self.state {
            match inbox.try_recv() {
                Ok(msg) => return Ok(Some(msg)),
                Err(mpsc::TryRecvError::Empty) => return Ok(None),
                Err(mpsc::TryRecvError::Disconnected) => {
                    // reader thread 종료 감지 → 상태 전이 (아래에서 처리)
                }
            }
        } else {
            return Ok(None);
        }
        // Connected에서 Disconnected 감지된 경우
        log::info!("[UnixSocketChannel] Reader thread exited — connection lost");
        self.state = ConnectionState::Disconnected;
        Ok(None)
    }

    fn is_connected(&self) -> bool {
        matches!(self.state, ConnectionState::Connected { .. })
    }
}

// ── Reader thread ─────────────────────────────────────────────────────────

/// `stream`에서 `EngineMessage`를 blocking read하여 `inbox_tx`로 전달하는 스레드를 spawn한다.
///
/// EOF 또는 I/O 에러 발생 시 루프를 종료하고, `inbox_tx`가 drop되어
/// `Receiver::try_recv()`에서 `Disconnected`가 반환된다.
fn spawn_reader(mut stream: UnixStream, inbox_tx: mpsc::SyncSender<EngineMessage>) -> ReaderHandle {
    let handle = thread::Builder::new()
        .name("engine-reader".into())
        .spawn(move || {
            loop {
                match read_engine_message(&mut stream) {
                    Ok(msg) => {
                        if inbox_tx.send(msg).is_err() {
                            // receiver dropped (UnixSocketChannel이 drop됨)
                            break;
                        }
                    }
                    Err(e) => {
                        log::debug!("[engine-reader] Read error: {} — exiting", e);
                        break; // EOF 또는 I/O 에러
                    }
                }
            }
            // inbox_tx drop → Receiver::try_recv()가 Disconnected 반환
        })
        .expect("spawn engine-reader");

    ReaderHandle {
        handle: Some(handle),
    }
}

/// PROTO-012: spec-defined 64KB cap on a single framed payload. Reject
/// oversize frames before allocating to keep faulty/malicious peers from
/// triggering large heap reservations.
const MAX_PAYLOAD_SIZE: u32 = 64 * 1024;

/// stream에서 length-prefixed JSON으로 `EngineMessage`를 하나 읽는다.
///
/// Wire format: `[4-byte BE u32 length][UTF-8 JSON]`
fn read_engine_message(stream: &mut UnixStream) -> anyhow::Result<EngineMessage> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let payload_len = u32::from_be_bytes(len_buf);
    if payload_len > MAX_PAYLOAD_SIZE {
        anyhow::bail!(
            "payload length {} exceeds MAX_PAYLOAD_SIZE {} (PROTO-012)",
            payload_len,
            MAX_PAYLOAD_SIZE
        );
    }

    let mut json_buf = vec![0u8; payload_len as usize];
    stream.read_exact(&mut json_buf)?;

    let msg: EngineMessage = serde_json::from_slice(&json_buf)?;
    Ok(msg)
}

// ── 단위 테스트 ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use argus_shared::{EngineCapability, EngineState, EngineStatus, Level, ResourceLevel};
    use std::io::Write as IoWrite;
    use std::os::unix::net::UnixStream as StdUnixStream;

    fn tmp_sock() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("channel_test.sock");
        (dir, path)
    }

    fn make_heartbeat(kv: f32, device: &str) -> EngineMessage {
        EngineMessage::Heartbeat(EngineStatus {
            active_device: device.to_string(),
            compute_level: ResourceLevel::Normal,
            actual_throughput: 15.0,
            memory_level: ResourceLevel::Normal,
            kv_cache_bytes: 0,
            kv_cache_tokens: 0,
            kv_cache_utilization: kv,
            memory_lossless_min: 1.0,
            memory_lossy_min: 0.01,
            state: EngineState::Running,
            tokens_generated: 0,
            available_actions: vec![],
            active_actions: vec![],
            eviction_policy: "none".to_string(),
            kv_dtype: "f16".to_string(),
            skip_ratio: 0.0,
            phase: String::new(),
            prefill_pos: 0,
            prefill_total: 0,
            partition_ratio: 0.0,
            self_cpu_pct: 0.0,
            self_gpu_pct: 0.0,
        })
    }

    fn make_capability() -> EngineMessage {
        EngineMessage::Capability(EngineCapability {
            available_devices: vec!["cpu".into(), "opencl".into()],
            active_device: "opencl".into(),
            max_kv_tokens: 2048,
            bytes_per_kv_token: 256,
            num_layers: 16,
            ..Default::default()
        })
    }

    /// 헬퍼: client stream에서 length-prefixed JSON `EngineMessage`를 전송.
    fn send_engine_msg(stream: &mut StdUnixStream, msg: &EngineMessage) {
        let json = serde_json::to_vec(msg).unwrap();
        let len = (json.len() as u32).to_be_bytes();
        stream.write_all(&len).unwrap();
        stream.write_all(&json).unwrap();
        stream.flush().unwrap();
    }

    /// 헬퍼: channel의 try_recv를 최대 N번 재시도 (reader thread가 처리할 시간 확보).
    fn try_recv_with_retry(ch: &mut UnixSocketChannel, retries: usize) -> Option<EngineMessage> {
        for _ in 0..retries {
            match ch.try_recv().unwrap() {
                Some(msg) => return Some(msg),
                None => std::thread::sleep(Duration::from_millis(10)),
            }
        }
        None
    }

    // ── 정상 수신 ─────────────────────────────────────────────────────────

    #[test]
    fn try_recv_returns_heartbeat_from_engine() {
        let (_dir, sock_path) = tmp_sock();
        let mut channel = UnixSocketChannel::new(&sock_path).unwrap();

        // 클라이언트 연결
        let mut client = StdUnixStream::connect(&sock_path).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        assert!(channel.wait_for_client(Duration::from_secs(1), &shutdown));
        assert!(channel.is_connected());

        // Engine → Manager: Heartbeat 전송
        send_engine_msg(&mut client, &make_heartbeat(0.75, "opencl"));

        // Manager가 수신
        let msg = try_recv_with_retry(&mut channel, 50).expect("Expected heartbeat");
        match msg {
            EngineMessage::Heartbeat(s) => {
                assert!((s.kv_cache_utilization - 0.75).abs() < 1e-5);
                assert_eq!(s.active_device, "opencl");
            }
            _ => panic!("Expected Heartbeat"),
        }
    }

    // ── 메시지 없음 (non-blocking) ─────────────────────────────────────────

    #[test]
    fn try_recv_returns_none_when_no_message() {
        let (_dir, sock_path) = tmp_sock();
        let mut channel = UnixSocketChannel::new(&sock_path).unwrap();

        let _client = StdUnixStream::connect(&sock_path).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        channel.wait_for_client(Duration::from_secs(1), &shutdown);

        // 메시지 없음 — None 반환
        let result = channel.try_recv().unwrap();
        assert!(result.is_none());
    }

    // ── Reader thread EOF → Disconnected 전이 ─────────────────────────────

    #[test]
    fn disconnected_after_reader_eof() {
        let (_dir, sock_path) = tmp_sock();
        let mut channel = UnixSocketChannel::new(&sock_path).unwrap();

        let client = StdUnixStream::connect(&sock_path).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        channel.wait_for_client(Duration::from_secs(1), &shutdown);
        assert!(channel.is_connected());

        // 클라이언트 연결 닫기 → reader thread EOF
        drop(client);

        // reader thread가 종료될 때까지 대기
        std::thread::sleep(Duration::from_millis(100));

        // try_recv 호출로 Disconnected 감지
        channel.try_recv().unwrap();
        assert!(!channel.is_connected());
    }

    // ── Writer 에러 → Disconnected 전이 ──────────────────────────────────

    #[test]
    fn disconnected_after_write_error() {
        let (_dir, sock_path) = tmp_sock();
        let mut channel = UnixSocketChannel::new(&sock_path).unwrap();

        let client = StdUnixStream::connect(&sock_path).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        channel.wait_for_client(Duration::from_secs(1), &shutdown);
        assert!(channel.is_connected());

        // 클라이언트 연결 닫기
        drop(client);
        // 소켓이 닫힌 후 write 시도 → 에러 → Disconnected
        std::thread::sleep(Duration::from_millis(50));

        let signal = SystemSignal::MemoryPressure {
            level: Level::Warning,
            available_bytes: 500_000_000,
            total_bytes: 2_000_000_000,
            reclaim_target_bytes: 50_000_000,
        };
        // emit은 에러를 전파하지 않고 Disconnected로 전이한다
        channel.emit(&signal).unwrap();
        assert!(!channel.is_connected());
    }

    // ── 재연결 ────────────────────────────────────────────────────────────

    #[test]
    fn reconnect_after_disconnect() {
        let (_dir, sock_path) = tmp_sock();
        let mut channel = UnixSocketChannel::new(&sock_path).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));

        // 1차 연결
        let client1 = StdUnixStream::connect(&sock_path).unwrap();
        channel.wait_for_client(Duration::from_secs(1), &shutdown);
        assert!(channel.is_connected());

        // 1차 연결 끊김
        drop(client1);
        std::thread::sleep(Duration::from_millis(100));
        channel.try_recv().unwrap(); // Disconnected 감지
        assert!(!channel.is_connected());

        // 2차 연결
        let mut client2 = StdUnixStream::connect(&sock_path).unwrap();
        // emit 호출 시 내부적으로 try_accept → Connected 전이
        let signal = SystemSignal::MemoryPressure {
            level: Level::Normal,
            available_bytes: 1_000_000_000,
            total_bytes: 2_000_000_000,
            reclaim_target_bytes: 0,
        };
        channel.emit(&signal).unwrap();
        assert!(channel.is_connected());

        // 2차 연결에서 메시지 수신
        send_engine_msg(&mut client2, &make_heartbeat(0.3, "cpu"));
        let msg =
            try_recv_with_retry(&mut channel, 50).expect("Expected heartbeat after reconnect");
        assert!(matches!(msg, EngineMessage::Heartbeat(_)));
    }

    // ── Capability 수신 ───────────────────────────────────────────────────

    #[test]
    fn try_recv_returns_capability_message() {
        let (_dir, sock_path) = tmp_sock();
        let mut channel = UnixSocketChannel::new(&sock_path).unwrap();

        let mut client = StdUnixStream::connect(&sock_path).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        channel.wait_for_client(Duration::from_secs(1), &shutdown);

        send_engine_msg(&mut client, &make_capability());

        let msg = try_recv_with_retry(&mut channel, 50).expect("Expected capability");
        match msg {
            EngineMessage::Capability(c) => {
                assert_eq!(c.active_device, "opencl");
                assert_eq!(c.num_layers, 16);
            }
            _ => panic!("Expected Capability"),
        }
    }

    // ── is_connected: 초기 상태 ───────────────────────────────────────────

    #[test]
    fn is_connected_false_before_client_connects() {
        let (_dir, sock_path) = tmp_sock();
        let channel = UnixSocketChannel::new(&sock_path).unwrap();
        assert!(!channel.is_connected());
    }

    // ── emit_directive 정상 전송 ──────────────────────────────────────────

    #[test]
    fn emit_directive_sends_manager_message() {
        use argus_shared::{EngineCommand, EngineDirective, ManagerMessage};
        use std::io::Read as IoRead;

        let (_dir, sock_path) = tmp_sock();
        let mut channel = UnixSocketChannel::new(&sock_path).unwrap();

        let mut client = StdUnixStream::connect(&sock_path).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        channel.wait_for_client(Duration::from_secs(1), &shutdown);

        let directive = EngineDirective {
            seq_id: 99,
            commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }],
        };
        channel.emit_directive(&directive).unwrap();

        // 클라이언트 쪽에서 수신
        client
            .set_read_timeout(Some(Duration::from_millis(500)))
            .unwrap();
        let mut len_buf = [0u8; 4];
        client.read_exact(&mut len_buf).unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut json_buf = vec![0u8; len];
        client.read_exact(&mut json_buf).unwrap();

        let msg: ManagerMessage = serde_json::from_slice(&json_buf).unwrap();
        match msg {
            ManagerMessage::Directive(d) => {
                assert_eq!(d.seq_id, 99);
                assert_eq!(d.commands.len(), 1);
            }
        }
    }

    // ── emit 클라이언트 없음 ─────────────────────────────────────────────

    #[test]
    fn emit_without_client_is_noop() {
        let (_dir, sock_path) = tmp_sock();
        let mut channel = UnixSocketChannel::new(&sock_path).unwrap();

        let signal = SystemSignal::ThermalAlert {
            level: Level::Normal,
            temperature_mc: 45_000,
            throttling_active: false,
            throttle_ratio: 1.0,
        };
        // 클라이언트 없음 — 에러 없이 통과
        channel.emit(&signal).unwrap();
        assert!(!channel.is_connected());
    }
}
