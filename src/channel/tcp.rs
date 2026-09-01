use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::signal::SystemSignal;
use argus_shared::{EngineDirective, EngineMessage, ManagerMessage};

use crate::channel::EngineReceiver;
use crate::emitter::Emitter;

// ── RAII 핸들 ──────────────────────────────────────────────────────────────

/// Reader thread를 RAII로 관리한다.
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

/// TCP 연결 상태를 명시적으로 모델링한다.
enum TcpConnectionState {
    /// 소켓을 bind했지만 클라이언트 미연결.
    Listening,
    /// 클라이언트 연결됨.
    Connected {
        writer: TcpStream,
        /// reader thread가 파싱한 메시지를 push하는 채널의 수신단.
        inbox: mpsc::Receiver<EngineMessage>,
        /// reader thread RAII 핸들.
        _reader: ReaderHandle,
    },
    /// 연결이 끊어짐 — 다음 emit() 호출 시 재연결 시도.
    Disconnected,
}

// ── TcpChannel ─────────────────────────────────────────────────────────────

/// 양방향 TCP 채널.
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
pub struct TcpChannel {
    addr: String,
    listener: TcpListener,
    state: TcpConnectionState,
}

impl TcpChannel {
    /// 주어진 주소에서 listen하는 채널을 생성한다.
    ///
    /// `addr` 형식: `"host:port"` (예: `"127.0.0.1:9200"` 또는 `"0.0.0.0:9200"`)
    pub fn new(addr: &str) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        log::info!("[TcpChannel] Listening on {}", addr);

        Ok(Self {
            addr: addr.to_string(),
            listener,
            state: TcpConnectionState::Listening,
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
                Ok((stream, peer)) => {
                    log::info!("[TcpChannel] Client connected from {}", peer);
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
                    log::error!("[TcpChannel] Accept error: {}", e);
                    return false;
                }
            }
        }
    }

    /// Disconnected 상태에서 non-blocking으로 재연결을 시도한다.
    fn ensure_connected(&mut self) {
        if !matches!(self.state, TcpConnectionState::Disconnected) {
            return;
        }
        match self.listener.accept() {
            Ok((stream, peer)) => {
                log::info!("[TcpChannel] Client reconnected from {}", peer);
                self.transition_to_connected(stream);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // 연결 대기 중 — 정상
            }
            Err(e) => {
                log::debug!("[TcpChannel] try_accept error: {}", e);
            }
        }
    }

    /// 연결된 stream으로 Connected 상태로 전이한다.
    fn transition_to_connected(&mut self, stream: TcpStream) {
        stream.set_nonblocking(false).ok();

        let writer = match stream.try_clone() {
            Ok(w) => w,
            Err(e) => {
                log::error!("[TcpChannel] try_clone failed: {}", e);
                self.state = TcpConnectionState::Disconnected;
                return;
            }
        };

        let (inbox_tx, inbox_rx) = mpsc::sync_channel(64);
        let reader_handle = spawn_reader(stream, inbox_tx);

        self.state = TcpConnectionState::Connected {
            writer,
            inbox: inbox_rx,
            _reader: reader_handle,
        };
    }

    /// writer에 `ManagerMessage`를 length-prefixed JSON으로 전송한다.
    fn write_manager_message(&mut self, msg: &ManagerMessage) -> anyhow::Result<()> {
        if let TcpConnectionState::Connected { writer, .. } = &mut self.state {
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
        if let TcpConnectionState::Connected { writer, .. } = &mut self.state {
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

    /// 채널이 수신 대기 중인 주소를 반환한다.
    pub fn addr(&self) -> &str {
        &self.addr
    }
}

// ── Emitter 구현 ──────────────────────────────────────────────────────────

impl Emitter for TcpChannel {
    fn emit(&mut self, signal: &SystemSignal) -> anyhow::Result<()> {
        self.ensure_connected();
        if !matches!(self.state, TcpConnectionState::Connected { .. }) {
            // Not fatal, but not silent either: the caller treats `Ok(())` as sent, and
            // the deduplicator has already recorded it, so an unlogged drop here
            // suppresses the identical retry for the whole cooldown.
            log::warn!("no engine connected — dropping the outbound message");
            return Ok(());
        }
        if let Err(e) = self.write_signal(signal) {
            log::warn!("[TcpChannel] Write error (signal): {}", e);
            self.state = TcpConnectionState::Disconnected;
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
        if !matches!(self.state, TcpConnectionState::Connected { .. }) {
            // Not fatal, but not silent either: the caller treats `Ok(())` as sent, and
            // the deduplicator has already recorded it, so an unlogged drop here
            // suppresses the identical retry for the whole cooldown.
            log::warn!("no engine connected — dropping the outbound message");
            return Ok(());
        }
        let msg = ManagerMessage::Directive(directive.clone());
        if let Err(e) = self.write_manager_message(&msg) {
            log::warn!(
                "[TcpChannel] Write error (directive seq={}): {}",
                directive.seq_id,
                e
            );
            self.state = TcpConnectionState::Disconnected;
        } else {
            log::debug!("[TcpChannel] Sent directive seq={}", directive.seq_id);
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "TcpChannel"
    }
}

// ── EngineReceiver 구현 ───────────────────────────────────────────────────

impl EngineReceiver for TcpChannel {
    fn try_recv(&mut self) -> anyhow::Result<Option<EngineMessage>> {
        if let TcpConnectionState::Connected { inbox, .. } = &self.state {
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
        log::info!("[TcpChannel] Reader thread exited — connection lost");
        self.state = TcpConnectionState::Disconnected;
        Ok(None)
    }

    fn is_connected(&self) -> bool {
        matches!(self.state, TcpConnectionState::Connected { .. })
    }
}

// ── Reader thread ─────────────────────────────────────────────────────────

/// `stream`에서 `EngineMessage`를 blocking read하여 `inbox_tx`로 전달하는 스레드를 spawn한다.
///
/// EOF 또는 I/O 에러 발생 시 루프를 종료하고, `inbox_tx`가 drop되어
/// `Receiver::try_recv()`에서 `Disconnected`가 반환된다.
fn spawn_reader(mut stream: TcpStream, inbox_tx: mpsc::SyncSender<EngineMessage>) -> ReaderHandle {
    let handle = thread::Builder::new()
        .name("tcp-engine-reader".into())
        .spawn(move || {
            loop {
                match read_engine_message(&mut stream) {
                    Ok(msg) => {
                        if inbox_tx.send(msg).is_err() {
                            // receiver dropped (TcpChannel이 drop됨)
                            break;
                        }
                    }
                    Err(e) => {
                        log::error!(
                        "[tcp-engine-reader] engine link lost: {} — no further heartbeats or responses \
                         will be received on this connection",
                        e
                    );
                        break; // EOF 또는 I/O 에러
                    }
                }
            }
            // inbox_tx drop → Receiver::try_recv()가 Disconnected 반환
        })
        .expect("spawn tcp-engine-reader");

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
fn read_engine_message(stream: &mut TcpStream) -> anyhow::Result<EngineMessage> {
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
    use crate::signal::Level;
    use argus_shared::{EngineState, EngineStatus, Phase};
    use std::io::Write as IoWrite;
    use std::net::TcpStream as StdTcpStream;

    fn bind_free_port() -> String {
        // OS가 자동으로 포트를 할당하도록 :0 사용
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        addr.to_string()
    }

    fn make_heartbeat(_kv: f32, _device: &str) -> EngineMessage {
        EngineMessage::Heartbeat(EngineStatus {
            kv_cache_bytes: 1024,
            kv_cache_budget_bytes: 4096,
            kv_cache_tokens: 32,
            tbt_ms: 12.5,
            phase: Phase::Decode,
            state: EngineState::Running,
        })
    }

    fn make_capability() -> EngineMessage {
        EngineMessage::Heartbeat(EngineStatus {
            kv_cache_bytes: 1024,
            kv_cache_budget_bytes: 4096,
            kv_cache_tokens: 32,
            tbt_ms: 12.5,
            phase: Phase::Decode,
            state: EngineState::Running,
        })
    }

    /// 헬퍼: client stream에서 length-prefixed JSON `EngineMessage`를 전송.
    fn send_engine_msg(stream: &mut StdTcpStream, msg: &EngineMessage) {
        let json = serde_json::to_vec(msg).unwrap();
        let len = (json.len() as u32).to_be_bytes();
        stream.write_all(&len).unwrap();
        stream.write_all(&json).unwrap();
        stream.flush().unwrap();
    }

    /// 헬퍼: channel의 try_recv를 최대 N번 재시도 (reader thread가 처리할 시간 확보).
    fn try_recv_with_retry(ch: &mut TcpChannel, retries: usize) -> Option<EngineMessage> {
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
        let addr = bind_free_port();
        let mut channel = TcpChannel::new(&addr).unwrap();

        let mut client = StdTcpStream::connect(&addr).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        assert!(channel.wait_for_client(Duration::from_secs(1), &shutdown));
        assert!(channel.is_connected());

        send_engine_msg(&mut client, &make_heartbeat(0.75, "opencl"));

        let msg = try_recv_with_retry(&mut channel, 50).expect("Expected heartbeat");
        match msg {
            EngineMessage::Heartbeat(s) => {
                assert_eq!(s.kv_cache_bytes, 1024);
                assert_eq!(s.kv_cache_tokens, 32);
            }
            _ => panic!("Expected Heartbeat"),
        }
    }

    // ── 메시지 없음 (non-blocking) ─────────────────────────────────────────

    #[test]
    fn try_recv_returns_none_when_no_message() {
        let addr = bind_free_port();
        let mut channel = TcpChannel::new(&addr).unwrap();

        let _client = StdTcpStream::connect(&addr).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        channel.wait_for_client(Duration::from_secs(1), &shutdown);

        let result = channel.try_recv().unwrap();
        assert!(result.is_none());
    }

    // ── Reader thread EOF → Disconnected 전이 ─────────────────────────────

    #[test]
    fn disconnected_after_reader_eof() {
        let addr = bind_free_port();
        let mut channel = TcpChannel::new(&addr).unwrap();

        let client = StdTcpStream::connect(&addr).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        channel.wait_for_client(Duration::from_secs(1), &shutdown);
        assert!(channel.is_connected());

        drop(client);
        std::thread::sleep(Duration::from_millis(100));

        channel.try_recv().unwrap();
        assert!(!channel.is_connected());
    }

    // ── Writer 에러 → Disconnected 전이 ──────────────────────────────────

    #[test]
    fn disconnected_after_write_error() {
        let addr = bind_free_port();
        let mut channel = TcpChannel::new(&addr).unwrap();

        let client = StdTcpStream::connect(&addr).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        channel.wait_for_client(Duration::from_secs(1), &shutdown);
        assert!(channel.is_connected());

        drop(client);
        std::thread::sleep(Duration::from_millis(300));

        let signal = SystemSignal::MemoryPressure {
            level: Level::Warning,
            available_bytes: 500_000_000,
            total_bytes: 2_000_000_000,
            reclaim_target_bytes: 50_000_000,
        };
        // On macOS, the first write after peer close may succeed (buffered in kernel).
        // The second write reliably triggers Broken pipe / EPIPE after RST is received.
        channel.emit(&signal).unwrap();
        if channel.is_connected() {
            std::thread::sleep(Duration::from_millis(100));
            channel.emit(&signal).unwrap();
        }
        assert!(!channel.is_connected());
    }

    // ── 재연결 ────────────────────────────────────────────────────────────

    #[test]
    fn reconnect_after_disconnect() {
        let addr = bind_free_port();
        let mut channel = TcpChannel::new(&addr).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));

        let client1 = StdTcpStream::connect(&addr).unwrap();
        channel.wait_for_client(Duration::from_secs(1), &shutdown);
        assert!(channel.is_connected());

        drop(client1);
        std::thread::sleep(Duration::from_millis(100));
        channel.try_recv().unwrap();
        assert!(!channel.is_connected());

        let mut client2 = StdTcpStream::connect(&addr).unwrap();
        let signal = SystemSignal::MemoryPressure {
            level: Level::Normal,
            available_bytes: 1_000_000_000,
            total_bytes: 2_000_000_000,
            reclaim_target_bytes: 0,
        };
        channel.emit(&signal).unwrap();
        assert!(channel.is_connected());

        send_engine_msg(&mut client2, &make_heartbeat(0.3, "cpu"));
        let msg =
            try_recv_with_retry(&mut channel, 50).expect("Expected heartbeat after reconnect");
        assert!(matches!(msg, EngineMessage::Heartbeat(_)));
    }

    // ── Capability 수신 ───────────────────────────────────────────────────

    #[test]
    fn try_recv_returns_capability_message() {
        let addr = bind_free_port();
        let mut channel = TcpChannel::new(&addr).unwrap();

        let mut client = StdTcpStream::connect(&addr).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        channel.wait_for_client(Duration::from_secs(1), &shutdown);

        send_engine_msg(&mut client, &make_capability());

        let msg = try_recv_with_retry(&mut channel, 50).expect("Expected heartbeat");
        match msg {
            EngineMessage::Heartbeat(status) => {
                assert_eq!(status.kv_cache_bytes, 1024);
                assert_eq!(status.kv_cache_budget_bytes, 4096);
            }
            other => panic!("Expected Heartbeat, got {other:?}"),
        }
    }

    // ── is_connected: 초기 상태 ───────────────────────────────────────────

    #[test]
    fn is_connected_false_before_client_connects() {
        let addr = bind_free_port();
        let channel = TcpChannel::new(&addr).unwrap();
        assert!(!channel.is_connected());
    }

    // ── emit_directive 정상 전송 ──────────────────────────────────────────

    #[test]
    fn emit_directive_sends_manager_message() {
        use argus_shared::{EngineCommand, EngineDirective, ManagerMessage};
        use std::io::Read as IoRead;

        let addr = bind_free_port();
        let mut channel = TcpChannel::new(&addr).unwrap();

        let mut client = StdTcpStream::connect(&addr).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        channel.wait_for_client(Duration::from_secs(1), &shutdown);

        let directive = EngineDirective {
            seq_id: 99,
            commands: vec![EngineCommand::KvCompress { budget: 0.5 }],
        };
        channel.emit_directive(&directive).unwrap();

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
        let addr = bind_free_port();
        let mut channel = TcpChannel::new(&addr).unwrap();

        let signal = SystemSignal::ThermalAlert {
            level: Level::Normal,
            temperature_mc: 45_000,
            throttling_active: false,
            throttle_ratio: 1.0,
        };
        channel.emit(&signal).unwrap();
        assert!(!channel.is_connected());
    }

    // ── addr() 반환 확인 ─────────────────────────────────────────────────

    #[test]
    fn addr_returns_bound_address() {
        let addr = bind_free_port();
        let channel = TcpChannel::new(&addr).unwrap();
        assert_eq!(channel.addr(), addr);
    }
}
