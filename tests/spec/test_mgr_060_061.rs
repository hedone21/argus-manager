//! MGR-060/061 ConnectionState FSM 테스트
//!
//! - MGR-060: Listening → Connected (accept), Connected → Disconnected (write err / reader EOF)
//! - MGR-061: Disconnected → Connected (try_accept on emit)
//!
//! UnixSocketChannel의 3-state FSM: Listening → Connected ↔ Disconnected
//! TcpChannel도 동일한 FSM을 따른다.

use argus_manager::channel::EngineReceiver;
use argus_manager::channel::TcpChannel;
use argus_manager::emitter::Emitter;
use argus_shared::{Level, SystemSignal};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

fn bind_free_port() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr.to_string()
}

fn make_signal() -> SystemSignal {
    SystemSignal::MemoryPressure {
        level: Level::Normal,
        available_bytes: 1_000_000_000,
        total_bytes: 2_000_000_000,
        reclaim_target_bytes: 0,
    }
}

/// MGR-060: 초기 상태 = Listening (클라이언트 미연결)
#[test]
fn test_mgr_060_listening_initial_state() {
    let addr = bind_free_port();
    let channel = TcpChannel::new(&addr).unwrap();
    assert!(
        !channel.is_connected(),
        "Initial state should be Listening (not connected)"
    );
}

/// MGR-060: Listening → Connected (클라이언트 accept)
#[test]
fn test_mgr_060_listening_to_connected() {
    let addr = bind_free_port();
    let mut channel = TcpChannel::new(&addr).unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));

    let _client = TcpStream::connect(&addr).unwrap();
    assert!(channel.wait_for_client(Duration::from_secs(1), &shutdown));
    assert!(
        channel.is_connected(),
        "After accept, should be in Connected state"
    );
}

/// MGR-060: Connected → Disconnected (클라이언트 연결 닫힘 → reader EOF)
#[test]
fn test_mgr_060_connected_to_disconnected() {
    let addr = bind_free_port();
    let mut channel = TcpChannel::new(&addr).unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));

    let client = TcpStream::connect(&addr).unwrap();
    channel.wait_for_client(Duration::from_secs(1), &shutdown);
    assert!(channel.is_connected());

    // 클라이언트 연결 닫기 → reader EOF
    drop(client);
    std::thread::sleep(Duration::from_millis(100));

    // try_recv가 Disconnected를 감지
    channel.try_recv().unwrap();
    assert!(
        !channel.is_connected(),
        "After client disconnect, should be in Disconnected state"
    );
}

/// MGR-061: Disconnected → Connected (재연결, emit 시 try_accept)
#[test]
fn test_mgr_061_disconnected_to_listening() {
    let addr = bind_free_port();
    let mut channel = TcpChannel::new(&addr).unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));

    // 1차 연결 + 끊기 → Disconnected
    let client1 = TcpStream::connect(&addr).unwrap();
    channel.wait_for_client(Duration::from_secs(1), &shutdown);
    drop(client1);
    std::thread::sleep(Duration::from_millis(100));
    channel.try_recv().unwrap();
    assert!(!channel.is_connected());

    // 2차 연결: emit 호출 시 ensure_connected → try_accept
    let _client2 = TcpStream::connect(&addr).unwrap();
    channel.emit(&make_signal()).unwrap();
    assert!(
        channel.is_connected(),
        "After reconnect via emit, should be Connected"
    );
}

/// MGR-061: 전체 사이클 Listening → Connected → Disconnected → Connected
#[test]
fn test_mgr_061_full_cycle() {
    let addr = bind_free_port();
    let mut channel = TcpChannel::new(&addr).unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));

    // Phase 1: Listening
    assert!(!channel.is_connected());

    // Phase 2: → Connected
    let client1 = TcpStream::connect(&addr).unwrap();
    channel.wait_for_client(Duration::from_secs(1), &shutdown);
    assert!(channel.is_connected());

    // Phase 3: → Disconnected
    drop(client1);
    std::thread::sleep(Duration::from_millis(100));
    channel.try_recv().unwrap();
    assert!(!channel.is_connected());

    // Phase 4: → Connected (재연결)
    let _client2 = TcpStream::connect(&addr).unwrap();
    // ensure_connected는 non-blocking accept이므로 약간의 대기 필요
    std::thread::sleep(Duration::from_millis(50));
    channel.emit(&make_signal()).unwrap();
    assert!(channel.is_connected());
}
