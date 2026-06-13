pub mod tcp;
pub mod unix_socket;

pub use tcp::TcpChannel;

use argus_shared::EngineMessage;

use crate::emitter::Emitter;

/// Engine → Manager 방향 수신 전용 인터페이스.
///
/// `Emitter`와 분리하여 ISP를 준수한다.
/// `DbusEmitter` 등 수신 기능이 없는 구현체는 이 trait을 구현하지 않는다.
pub trait EngineReceiver: Send {
    /// Engine으로부터 메시지를 수신한다 (non-blocking).
    ///
    /// 수신된 메시지가 없으면 `Ok(None)`을 반환한다.
    fn try_recv(&mut self) -> anyhow::Result<Option<EngineMessage>>;

    /// 수신 채널이 살아있는지 확인한다.
    fn is_connected(&self) -> bool;
}

/// `Emitter` + `EngineReceiver`를 모두 구현하는 구조체를 위한 조합 trait.
///
/// main.rs는 이 trait을 통해 양방향 채널에 접근한다.
pub trait EngineChannel: Emitter + EngineReceiver {}

// 블랭킷 구현 — Emitter + EngineReceiver를 모두 구현하면 자동으로 EngineChannel.
impl<T: Emitter + EngineReceiver> EngineChannel for T {}
