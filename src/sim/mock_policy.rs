//! MockPolicy: 테스트 전용 PolicyStrategy 구현체.
//!
//! - process_signal 호출 횟수/내용 기록
//! - update_engine_state 호출 횟수 기록
//! - check_qcf_timeout 호출 횟수 기록
//! - directive_on_signal 콜백으로 커스텀 응답 주입 가능

#![allow(dead_code)]

use crate::pipeline::PolicyStrategy;
use crate::types::OperatingMode;
use argus_shared::{EngineDirective, EngineMessage, SystemSignal};

/// 신호 핸들러 콜백 타입.
type SignalHandler = Box<dyn Fn(&SystemSignal) -> Option<EngineDirective> + Send>;

/// test-only PolicyStrategy 구현.
#[derive(Default)]
pub struct MockPolicy {
    pub received_signals: Vec<SystemSignal>,
    pub received_heartbeats: usize,
    pub qcf_timeout_count: usize,
    /// 신호를 받을 때 호출되는 콜백. None이면 항상 None 반환.
    pub directive_on_signal: Option<SignalHandler>,
    /// check_qcf_timeout이 반환할 Directive (일회성).
    pub pending_qcf_directive: Option<EngineDirective>,
}

impl std::fmt::Debug for MockPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockPolicy")
            .field("received_signals", &self.received_signals.len())
            .field("received_heartbeats", &self.received_heartbeats)
            .field("qcf_timeout_count", &self.qcf_timeout_count)
            .finish()
    }
}

impl MockPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// process_signal에서 항상 지정된 Directive를 반환하는 MockPolicy를 생성한다.
    pub fn with_fixed_directive(dir: EngineDirective) -> Self {
        let dir_clone = dir.clone();
        Self {
            directive_on_signal: Some(Box::new(move |_| Some(dir_clone.clone()))),
            ..Default::default()
        }
    }
}

impl PolicyStrategy for MockPolicy {
    fn process_signal(&mut self, signal: &SystemSignal) -> Option<EngineDirective> {
        self.received_signals.push(signal.clone());
        if let Some(ref cb) = self.directive_on_signal {
            cb(signal)
        } else {
            None
        }
    }

    fn update_engine_state(&mut self, _msg: &EngineMessage) {
        self.received_heartbeats += 1;
    }

    fn mode(&self) -> OperatingMode {
        OperatingMode::Normal
    }

    fn check_qcf_timeout(&mut self) -> Option<EngineDirective> {
        self.qcf_timeout_count += 1;
        self.pending_qcf_directive.take()
    }
}
