//! Policy pipeline — 새 계층형 정책 메인 루프용 상태 캡슐화.
//!
//! `HierarchicalPolicy`는 PI Controller, Supervisory Layer, Action Selector,
//! Relief Estimator를 연결하여 `SystemSignal` 입력에서 `EngineDirective`를
//! 생성하는 전체 파이프라인을 담당한다.
//!
//! # 설계 참고
//!
//! `docs/36_policy_design.md` §9 (Manager Main Loop)를 참조한다.

use crate::signal::SystemSignal;
use argus_shared::{EngineCommand, EngineDirective, EngineMessage};

use crate::types::OperatingMode;

/// 정책의 내부 상태를 타입 안전하고 선언적으로 수집하기 위한 인스펙터 인터페이스
pub trait PolicyVisitor {
    fn record_f32(&mut self, key: &str, value: f32);
    fn record_u64(&mut self, key: &str, value: u64);
    fn record_string(&mut self, key: &str, value: &str);
}

/// 핫-리로드를 지원하는 정책 대상의 인터페이스
pub trait ReloadablePolicy {
    fn reload_script(&mut self, path: &std::path::Path) -> anyhow::Result<()>;
    fn script_path(&self) -> Option<&std::path::Path>;
}

/// 정책 판단 계층의 공통 인터페이스.
///
/// Monitor가 수집한 SystemSignal을 처리하여 EngineDirective를 생성한다.
/// 구현체에 따라 PI+Supervisory+Selector(HierarchicalPolicy) 또는
/// 규칙 기반(ThresholdPolicy) 등 다양한 전략이 가능하다.
pub trait PolicyStrategy: Send {
    /// SystemSignal을 처리하여 필요 시 EngineDirective를 반환한다.
    fn process_signal(&mut self, signal: &SystemSignal) -> Option<EngineDirective>;

    /// Engine의 heartbeat/capability/response 메시지로 내부 상태를 갱신한다.
    fn update_engine_state(&mut self, msg: &EngineMessage);

    /// 현재 operating mode를 반환한다 (로깅/모니터링용).
    fn mode(&self) -> OperatingMode;

    /// 세션 종료 시 내부 모델을 저장한다. 기본 구현은 no-op.
    fn save_model(&self) {}

    /// 진단 및 시뮬레이션을 위한 상태 노출 인터페이스 (대안 A)
    fn inspect_state(&mut self, _visitor: &mut dyn PolicyVisitor) {}

    /// 직전 process_signal() 호출에서 큐잉된 observation을 취소한다.
    ///
    /// dedup이 directive를 suppress했을 때 호출된다.
    /// 기본 구현은 no-op (관측 기능 없는 policy에서 그냥 무시).
    fn cancel_last_observation(&mut self) {}

    /// 핫-리로드 기능 접근을 위한 다운캐스트 헬퍼
    fn as_reloadable(&mut self) -> Option<&mut dyn ReloadablePolicy> {
        None
    }
}

/// Helper to get observation overrun count using inspect_state
pub fn get_observation_overrun_count(policy: &mut dyn PolicyStrategy) -> u64 {
    struct OverrunCollector {
        count: u64,
    }
    impl PolicyVisitor for OverrunCollector {
        fn record_f32(&mut self, _key: &str, _value: f32) {}
        fn record_u64(&mut self, key: &str, value: u64) {
            if key == "observation_overrun_count" {
                self.count = value;
            }
        }
        fn record_string(&mut self, _key: &str, _value: &str) {}
    }
    let mut collector = OverrunCollector { count: 0 };
    policy.inspect_state(&mut collector);
    collector.count
}

/// Seq ID 생성을 위한 단조 증가 카운터.
static SEQ_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub fn next_seq_id() -> u64 {
    SEQ_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// 연속으로 동일한 directive가 방출되는 것을 억제한다.
///
/// `process(directive, now_secs)`는 `commands`가 마지막으로 방출된 배치와 다르거나
/// cooldown이 경과했을 때만 `Some(directive)`를 반환하고, 그 외에는 `None`을 반환한다.
///
/// cooldown이 경과하면 동일한 directive도 재방출하여 relief observation이
/// 쌓일 수 있도록 한다.
pub struct DirectiveDeduplicator {
    last_commands: Option<Vec<EngineCommand>>,
    /// 마지막으로 directive를 방출한 시각 (seconds).
    last_sent_at: Option<f64>,
    /// 동일한 directive를 재방출하기 위한 cooldown (seconds).
    cooldown_secs: f64,
}

impl DirectiveDeduplicator {
    /// 기본 cooldown(60s)으로 생성한다.
    pub fn new() -> Self {
        Self::with_cooldown(60.0)
    }

    /// 지정한 cooldown(seconds)으로 생성한다.
    pub fn with_cooldown(cooldown_secs: f64) -> Self {
        Self {
            last_commands: None,
            last_sent_at: None,
            cooldown_secs,
        }
    }

    /// 방출해야 하면 `Some(directive)`, 억제해야 하면 `None`을 반환한다.
    ///
    /// `now_secs`는 프로세스 시작 시각 기준 경과 초(seconds)이다.
    pub fn process(
        &mut self,
        directive: EngineDirective,
        now_secs: f64,
    ) -> Option<EngineDirective> {
        let is_dup = self.last_commands.as_ref() == Some(&directive.commands);
        let cooldown_elapsed = self
            .last_sent_at
            .map(|t| now_secs - t >= self.cooldown_secs)
            .unwrap_or(false);
        let suppress = is_dup && !cooldown_elapsed;

        if suppress {
            None
        } else {
            self.last_commands = Some(directive.commands.clone());
            self.last_sent_at = Some(now_secs);
            Some(directive)
        }
    }
}

impl Default for DirectiveDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod dedup_tests {
    use super::*;
    use argus_shared::{EngineCommand, EngineDirective};

    fn directive(commands: Vec<EngineCommand>) -> EngineDirective {
        EngineDirective {
            seq_id: 0,
            commands,
        }
    }

    /// A compression at a given budget. The dedup unit is the whole command vector, so
    /// two budgets that differ are two different directives.
    fn compress(budget: f32) -> EngineCommand {
        EngineCommand::KvCompress { budget }
    }

    /// A second, distinct command — the "different action" half of the dedup tests.
    fn other() -> EngineCommand {
        EngineCommand::RestoreDefaults
    }

    // 케이스 1: 첫 번째 directive는 항상 통과
    #[test]
    fn first_directive_always_passes() {
        let mut dedup = DirectiveDeduplicator::new();
        let d = directive(vec![compress(0.8)]);
        assert!(dedup.process(d, 0.0).is_some());
    }

    // 케이스 2: 동일 commands 연속 → 두 번째부터 억제
    #[test]
    fn identical_commands_suppressed() {
        let mut dedup = DirectiveDeduplicator::new();
        assert!(dedup.process(directive(vec![compress(0.8)]), 0.0).is_some());
        assert!(dedup.process(directive(vec![compress(0.8)]), 1.0).is_none());
        assert!(dedup.process(directive(vec![compress(0.8)]), 2.0).is_none());
    }

    // 케이스 3: 같은 타입, 다른 파라미터 → 방출
    #[test]
    fn same_type_different_param_passes() {
        let mut dedup = DirectiveDeduplicator::new();
        assert!(dedup.process(directive(vec![compress(0.8)]), 0.0).is_some());
        assert!(dedup.process(directive(vec![compress(0.5)]), 1.0).is_some());
    }

    // 케이스 4: A → B → A 전환: 두 번째 A는 방출 (last=B이므로)
    #[test]
    fn cycle_a_b_a_emits_second_a() {
        let mut dedup = DirectiveDeduplicator::new();
        assert!(dedup.process(directive(vec![compress(0.8)]), 0.0).is_some()); // A 방출
        assert!(dedup.process(directive(vec![other()]), 1.0).is_some()); // B 방출
        assert!(dedup.process(directive(vec![compress(0.8)]), 2.0).is_some()); // A 재방출
    }

    // 케이스 5: multi-command directive 동일하면 억제
    #[test]
    fn multi_command_identical_suppressed() {
        let mut dedup = DirectiveDeduplicator::new();
        let cmds = vec![compress(0.8), other()];
        assert!(dedup.process(directive(cmds.clone()), 0.0).is_some());
        assert!(dedup.process(directive(cmds), 1.0).is_none());
    }

    // 케이스 6: None(억제) 이후 다른 directive 방출 가능 확인
    #[test]
    fn after_suppression_different_command_passes() {
        let mut dedup = DirectiveDeduplicator::new();
        assert!(dedup.process(directive(vec![compress(0.8)]), 0.0).is_some());
        assert!(dedup.process(directive(vec![compress(0.8)]), 1.0).is_none()); // 억제
        assert!(dedup.process(directive(vec![other()]), 2.0).is_some()); // 다른 커맨드 → 방출
    }

    // 케이스 7: 억제된 directive는 last_commands를 갱신하지 않음
    // A → A(억제) → B 시퀀스에서 B는 A와 다르므로 방출되어야 한다.
    #[test]
    fn suppressed_does_not_update_last() {
        let mut dedup = DirectiveDeduplicator::new();
        assert!(dedup.process(directive(vec![compress(0.8)]), 0.0).is_some()); // A
        assert!(dedup.process(directive(vec![compress(0.8)]), 1.0).is_none()); // A 억제 → last=A 유지
        // B는 A와 다르므로 방출되어야 함
        assert!(dedup.process(directive(vec![other()]), 2.0).is_some()); // B
    }

    // 케이스 8: cooldown 후 동일한 directive 통과
    #[test]
    fn cooldown_allows_same_after_timeout() {
        let mut dedup = DirectiveDeduplicator::with_cooldown(60.0);
        // t=0: 첫 방출
        assert!(dedup.process(directive(vec![compress(0.8)]), 0.0).is_some());
        // t=30: cooldown 미경과 → 억제
        assert!(
            dedup
                .process(directive(vec![compress(0.8)]), 30.0)
                .is_none()
        );
        // t=59: cooldown 미경과 → 억제
        assert!(
            dedup
                .process(directive(vec![compress(0.8)]), 59.0)
                .is_none()
        );
        // t=60: cooldown 경과 → 재방출
        assert!(
            dedup
                .process(directive(vec![compress(0.8)]), 60.0)
                .is_some()
        );
        // t=100: cooldown 미경과(마지막 방출=60) → 억제
        assert!(
            dedup
                .process(directive(vec![compress(0.8)]), 100.0)
                .is_none()
        );
        // t=120: cooldown 경과(마지막 방출=60) → 재방출
        assert!(
            dedup
                .process(directive(vec![compress(0.8)]), 120.0)
                .is_some()
        );
    }

    // 케이스 9: directive가 바뀌면 cooldown 타이머 리셋
    #[test]
    fn cooldown_reset_on_change() {
        let mut dedup = DirectiveDeduplicator::with_cooldown(60.0);
        // t=0: A 방출
        assert!(dedup.process(directive(vec![compress(0.8)]), 0.0).is_some());
        // t=50: B 방출 (다른 directive → 타이머 리셋)
        assert!(dedup.process(directive(vec![other()]), 50.0).is_some());
        // t=90: B 동일 → cooldown 미경과(50+60=110) → 억제
        assert!(dedup.process(directive(vec![other()]), 90.0).is_none());
        // t=110: cooldown 경과 → 재방출
        assert!(dedup.process(directive(vec![other()]), 110.0).is_some());
    }
}
