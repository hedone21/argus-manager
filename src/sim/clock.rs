//! VirtualClock: Duration 누적 + binary heap 기반 이벤트 큐.
//!
//! sub-tick 이벤트(heartbeat, signal polling, observation delay 등)를
//! 정확한 시점에 firing시키기 위해 min-heap을 사용한다.

#![allow(dead_code)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Duration;

// ─────────────────────────────────────────────────────────
// EventKind
// ─────────────────────────────────────────────────────────

/// 이벤트의 종류.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventKind {
    Heartbeat,
    SignalMemory,
    SignalCompute,
    SignalThermal,
    SignalEnergy,
    /// action 발동 후 observation_delay_secs(기본 3.0s) 경과 시 관측.
    ObservationDue {
        action: String,
        recorded_at: Duration,
    },
    /// external_injections[index] 시작.
    ExternalInjectionStart(usize),
    /// external_injections[index] 종료.
    ExternalInjectionEnd(usize),
    /// 시나리오 스크립트 확장용 커스텀 이벤트.
    Custom(String),
}

// ─────────────────────────────────────────────────────────
// ScheduledEvent
// ─────────────────────────────────────────────────────────

/// 특정 시각에 firing될 이벤트.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledEvent {
    pub at: Duration,
    /// 동일 시각의 이벤트는 id 오름차순으로 반환된다.
    pub id: u64,
    pub kind: EventKind,
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap은 max-heap이므로 Reverse로 감싸야 min-heap이 된다.
        // 여기서는 Reverse<ScheduledEvent>를 heap에 넣으므로
        // 자연 순서는 at asc, id asc (작을수록 "크다"고 판정되어 먼저 나온다).
        self.at.cmp(&other.at).then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ─────────────────────────────────────────────────────────
// VirtualClock
// ─────────────────────────────────────────────────────────

/// 결정론적 가상 시계.
///
/// - `advance(dt)`: 시간 전진.
/// - `drain_until(target)`: target 이하의 모든 이벤트를 시간 오름차순으로 반환.
/// - `schedule` / `schedule_periodic`: 이벤트 스케줄.
pub struct VirtualClock {
    now: Duration,
    events: BinaryHeap<Reverse<ScheduledEvent>>,
    next_id: u64,
}

impl VirtualClock {
    pub fn new() -> Self {
        Self {
            now: Duration::ZERO,
            events: BinaryHeap::new(),
            next_id: 0,
        }
    }

    /// 현재 시각 반환.
    pub fn now(&self) -> Duration {
        self.now
    }

    /// 현재 시각을 초 단위 f64로 반환.
    pub fn now_secs(&self) -> f64 {
        self.now.as_secs_f64()
    }

    /// 이벤트를 지정 시각에 스케줄하고 id를 반환한다.
    pub fn schedule(&mut self, kind: EventKind, at: Duration) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.events.push(Reverse(ScheduledEvent { at, id, kind }));
        id
    }

    /// [first, first+period, first+2*period, ...] ≤ until 까지의 이벤트를 프리로드한다.
    ///
    /// `kind_fn`은 각 시각마다 호출되어 EventKind를 생성한다.
    pub fn schedule_periodic(
        &mut self,
        kind_fn: impl Fn() -> EventKind,
        first: Duration,
        period: Duration,
        until: Duration,
    ) {
        let mut t = first;
        while t <= until {
            self.schedule(kind_fn(), t);
            t += period;
        }
    }

    /// `target` 이하의 모든 이벤트를 pop하여 시간 오름차순으로 반환한다.
    ///
    /// 이 함수는 `now`를 변경하지 않는다. 이후 `advance(dt)`를 별도로 호출할 것.
    pub fn drain_until(&mut self, target: Duration) -> Vec<ScheduledEvent> {
        let mut result = Vec::new();
        while let Some(Reverse(ev)) = self.events.peek() {
            if ev.at <= target {
                result.push(self.events.pop().unwrap().0);
            } else {
                break;
            }
        }
        // BinaryHeap에서 꺼낸 순서는 min-heap 보장이므로 이미 시간 오름차순.
        result
    }

    /// 시계를 dt만큼 전진시킨다.
    pub fn advance(&mut self, dt: Duration) {
        self.now += dt;
    }

    /// 아직 firing되지 않은 이벤트 수.
    pub fn pending(&self) -> usize {
        self.events.len()
    }
}

impl Default for VirtualClock {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────
// 단위 테스트
// ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_advances_monotonically() {
        let mut clock = VirtualClock::new();
        clock.advance(Duration::from_millis(100));
        clock.advance(Duration::from_millis(200));
        clock.advance(Duration::from_millis(50));
        assert_eq!(clock.now(), Duration::from_millis(350));
    }

    #[test]
    fn test_clock_drain_returns_events_in_time_order() {
        let mut clock = VirtualClock::new();
        clock.schedule(EventKind::Custom("a".into()), Duration::from_millis(500));
        clock.schedule(EventKind::Custom("b".into()), Duration::from_millis(1000));
        clock.schedule(EventKind::Custom("c".into()), Duration::from_millis(300));

        let events = clock.drain_until(Duration::from_millis(1000));
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].at, Duration::from_millis(300));
        assert_eq!(events[1].at, Duration::from_millis(500));
        assert_eq!(events[2].at, Duration::from_millis(1000));
    }

    #[test]
    fn test_clock_schedule_periodic() {
        let mut clock = VirtualClock::new();
        clock.schedule_periodic(
            || EventKind::Heartbeat,
            Duration::from_millis(250),
            Duration::from_millis(250),
            Duration::from_millis(1000),
        );
        // 250, 500, 750, 1000 → 4개
        assert_eq!(clock.pending(), 4);
        let events = clock.drain_until(Duration::from_millis(1000));
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].at, Duration::from_millis(250));
        assert_eq!(events[3].at, Duration::from_millis(1000));
    }

    #[test]
    fn test_clock_fires_exactly_at_sub_tick_boundary() {
        // dt=50ms, heartbeat period=1s → 20 ticks 후 정확히 1개 Heartbeat
        let mut clock = VirtualClock::new();
        let tick = Duration::from_millis(50);
        let period = Duration::from_secs(1);

        clock.schedule_periodic(|| EventKind::Heartbeat, period, period, period * 10);

        let mut heartbeat_count = 0;
        for _ in 0..20 {
            let next = clock.now() + tick;
            let events = clock.drain_until(next);
            for ev in &events {
                if ev.kind == EventKind::Heartbeat {
                    heartbeat_count += 1;
                }
            }
            clock.advance(tick);
        }
        // 20 ticks = 1000ms = 정확히 1 heartbeat
        assert_eq!(heartbeat_count, 1, "20 ticks(1s) 후 heartbeat 정확히 1회");
    }

    #[test]
    fn test_clock_drain_does_not_advance_now() {
        let mut clock = VirtualClock::new();
        clock.schedule(EventKind::Custom("x".into()), Duration::from_millis(100));
        clock.drain_until(Duration::from_millis(200));
        assert_eq!(
            clock.now(),
            Duration::ZERO,
            "drain_until은 now를 변경하지 않음"
        );
    }

    #[test]
    fn test_clock_same_time_order_by_id() {
        let mut clock = VirtualClock::new();
        let at = Duration::from_millis(500);
        clock.schedule(EventKind::Custom("first".into()), at);
        clock.schedule(EventKind::Custom("second".into()), at);
        clock.schedule(EventKind::Custom("third".into()), at);

        let events = clock.drain_until(at);
        assert_eq!(events.len(), 3);
        // id 오름차순 (0, 1, 2)
        assert!(events[0].id < events[1].id);
        assert!(events[1].id < events[2].id);
    }
}
