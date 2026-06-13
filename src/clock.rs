//! MGR-030 / INV-093: 결정론적 시간 추상화.
//!
//! Production은 `SystemClock`(단조 Instant 기반), 테스트/시뮬레이터는
//! `VirtualClockHandle`(별도 어댑터)로 `Clock` trait을 구현한다.
//! `LogicalInstant`는 프로세스 시작 대비 누적 Duration을 래핑하여
//! 두 구현체 간 호환 가능한 비교/뺄셈을 제공한다.

use std::time::{Duration, Instant, SystemTime};

/// 단조증가 논리 시각 (프로세스 시작 대비 누적 Duration).
/// `Instant`의 opaqueness를 피하면서 동일한 비교/산술 연산을 제공.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalInstant(Duration);

impl LogicalInstant {
    pub const ZERO: LogicalInstant = LogicalInstant(Duration::ZERO);

    pub fn from_duration_since_start(d: Duration) -> Self {
        LogicalInstant(d)
    }

    pub fn as_duration_since_start(self) -> Duration {
        self.0
    }

    /// 단조 감산. `earlier > self`이면 `Duration::ZERO` 반환 (saturating).
    pub fn saturating_duration_since(self, earlier: LogicalInstant) -> Duration {
        self.0.saturating_sub(earlier.0)
    }

    pub fn checked_add(self, rhs: Duration) -> Option<LogicalInstant> {
        self.0.checked_add(rhs).map(LogicalInstant)
    }
}

impl std::ops::Add<Duration> for LogicalInstant {
    type Output = LogicalInstant;
    fn add(self, rhs: Duration) -> LogicalInstant {
        LogicalInstant(self.0 + rhs)
    }
}

/// 결정론적 시계 추상화.
/// - Production: `SystemClock` (std::time::Instant 래핑)
/// - 테스트: VirtualClock 어댑터 (후속 PR에서 구현)
pub trait Clock: Send + Sync {
    /// 프로세스 시작 대비 누적 논리 시각. 단조증가 보장.
    fn now(&self) -> LogicalInstant;

    /// `earlier` 이후 경과 시간. `earlier`가 미래면 `Duration::ZERO` 반환 (saturating).
    fn elapsed_since(&self, earlier: LogicalInstant) -> Duration {
        self.now().saturating_duration_since(earlier)
    }

    /// Unix epoch 기준 현재 시각 (로깅/JSON 직렬화용).
    /// 기본 구현은 `SystemTime::now()` — 가상 시계는 override.
    fn now_system(&self) -> SystemTime {
        SystemTime::now()
    }
}

/// Production 구현. 프로세스 시작 시각을 기록해 상대 Duration을 반환.
#[derive(Debug)]
pub struct SystemClock {
    start: Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now(&self) -> LogicalInstant {
        LogicalInstant(self.start.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_system_clock_is_monotonic() {
        let clock = SystemClock::new();
        let t1 = clock.now();
        thread::sleep(Duration::from_millis(1));
        let t2 = clock.now();
        assert!(t2 >= t1, "두 번째 now()가 첫 번째보다 크거나 같아야 한다");
        assert!(
            t2 > t1,
            "1ms sleep 후에는 명확히 증가해야 한다: t1={:?}, t2={:?}",
            t1,
            t2
        );
    }

    #[test]
    fn test_logical_instant_saturating_duration_since_handles_before() {
        let earlier = LogicalInstant::from_duration_since_start(Duration::from_secs(10));
        let later = LogicalInstant::from_duration_since_start(Duration::from_secs(5));
        // earlier > self (later) → Duration::ZERO 반환
        let result = later.saturating_duration_since(earlier);
        assert_eq!(result, Duration::ZERO);
    }

    #[test]
    fn test_logical_instant_add_duration_roundtrip() {
        let zero = LogicalInstant::ZERO;
        let five_secs = Duration::from_secs(5);
        let result = (zero + five_secs).as_duration_since_start();
        assert_eq!(result, five_secs);
    }

    #[test]
    fn test_logical_instant_is_copy() {
        let inst = LogicalInstant::from_duration_since_start(Duration::from_secs(1));
        // Copy 바운드 확인: move 없이 두 변수가 동시에 유효해야 한다
        let a = inst;
        let b = inst;
        assert_eq!(a, b);
    }

    #[test]
    fn test_system_clock_now_within_reasonable_range() {
        let clock = SystemClock::new();
        let elapsed = clock.now().as_duration_since_start();
        assert!(
            elapsed < Duration::from_secs(1),
            "SystemClock::new() 직후 now()는 1초 미만이어야 한다: {:?}",
            elapsed
        );
    }

    #[test]
    fn test_clock_trait_is_dyn_safe() {
        // dyn-safe 확인: Box<dyn Clock>으로 생성 가능해야 한다
        let _: Box<dyn Clock> = Box::new(SystemClock::new());
    }
}
