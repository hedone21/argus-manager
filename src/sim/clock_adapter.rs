//! VirtualClockHandle: VirtualClock을 Clock trait에 연결하는 어댑터.
//!
//! 시뮬레이터가 소유한 `Arc<Mutex<VirtualClock>>`을 공유하여, LuaPolicy가
//! 가상 시계를 통해 시간을 읽을 수 있도록 한다.
//!
//! - `Simulator`만이 `advance()`/`drain_until()`을 호출한다 (쓰기).
//! - `LuaPolicy`는 이 핸들을 통해 `now()`만 호출한다 (읽기).

#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::clock::{Clock, LogicalInstant};

use super::clock::VirtualClock;

/// 시뮬레이터의 `VirtualClock`을 `Clock` trait에 노출하는 어댑터.
///
/// `Arc<Mutex<VirtualClock>>`을 공유하므로, 시뮬레이터가 `advance()`를 호출하면
/// 이 핸들을 통해 `LuaPolicy`에서도 갱신된 시각을 읽을 수 있다.
#[derive(Clone)]
pub struct VirtualClockHandle {
    inner: Arc<Mutex<VirtualClock>>,
    /// `now_system()` 기준점. 테스트의 결정론적 재현을 위해 UNIX_EPOCH 고정.
    start_epoch: SystemTime,
}

impl VirtualClockHandle {
    /// 시뮬레이터가 소유한 VirtualClock을 공유한다.
    pub fn new(shared: Arc<Mutex<VirtualClock>>) -> Self {
        Self {
            inner: shared,
            start_epoch: SystemTime::UNIX_EPOCH,
        }
    }

    /// 내부 `Arc<Mutex<VirtualClock>>`을 clone하여 반환한다 (Simulator 재접근용).
    pub fn virtual_clock(&self) -> Arc<Mutex<VirtualClock>> {
        Arc::clone(&self.inner)
    }
}

impl Clock for VirtualClockHandle {
    fn now(&self) -> LogicalInstant {
        let vc = self.inner.lock().unwrap();
        LogicalInstant::from_duration_since_start(vc.now())
    }

    fn now_system(&self) -> SystemTime {
        // 결정론적 시각: UNIX_EPOCH + virtual_now
        let vc = self.inner.lock().unwrap();
        self.start_epoch + vc.now()
    }
}

// ─────────────────────────────────────────────────────────
// 단위 테스트
// ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_virtual_clock_handle_reflects_advance() {
        let clock = Arc::new(Mutex::new(VirtualClock::new()));
        let handle = VirtualClockHandle::new(Arc::clone(&clock));

        // 초기값: ZERO
        assert_eq!(
            handle.now(),
            LogicalInstant::from_duration_since_start(Duration::ZERO)
        );

        // advance 후 반영
        clock.lock().unwrap().advance(Duration::from_secs(5));
        assert_eq!(
            handle.now(),
            LogicalInstant::from_duration_since_start(Duration::from_secs(5))
        );
    }

    #[test]
    fn test_virtual_clock_handle_shared_arc() {
        let clock = Arc::new(Mutex::new(VirtualClock::new()));
        let handle1 = VirtualClockHandle::new(Arc::clone(&clock));
        let handle2 = VirtualClockHandle::new(Arc::clone(&clock));

        clock.lock().unwrap().advance(Duration::from_millis(500));

        // 동일 Arc → 두 핸들 모두 같은 값
        assert_eq!(handle1.now(), handle2.now());
        assert_eq!(
            handle1.now(),
            LogicalInstant::from_duration_since_start(Duration::from_millis(500))
        );
    }

    #[test]
    fn test_virtual_clock_handle_now_system_is_deterministic() {
        let clock = Arc::new(Mutex::new(VirtualClock::new()));
        let handle = VirtualClockHandle::new(Arc::clone(&clock));

        clock.lock().unwrap().advance(Duration::from_secs(10));

        let expected = SystemTime::UNIX_EPOCH + Duration::from_secs(10);
        assert_eq!(handle.now_system(), expected);
    }

    #[test]
    fn test_virtual_clock_handle_is_clock_trait_object() {
        let clock = Arc::new(Mutex::new(VirtualClock::new()));
        let handle = VirtualClockHandle::new(Arc::clone(&clock));
        // dyn Clock으로 업캐스트 가능한지 확인
        let _: Arc<dyn Clock> = Arc::new(handle);
    }

    #[test]
    fn test_virtual_clock_handle_elapsed_since() {
        let clock = Arc::new(Mutex::new(VirtualClock::new()));
        let handle = VirtualClockHandle::new(Arc::clone(&clock));

        let t0 = handle.now(); // ZERO

        clock.lock().unwrap().advance(Duration::from_secs(3));

        let elapsed = handle.elapsed_since(t0);
        assert_eq!(elapsed, Duration::from_secs(3));
    }
}
