use std::time::{Duration, Instant};

use crate::config::SupervisoryConfig;
use crate::types::{OperatingMode, PressureVector};

/// Supervisory Layer — pressure vector를 받아 시스템 운영 모드를 결정한다.
///
/// - 상승 전환: 즉시 (Normal → Warning 또는 Critical 직행 가능)
/// - 하강 전환: hold_time 동안 안정적으로 유지된 후에만 1단계씩 (Critical→Warning→Normal)
///
/// Hysteresis의 의도: 올라갈 때는 안전 우선으로 즉시 반응하고, 내려갈 때는
/// 잦은 모드 전환(flickering)을 방지하여 irreversible lossy 액션의 반복 적용을 막는다.
pub struct SupervisoryLayer {
    mode: OperatingMode,
    warning_threshold: f32,
    critical_threshold: f32,
    warning_release: f32,
    critical_release: f32,
    hold_time: Duration,
    /// 하강 안정화 시작 시각. 상승 또는 현상 유지 시 None.
    stable_since: Option<Instant>,
}

impl SupervisoryLayer {
    pub fn new(config: &SupervisoryConfig) -> Self {
        Self {
            mode: OperatingMode::Normal,
            warning_threshold: config.warning_threshold,
            critical_threshold: config.critical_threshold,
            warning_release: config.warning_release,
            critical_release: config.critical_release,
            hold_time: Duration::from_secs_f32(config.hold_time_secs),
            stable_since: None,
        }
    }

    /// 현재 운영 모드를 반환한다.
    pub fn mode(&self) -> OperatingMode {
        self.mode
    }

    /// Pressure vector를 평가하여 새 운영 모드를 결정하고 반환한다.
    ///
    /// 내부적으로 `Instant::now()`를 사용한다. 테스트에서는 `evaluate_at`을 사용할 것.
    pub fn evaluate(&mut self, pressure: &PressureVector) -> OperatingMode {
        self.evaluate_at(pressure, Instant::now())
    }

    /// 시간을 주입할 수 있는 내부 메서드 (테스트 용이성).
    pub fn evaluate_at(&mut self, pressure: &PressureVector, now: Instant) -> OperatingMode {
        let peak = pressure.max();
        self.mode = self.next_mode(peak, now);
        self.mode
    }

    fn next_mode(&mut self, peak: f32, now: Instant) -> OperatingMode {
        match self.mode {
            OperatingMode::Normal => {
                // 상승: 즉시, Critical 직행 가능
                if peak >= self.critical_threshold {
                    self.stable_since = None;
                    OperatingMode::Critical
                } else if peak >= self.warning_threshold {
                    self.stable_since = None;
                    OperatingMode::Warning
                } else {
                    // 현상 유지
                    self.stable_since = None;
                    OperatingMode::Normal
                }
            }
            OperatingMode::Warning => {
                if peak >= self.critical_threshold {
                    // 상승: 즉시
                    self.stable_since = None;
                    OperatingMode::Critical
                } else if peak >= self.warning_threshold {
                    // Warning 내 유지 — 상승 방향 또는 현상 유지이므로 stable_since 리셋
                    self.stable_since = None;
                    OperatingMode::Warning
                } else if peak < self.warning_release {
                    // 하강 후보: hold_time 확인
                    match self.stable_since {
                        None => {
                            self.stable_since = Some(now);
                            OperatingMode::Warning
                        }
                        Some(since) => {
                            if now.duration_since(since) >= self.hold_time {
                                self.stable_since = None;
                                OperatingMode::Normal
                            } else {
                                OperatingMode::Warning
                            }
                        }
                    }
                } else {
                    // warning_release <= peak < warning_threshold: 아직 충분히 내려가지 않음
                    self.stable_since = None;
                    OperatingMode::Warning
                }
            }
            OperatingMode::Critical => {
                if peak >= self.critical_threshold {
                    // Critical 내 유지 또는 재상승
                    self.stable_since = None;
                    OperatingMode::Critical
                } else if peak < self.critical_release {
                    // 하강 후보: hold_time 확인 (1단계씩만 — Warning으로)
                    match self.stable_since {
                        None => {
                            self.stable_since = Some(now);
                            OperatingMode::Critical
                        }
                        Some(since) => {
                            if now.duration_since(since) >= self.hold_time {
                                self.stable_since = None;
                                OperatingMode::Warning
                            } else {
                                OperatingMode::Critical
                            }
                        }
                    }
                } else {
                    // critical_release <= peak < critical_threshold: 하강 불충분, stable_since 리셋
                    self.stable_since = None;
                    OperatingMode::Critical
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn default_config() -> SupervisoryConfig {
        SupervisoryConfig::default()
    }

    fn fast_config() -> SupervisoryConfig {
        SupervisoryConfig {
            warning_threshold: 0.4,
            critical_threshold: 0.7,
            warning_release: 0.25,
            critical_release: 0.50,
            hold_time_secs: 0.001, // 1ms — 테스트에서 쉽게 경과시킬 수 있다.
        }
    }

    fn pressure(compute: f32, memory: f32, thermal: f32) -> PressureVector {
        PressureVector {
            compute,
            memory,
            thermal,
        }
    }

    fn low() -> PressureVector {
        pressure(0.1, 0.1, 0.1) // peak = 0.1 < warning_threshold(0.4)
    }

    fn warning_level() -> PressureVector {
        pressure(0.5, 0.1, 0.1) // peak = 0.5, warning(0.4) <= peak < critical(0.7)
    }

    fn critical_level() -> PressureVector {
        pressure(0.8, 0.1, 0.1) // peak = 0.8 >= critical(0.7)
    }

    fn below_warning_release() -> PressureVector {
        pressure(0.1, 0.2, 0.1) // peak = 0.2 < warning_release(0.25)
    }

    fn below_critical_release() -> PressureVector {
        pressure(0.3, 0.3, 0.1) // peak = 0.3 < critical_release(0.50)
    }

    /// peak < warning_threshold → Normal 유지
    #[test]
    fn test_supervisory_normal_below_threshold() {
        let mut s = SupervisoryLayer::new(&default_config());
        let mode = s.evaluate(&low());
        assert_eq!(mode, OperatingMode::Normal);
    }

    /// peak >= warning_threshold → 즉시 Warning
    #[test]
    fn test_supervisory_escalation_to_warning() {
        let mut s = SupervisoryLayer::new(&default_config());
        let mode = s.evaluate(&warning_level());
        assert_eq!(mode, OperatingMode::Warning);
    }

    /// peak >= critical_threshold → 즉시 Critical
    #[test]
    fn test_supervisory_escalation_to_critical() {
        let mut s = SupervisoryLayer::new(&default_config());
        let mode = s.evaluate(&critical_level());
        assert_eq!(mode, OperatingMode::Critical);
    }

    /// Normal에서 바로 Critical 도달 가능 (중간 단계 없음)
    #[test]
    fn test_supervisory_skip_to_critical() {
        let mut s = SupervisoryLayer::new(&default_config());
        assert_eq!(s.mode(), OperatingMode::Normal);
        let mode = s.evaluate(&critical_level());
        assert_eq!(
            mode,
            OperatingMode::Critical,
            "Normal should jump directly to Critical"
        );
    }

    /// hold_time 미충족 시 Critical → Critical 유지
    #[test]
    fn test_supervisory_deescalation_needs_hold_time() {
        let config = SupervisoryConfig {
            hold_time_secs: 1000.0, // 사실상 무한대
            ..default_config()
        };
        let mut s = SupervisoryLayer::new(&config);
        // Critical 상태로 전환
        s.evaluate(&critical_level());
        assert_eq!(s.mode(), OperatingMode::Critical);

        // peak가 critical_release 미만이지만 hold_time 미충족
        let mode = s.evaluate(&below_critical_release());
        assert_eq!(
            mode,
            OperatingMode::Critical,
            "should stay Critical until hold_time elapses"
        );
    }

    /// hold_time 경과 후 Critical → Warning
    #[test]
    fn test_supervisory_deescalation_after_hold_time() {
        let config = fast_config();
        let mut s = SupervisoryLayer::new(&config);

        // Critical 상태로 전환
        s.evaluate(&critical_level());
        assert_eq!(s.mode(), OperatingMode::Critical);

        // 하강 첫 번째 평가 → stable_since 설정, Critical 유지
        let mode_first = s.evaluate(&below_critical_release());
        assert_eq!(
            mode_first,
            OperatingMode::Critical,
            "first check should stay Critical"
        );

        // hold_time(1ms) 이상 경과
        thread::sleep(Duration::from_millis(5));

        // 두 번째 평가 → hold_time 경과 → Warning
        let mode = s.evaluate(&below_critical_release());
        assert_eq!(
            mode,
            OperatingMode::Warning,
            "after hold_time, should transition Critical→Warning"
        );
    }

    /// 하강 중 peak 재상승 → stable_since 리셋되어 hold_time 재시작
    #[test]
    fn test_supervisory_deescalation_reset_on_spike() {
        let config = fast_config();
        let mut s = SupervisoryLayer::new(&config);

        // Critical 상태로 전환
        s.evaluate(&critical_level());

        // 하강 시작 (stable_since 설정)
        s.evaluate(&below_critical_release());
        // 재상승으로 stable_since 리셋
        s.evaluate(&critical_level());

        // hold_time 경과시켜도 방금 전에 stable_since가 리셋되었으므로
        // 새로 하강을 시작하지 않으면 아직 Critical이어야 한다.
        let mode = s.evaluate(&critical_level());
        assert_eq!(
            mode,
            OperatingMode::Critical,
            "spike should reset stable_since and keep Critical"
        );
    }

    /// Critical→Warning→Normal 순서로만 하강 가능 (Critical→Normal 직행 불가)
    #[test]
    fn test_supervisory_stepwise_deescalation() {
        let config = fast_config();
        let mut s = SupervisoryLayer::new(&config);

        // 1. Critical 진입
        s.evaluate(&critical_level());
        assert_eq!(s.mode(), OperatingMode::Critical);

        // 2. Critical→Warning: stable_since 설정 후 hold_time 경과
        s.evaluate(&below_critical_release()); // stable_since 설정
        thread::sleep(Duration::from_millis(5));
        let mode = s.evaluate(&below_critical_release());
        assert_eq!(mode, OperatingMode::Warning, "step 1: Critical→Warning");

        // 3. Warning→Normal: stable_since 설정 후 hold_time 경과
        s.evaluate(&below_warning_release()); // stable_since 설정
        thread::sleep(Duration::from_millis(5));
        let mode = s.evaluate(&below_warning_release());
        assert_eq!(mode, OperatingMode::Normal, "step 2: Warning→Normal");
    }
}
