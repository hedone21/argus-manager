use crate::signal::Level;

/// Direction of threshold evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Higher values are worse (temperature, CPU usage).
    Ascending,
    /// Lower values are worse (available memory, battery %).
    Descending,
}

/// Threshold configuration for a single metric.
#[derive(Debug, Clone)]
pub struct Thresholds {
    pub warning: f64,
    pub critical: f64,
    /// Set to `f64::MAX` (ascending) or `f64::MIN` (descending) to disable Emergency level.
    pub emergency: f64,
    pub hysteresis: f64,
}

/// Reusable hysteresis-based threshold evaluator.
///
/// Escalation is immediate and can skip levels.
/// Recovery requires crossing the recovery threshold (= threshold +/- hysteresis).
pub struct ThresholdEvaluator {
    direction: Direction,
    thresholds: Thresholds,
    current: Level,
}

impl ThresholdEvaluator {
    pub fn new(direction: Direction, thresholds: Thresholds) -> Self {
        Self {
            direction,
            thresholds,
            current: Level::Normal,
        }
    }

    /// Evaluate a new value. Returns `Some(new_level)` if the level changed.
    pub fn evaluate(&mut self, value: f64) -> Option<Level> {
        let new_level = match self.direction {
            Direction::Ascending => level_ascending(
                value,
                self.current,
                self.thresholds.warning,
                self.thresholds.critical,
                self.thresholds.emergency,
                self.thresholds.hysteresis,
            ),
            Direction::Descending => level_descending(
                value,
                self.current,
                self.thresholds.warning,
                self.thresholds.critical,
                self.thresholds.emergency,
                self.thresholds.hysteresis,
            ),
        };

        if new_level != self.current {
            self.current = new_level;
            Some(new_level)
        } else {
            None
        }
    }

    pub fn level(&self) -> Level {
        self.current
    }
}

/// Evaluate level for "higher is worse" metrics.
fn level_ascending(
    value: f64,
    current: Level,
    warning_up: f64,
    critical_up: f64,
    emergency_up: f64,
    hysteresis: f64,
) -> Level {
    let warning_down = warning_up - hysteresis;
    let critical_down = critical_up - hysteresis;
    let emergency_down = emergency_up - hysteresis;

    // Escalation: jump to highest triggered level
    if value >= emergency_up && current < Level::Emergency {
        return Level::Emergency;
    }
    if value >= critical_up && current < Level::Critical {
        return Level::Critical;
    }
    if value >= warning_up && current < Level::Warning {
        return Level::Warning;
    }

    // Recovery: drop when value convincingly crosses recovery threshold
    match current {
        Level::Emergency if value < emergency_down => {
            if value < warning_down {
                Level::Normal
            } else if value < critical_down {
                Level::Warning
            } else {
                Level::Critical
            }
        }
        Level::Critical if value < critical_down => {
            if value < warning_down {
                Level::Normal
            } else {
                Level::Warning
            }
        }
        Level::Warning if value < warning_down => Level::Normal,
        _ => current,
    }
}

/// Evaluate level for "lower is worse" metrics.
fn level_descending(
    value: f64,
    current: Level,
    warning_below: f64,
    critical_below: f64,
    emergency_below: f64,
    hysteresis: f64,
) -> Level {
    let warning_up = warning_below + hysteresis;
    let critical_up = critical_below + hysteresis;
    let emergency_up = emergency_below + hysteresis;

    // Escalation: value drops below thresholds
    if value <= emergency_below && current < Level::Emergency {
        return Level::Emergency;
    }
    if value <= critical_below && current < Level::Critical {
        return Level::Critical;
    }
    if value <= warning_below && current < Level::Warning {
        return Level::Warning;
    }

    // Recovery: value rises above recovery thresholds
    match current {
        Level::Emergency if value > emergency_up => {
            if value > warning_up {
                Level::Normal
            } else if value > critical_up {
                Level::Warning
            } else {
                Level::Critical
            }
        }
        Level::Critical if value > critical_up => {
            if value > warning_up {
                Level::Normal
            } else {
                Level::Warning
            }
        }
        Level::Warning if value > warning_up => Level::Normal,
        _ => current,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Ascending tests ---

    #[test]
    fn ascending_escalation_path() {
        let mut e = ThresholdEvaluator::new(
            Direction::Ascending,
            Thresholds {
                warning: 60.0,
                critical: 75.0,
                emergency: 85.0,
                hysteresis: 5.0,
            },
        );

        assert!(e.evaluate(50.0).is_none()); // Normal stays Normal
        assert_eq!(e.evaluate(65.0), Some(Level::Warning));
        assert_eq!(e.evaluate(76.0), Some(Level::Critical));
        assert_eq!(e.evaluate(90.0), Some(Level::Emergency));
    }

    #[test]
    fn ascending_skip_to_emergency() {
        let mut e = ThresholdEvaluator::new(
            Direction::Ascending,
            Thresholds {
                warning: 60.0,
                critical: 75.0,
                emergency: 85.0,
                hysteresis: 5.0,
            },
        );
        assert_eq!(e.evaluate(90.0), Some(Level::Emergency));
    }

    #[test]
    fn ascending_hysteresis_prevents_oscillation() {
        let mut e = ThresholdEvaluator::new(
            Direction::Ascending,
            Thresholds {
                warning: 60.0,
                critical: 75.0,
                emergency: 85.0,
                hysteresis: 5.0,
            },
        );

        e.evaluate(65.0); // → Warning
        assert_eq!(e.level(), Level::Warning);

        assert!(e.evaluate(58.0).is_none()); // Still Warning (need < 55)
        assert_eq!(e.evaluate(54.0), Some(Level::Normal));
    }

    #[test]
    fn ascending_multi_level_recovery() {
        let mut e = ThresholdEvaluator::new(
            Direction::Ascending,
            Thresholds {
                warning: 60.0,
                critical: 75.0,
                emergency: 85.0,
                hysteresis: 5.0,
            },
        );

        e.evaluate(90.0); // → Emergency
        assert_eq!(e.evaluate(40.0), Some(Level::Normal)); // Full recovery
    }

    #[test]
    fn ascending_stay_in_hysteresis_zone() {
        let level = level_ascending(57.0, Level::Warning, 60.0, 75.0, 85.0, 5.0);
        assert_eq!(level, Level::Warning);
    }

    // --- Descending tests ---

    #[test]
    fn descending_escalation_path() {
        let mut e = ThresholdEvaluator::new(
            Direction::Descending,
            Thresholds {
                warning: 40.0,
                critical: 20.0,
                emergency: 10.0,
                hysteresis: 5.0,
            },
        );

        assert!(e.evaluate(50.0).is_none()); // Normal stays Normal
        assert_eq!(e.evaluate(35.0), Some(Level::Warning));
        assert_eq!(e.evaluate(15.0), Some(Level::Critical));
        assert_eq!(e.evaluate(5.0), Some(Level::Emergency));
    }

    #[test]
    fn descending_skip_to_emergency() {
        let level = level_descending(3.0, Level::Normal, 40.0, 20.0, 10.0, 5.0);
        assert_eq!(level, Level::Emergency);
    }

    #[test]
    fn descending_hysteresis_prevents_oscillation() {
        let mut e = ThresholdEvaluator::new(
            Direction::Descending,
            Thresholds {
                warning: 40.0,
                critical: 20.0,
                emergency: 10.0,
                hysteresis: 5.0,
            },
        );

        e.evaluate(35.0); // → Warning
        assert!(e.evaluate(42.0).is_none()); // Still Warning (need > 45)
        assert_eq!(e.evaluate(46.0), Some(Level::Normal));
    }

    #[test]
    fn descending_multi_level_recovery() {
        let level = level_descending(80.0, Level::Emergency, 40.0, 20.0, 10.0, 5.0);
        assert_eq!(level, Level::Normal);
    }

    #[test]
    fn descending_step_recovery() {
        let mut e = ThresholdEvaluator::new(
            Direction::Descending,
            Thresholds {
                warning: 40.0,
                critical: 20.0,
                emergency: 10.0,
                hysteresis: 5.0,
            },
        );

        e.evaluate(5.0); // → Emergency
        assert_eq!(e.evaluate(16.0), Some(Level::Critical)); // > 15 but < 25
        assert_eq!(e.evaluate(30.0), Some(Level::Warning)); // > 25 but < 45
        assert_eq!(e.evaluate(50.0), Some(Level::Normal)); // > 45
    }

    #[test]
    fn no_emergency_level() {
        let mut e = ThresholdEvaluator::new(
            Direction::Ascending,
            Thresholds {
                warning: 70.0,
                critical: 90.0,
                emergency: f64::MAX,
                hysteresis: 5.0,
            },
        );

        assert_eq!(e.evaluate(99.0), Some(Level::Critical));
        assert_eq!(e.level(), Level::Critical); // Cannot reach Emergency
    }
}
