//! Fallback decider for LuaPolicy — Lua VM 오동작 시 로컬 규칙 기반 fallback 처리.

use argus_shared::{EngineCommand, Level, SystemSignal};

/// 내장 fallback 정책 — Lua VM 없이 신호 레벨만으로 커맨드를 생성한다.
pub fn fallback_decide(signal: &SystemSignal) -> Vec<EngineCommand> {
    match signal {
        SystemSignal::MemoryPressure { level, .. } => match level {
            Level::Normal => vec![EngineCommand::RestoreDefaults],
            Level::Warning => vec![EngineCommand::KvEvictSliding { keep_ratio: 0.85 }],
            Level::Critical => vec![EngineCommand::KvEvictSliding { keep_ratio: 0.50 }],
            Level::Emergency => vec![
                EngineCommand::KvEvictSliding { keep_ratio: 0.25 },
                EngineCommand::SetTargetTbt { target_ms: 500 },
            ],
        },
        SystemSignal::ThermalAlert { level, .. } => match level {
            Level::Normal | Level::Warning => vec![],
            Level::Critical => vec![EngineCommand::Throttle { delay_ms: 100 }],
            Level::Emergency => vec![EngineCommand::Throttle { delay_ms: 200 }],
        },
        SystemSignal::ComputeGuidance { level, .. } => match level {
            Level::Normal | Level::Warning => vec![],
            Level::Critical => vec![EngineCommand::Throttle { delay_ms: 50 }],
            Level::Emergency => vec![EngineCommand::Throttle { delay_ms: 150 }],
        },
        SystemSignal::EnergyConstraint { .. } => vec![],
    }
}
