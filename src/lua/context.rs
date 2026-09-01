//! The `ctx` table handed to the Lua `decide(ctx)` call.
//!
//! Everything the policy is allowed to see, and nothing else. The split is deliberate:
//! `pressure` and `trigger` are platform-independent and computed in Rust, `engine` is
//! what the heartbeat reported, and `signal` is the raw reading behind `pressure` for a
//! script that wants to normalize differently.
//!
//! What used to be here and is not any more: a `relief` table of per-action coefficients
//! keyed by ten hard-coded command names, a QCF penalty weight, an `available` action
//! list, and `is_joint_valid` for checking exclusion groups. All four existed to rank a
//! set of named actions. There is one command now, so the ranking — and the vocabulary it
//! ranked over — went with it.

use mlua::{Lua, Result as LuaResult, Table};

use crate::policy::common::state::{Pressure6D, SignalState, TriggerState};
use argus_shared::{EngineState, EngineStatus, Phase};

/// Build the `ctx` table for one `decide` call.
pub fn build_ctx(
    lua: &Lua,
    pressure: &Pressure6D,
    trigger: &TriggerState,
    signals: &SignalState,
    engine: Option<&EngineStatus>,
) -> LuaResult<Table> {
    let ctx = lua.create_table()?;

    let p = lua.create_table()?;
    p.set("gpu", pressure.gpu)?;
    p.set("cpu", pressure.cpu)?;
    p.set("memory", pressure.memory)?;
    p.set("thermal", pressure.thermal)?;
    p.set("latency", pressure.latency)?;
    p.set("main_app", pressure.main_app)?;
    ctx.set("pressure", p)?;

    let t = lua.create_table()?;
    t.set("tbt_degraded", trigger.tbt_degraded)?;
    t.set("mem_low", trigger.mem_low)?;
    t.set("temp_high", trigger.temp_high)?;
    ctx.set("trigger", t)?;

    ctx.set("engine", engine_table(lua, engine)?)?;

    let s = lua.create_table()?;
    let mem = lua.create_table()?;
    mem.set("available", signals.mem_available)?;
    mem.set("total", signals.mem_total)?;
    s.set("memory", mem)?;
    let compute = lua.create_table()?;
    compute.set("cpu_pct", signals.cpu_pct)?;
    compute.set("gpu_pct", signals.gpu_pct)?;
    s.set("compute", compute)?;
    let thermal = lua.create_table()?;
    thermal.set("temp_c", signals.temp_mc as f64 / 1000.0)?;
    thermal.set("throttling", signals.throttling)?;
    s.set("thermal", thermal)?;
    ctx.set("signal", s)?;

    Ok(ctx)
}

/// The heartbeat, or an explicit "nothing heard yet".
///
/// `seen` is the load-bearing field. Without it a manager that has never received a
/// heartbeat is indistinguishable from one talking to an engine whose cache is empty:
/// both read `kv_cache_bytes == 0`. That matters more in v2 than it did before, because
/// a KV budget is a fraction of `kv_cache_budget_bytes` — a script that divides by it
/// without checking `seen` produces `0/0`, and a non-finite budget does not survive
/// serialization at all (serde writes it as `null`, which will not parse back).
fn engine_table(lua: &Lua, engine: Option<&EngineStatus>) -> LuaResult<Table> {
    let e = lua.create_table()?;
    match engine {
        Some(st) => {
            e.set("seen", true)?;
            e.set("kv_cache_bytes", st.kv_cache_bytes)?;
            e.set("kv_cache_budget_bytes", st.kv_cache_budget_bytes)?;
            e.set("kv_cache_tokens", st.kv_cache_tokens)?;
            e.set("tbt_ms", st.tbt_ms)?;
            e.set(
                "phase",
                match st.phase {
                    Phase::Idle => "idle",
                    Phase::Prefill => "prefill",
                    Phase::Decode => "decode",
                },
            )?;
            e.set(
                "state",
                match st.state {
                    EngineState::Idle => "idle",
                    EngineState::Running => "running",
                    EngineState::Suspended => "suspended",
                },
            )?;
            // Convenience: the fraction of the uncompressed footprint currently resident,
            // which is the quantity a budget is expressed against. `nil` rather than a
            // fabricated 0 or 1 when the denominator is not known yet.
            if st.kv_cache_budget_bytes > 0 {
                e.set(
                    "kv_fill",
                    st.kv_cache_bytes as f64 / st.kv_cache_budget_bytes as f64,
                )?;
            }
        }
        None => {
            e.set("seen", false)?;
        }
    }
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status() -> EngineStatus {
        EngineStatus {
            kv_cache_bytes: 1024,
            kv_cache_budget_bytes: 4096,
            kv_cache_tokens: 32,
            tbt_ms: 12.5,
            phase: Phase::Decode,
            state: EngineState::Running,
        }
    }

    fn ctx_with(engine: Option<&EngineStatus>) -> (Lua, Table) {
        let lua = Lua::new();
        let ctx = build_ctx(
            &lua,
            &Pressure6D {
                gpu: 0.1,
                cpu: 0.2,
                memory: 0.3,
                thermal: 0.4,
                latency: 0.5,
                main_app: 0.6,
            },
            &TriggerState {
                tbt_degraded: true,
                mem_low: false,
                temp_high: true,
            },
            &SignalState {
                cpu_pct: 20.0,
                gpu_pct: 10.0,
                mem_available: 700,
                mem_total: 1000,
                temp_mc: 55_000,
                throttling: false,
            },
            engine,
        )
        .unwrap();
        (lua, ctx)
    }

    #[test]
    fn pressure_trigger_and_signal_are_exposed() {
        let (_lua, ctx) = ctx_with(None);
        let p: Table = ctx.get("pressure").unwrap();
        assert_eq!(p.get::<f32>("memory").unwrap(), 0.3);
        let t: Table = ctx.get("trigger").unwrap();
        assert!(t.get::<bool>("tbt_degraded").unwrap());
        assert!(!t.get::<bool>("mem_low").unwrap());
        let s: Table = ctx.get("signal").unwrap();
        let thermal: Table = s.get("thermal").unwrap();
        assert_eq!(thermal.get::<f64>("temp_c").unwrap(), 55.0);
    }

    #[test]
    fn engine_table_carries_the_heartbeat() {
        let st = status();
        let (_lua, ctx) = ctx_with(Some(&st));
        let e: Table = ctx.get("engine").unwrap();
        assert!(e.get::<bool>("seen").unwrap());
        assert_eq!(e.get::<u64>("kv_cache_budget_bytes").unwrap(), 4096);
        assert_eq!(e.get::<String>("phase").unwrap(), "decode");
        assert_eq!(e.get::<f64>("kv_fill").unwrap(), 0.25);
    }

    /// A script must be able to tell "no heartbeat" from "empty cache" — both would read
    /// zero bytes, and dividing by the missing denominator yields a budget that cannot be
    /// sent.
    #[test]
    fn no_heartbeat_is_distinguishable_from_an_empty_cache() {
        let (_lua, ctx) = ctx_with(None);
        let e: Table = ctx.get("engine").unwrap();
        assert!(!e.get::<bool>("seen").unwrap());
        assert!(
            e.get::<Option<u64>>("kv_cache_budget_bytes")
                .unwrap()
                .is_none(),
            "no fabricated denominator"
        );
        assert!(e.get::<Option<f64>>("kv_fill").unwrap().is_none());
    }

    #[test]
    fn kv_fill_is_absent_when_the_denominator_is_unknown() {
        let st = EngineStatus {
            kv_cache_budget_bytes: 0,
            ..status()
        };
        let (_lua, ctx) = ctx_with(Some(&st));
        let e: Table = ctx.get("engine").unwrap();
        assert!(e.get::<bool>("seen").unwrap());
        assert!(
            e.get::<Option<f64>>("kv_fill").unwrap().is_none(),
            "0/0 must not reach the script as a number"
        );
    }
}
