//! Context Builder for LuaPolicy — Lua VM 전달용 context 테이블 조립.

use crate::lua_policy::{LINUCB_FEATURE_DIM, LuaPolicy};
use mlua::{Result as LuaResult, Table};

/// 현재 엔진 상태 + signal_state에서 13차원 feature vector를 빌드.
pub fn build_feature_vec(policy: &LuaPolicy) -> [f32; LINUCB_FEATURE_DIM] {
    let mut phi = [0.0f32; LINUCB_FEATURE_DIM];

    if let Some(ref status) = policy.engine_state {
        // 0: KV_OCCUPANCY
        phi[0] = status.kv_cache_utilization.clamp(0.0, 1.0);
        // 1: IS_GPU
        phi[1] = if status.active_device.to_lowercase().contains("opencl") {
            1.0
        } else {
            0.0
        };
        // 2 & 6: TOKEN_PROGRESS / TOKENS_GEN_NORM (같은 값)
        let tok_norm = (status.tokens_generated as f32 / 2048.0).min(1.0);
        phi[2] = tok_norm;
        phi[6] = tok_norm;
        // 3: IS_PREFILL
        phi[3] = if status.phase == "prefill" { 1.0 } else { 0.0 };
        // 4: KV_DTYPE_NORM
        phi[4] = match status.kv_dtype.as_str() {
            "f32" => 0.0,
            "f16" => 0.5,
            _ => 1.0, // q4_0 등 양자화 포맷
        };
        // 7-12: ACTIVE_* flags
        for action in &status.active_actions {
            match action.as_str() {
                "switch_hw" => phi[7] = 1.0,
                "throttle" | "set_target_tbt" => phi[8] = 1.0,
                "kv_offload_disk" => phi[9] = 1.0,
                "kv.evict_h2o" | "kv.evict_sliding" | "kv.merge_d2o" => phi[10] = 1.0,
                "weight.skip" => phi[11] = 1.0,
                "kv.quant_dynamic" => phi[12] = 1.0,
                _ => {}
            }
        }
    }

    // 5: TBT_RATIO
    phi[5] = policy.trigger_engine.tbt_degradation_ratio().unwrap_or(0.0) as f32;

    phi
}

/// Build the `ctx` Lua table from current engine state.
pub fn build_ctx(policy: &LuaPolicy) -> LuaResult<Table> {
    let lua = &policy.lua;
    let ctx = lua.create_table()?;

    // ctx.engine
    let engine_tbl = lua.create_table()?;
    if let Some(ref status) = policy.engine_state {
        engine_tbl.set("device", status.active_device.as_str())?;
        engine_tbl.set("throughput", status.actual_throughput)?;
        engine_tbl.set("kv_util", status.kv_cache_utilization)?;
        engine_tbl.set("cache_tokens", status.kv_cache_tokens)?;
        engine_tbl.set("cache_bytes", status.kv_cache_bytes)?;
        engine_tbl.set("tokens_generated", status.tokens_generated)?;
        let state_str = match status.state {
            argus_shared::EngineState::Idle => "idle",
            argus_shared::EngineState::Running => "running",
            argus_shared::EngineState::Suspended => "suspended",
        };
        engine_tbl.set("state", state_str)?;
        engine_tbl.set("kv_dtype", status.kv_dtype.as_str())?;
        engine_tbl.set("skip_ratio", status.skip_ratio)?;
        engine_tbl.set("phase", status.phase.as_str())?;
        engine_tbl.set("prefill_pos", status.prefill_pos)?;
        engine_tbl.set("prefill_total", status.prefill_total)?;
        engine_tbl.set("partition_ratio", status.partition_ratio)?;
        engine_tbl.set("cpu_pct", status.self_cpu_pct)?;
        engine_tbl.set("gpu_pct", status.self_gpu_pct)?;
    } else {
        engine_tbl.set("device", "unknown")?;
        engine_tbl.set("throughput", 0.0)?;
        engine_tbl.set("kv_util", 0.0)?;
        engine_tbl.set("cache_tokens", 0)?;
        engine_tbl.set("cache_bytes", 0)?;
        engine_tbl.set("tokens_generated", 0)?;
        engine_tbl.set("state", "idle")?;
        engine_tbl.set("kv_dtype", "")?;
        engine_tbl.set("skip_ratio", 0.0)?;
        engine_tbl.set("phase", "")?;
        engine_tbl.set("prefill_pos", 0)?;
        engine_tbl.set("prefill_total", 0)?;
        engine_tbl.set("partition_ratio", 0.0)?;
        engine_tbl.set("cpu_pct", 0.0)?;
        engine_tbl.set("gpu_pct", 0.0)?;
    }
    ctx.set("engine", engine_tbl)?;

    // ctx.active
    let active_tbl = lua.create_table()?;
    if let Some(ref status) = policy.engine_state {
        for (i, action) in status.active_actions.iter().enumerate() {
            active_tbl.set(i + 1, action.as_str())?;
        }
    }
    ctx.set("active", active_tbl)?;

    // ctx.available
    let avail_tbl = lua.create_table()?;
    let avail_source: &[String] = if let Some(ref status) = policy.engine_state
        && !status.available_actions.is_empty()
    {
        &status.available_actions
    } else {
        &policy.engine_available_actions
    };
    for (i, action) in avail_source.iter().enumerate() {
        avail_tbl.set(i + 1, action.as_str())?;
    }
    ctx.set("available", avail_tbl)?;

    // ctx.signal
    let signal_tbl = lua.create_table()?;
    {
        let mem = lua.create_table()?;
        mem.set("available", policy.signal_state.mem_available)?;
        mem.set("total", policy.signal_state.mem_total)?;
        signal_tbl.set("memory", mem)?;

        let compute = lua.create_table()?;
        compute.set("cpu_pct", policy.signal_state.cpu_pct)?;
        compute.set("gpu_pct", policy.signal_state.gpu_pct)?;
        signal_tbl.set("compute", compute)?;

        let thermal = lua.create_table()?;
        thermal.set("temp_c", policy.signal_state.temp_mc as f64 / 1000.0)?;
        thermal.set("throttling", policy.signal_state.throttling)?;
        signal_tbl.set("thermal", thermal)?;
    }
    ctx.set("signal", signal_tbl)?;

    // ctx.coef
    let coef = lua.create_table()?;
    {
        let pressure = policy.signal_state.pressure_with_thermal(
            policy.adaptation_config.temp_safe_c,
            policy.adaptation_config.temp_critical_c,
            policy.trigger_engine.tbt_degradation_ratio(),
        );
        let p_tbl = lua.create_table()?;
        p_tbl.set("gpu", pressure.gpu)?;
        p_tbl.set("cpu", pressure.cpu)?;
        p_tbl.set("memory", pressure.memory)?;
        p_tbl.set("thermal", pressure.thermal)?;
        p_tbl.set("latency", pressure.latency)?;
        p_tbl.set("main_app", pressure.main_app)?;
        coef.set("pressure", p_tbl)?;

        let trigger = policy.trigger_engine.state();
        let t_tbl = lua.create_table()?;
        t_tbl.set("tbt_degraded", trigger.tbt_degraded)?;
        t_tbl.set("mem_low", trigger.mem_low)?;
        t_tbl.set("temp_high", trigger.temp_high)?;
        coef.set("trigger", t_tbl)?;

        let r_tbl = lua.create_table()?;
        let action_names = [
            "switch_hw",
            "throttle",
            "set_target_tbt",
            "weight.skip",
            "kv.evict_h2o",
            "kv.evict_sliding",
            "kv.evict_streaming",
            "kv.merge_d2o",
            "kv.quant_dynamic",
            "set_partition_ratio",
        ];
        for name in &action_names {
            let relief = policy.relief_table.predict(name);
            let ucb = policy.linucb.ucb_bonus(name, &policy.feature_state) * policy.linucb_alpha;
            let entry = lua.create_table()?;
            entry.set("gpu", relief[0])?;
            entry.set("cpu", relief[1])?;
            entry.set("memory", relief[2])?;
            entry.set("thermal", relief[3])?;
            entry.set("lat", relief[4])?;
            entry.set("qos", relief[5])?;
            entry.set("ucb_bonus", ucb)?;

            let (qcf_cost, qcf_age_s) = match policy.qcf_cache.get(*name) {
                Some(&(cost, observed_at)) => {
                    let age = policy.clock.elapsed_since(observed_at).as_secs_f64() as f32;
                    (cost, age)
                }
                None => (0.0_f32, f32::MAX),
            };
            entry.set("qcf_cost", qcf_cost)?;
            entry.set("qcf_age_s", qcf_age_s)?;
            r_tbl.set(*name, entry)?;
        }
        coef.set("relief", r_tbl)?;

        coef.set("qcf_penalty_weight", policy.qcf_penalty_weight)?;
    }
    ctx.set("coef", coef)?;

    // ctx.history
    let history_table = lua.create_table()?;
    for (i, entry) in policy.history.iter().enumerate() {
        let h = lua.create_table()?;
        h.set("at_s", entry.at_s)?;

        let p = lua.create_table()?;
        p.set("gpu", entry.pressure[0])?;
        p.set("cpu", entry.pressure[1])?;
        p.set("memory", entry.pressure[2])?;
        p.set("thermal", entry.pressure[3])?;
        p.set("latency", entry.pressure[4])?;
        p.set("main_app", entry.pressure[5])?;
        h.set("pressure", p)?;

        let acts = lua.create_table()?;
        for (j, a) in entry.active_actions.iter().enumerate() {
            acts.set(j + 1, a.as_str())?;
        }
        h.set("active", acts)?;

        history_table.set(i + 1, h)?;
    }
    ctx.set("history", history_table)?;

    // ctx.is_joint_valid(action_list)
    {
        let groups = policy.exclusion_groups.clone();
        let is_joint_valid = lua.create_function(move |_, tbl: Table| {
            let mut names: Vec<String> = Vec::new();
            for pair in tbl.pairs::<mlua::Value, String>() {
                match pair {
                    Ok((_, v)) => names.push(v),
                    Err(_) => continue,
                }
            }
            for members in groups.values() {
                let mut count = 0usize;
                for name in &names {
                    if members.contains(name) {
                        count += 1;
                    }
                    if count >= 2 {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        })?;
        ctx.set("is_joint_valid", is_joint_valid)?;
    }

    Ok(ctx)
}
