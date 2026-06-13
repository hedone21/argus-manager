-- memory_and_thermal_combined.lua
--
-- 목적: memory + thermal 두 압박을 동시에 처리 (composition 테스트용).
--
-- 우선순위 (높은 순):
--   1. thermal pressure ≥ 0.8 → SwitchHw("cpu") + evict 0.5
--   2. thermal pressure ≥ 0.5 → Throttle(150) + evict 0.7
--   3. memory pressure ≥ 0.7 → evict 0.5
--   4. memory pressure ≥ 0.4 → evict 0.8
--   5. thermal pressure ≥ 0.2 → Throttle(100)만
--
-- 복수 조건 동시 충족 시 높은 우선순위 하나만 적용.

local switched_to_cpu = false

function decide(ctx)
  local cmds = {}
  local p_mem   = ctx.coef and ctx.coef.pressure and ctx.coef.pressure.memory  or 0.0
  local p_therm = ctx.coef and ctx.coef.pressure and ctx.coef.pressure.thermal or 0.0

  if p_therm >= 0.8 then
    if not switched_to_cpu then
      table.insert(cmds, { type = "switch_hw",        device     = "cpu" })
      switched_to_cpu = true
    end
    table.insert(cmds, { type = "kv.evict_sliding", keep_ratio = 0.5  })
  elseif p_therm >= 0.5 then
    table.insert(cmds, { type = "throttle",         delay_ms   = 150  })
    table.insert(cmds, { type = "kv.evict_sliding", keep_ratio = 0.7  })
  elseif p_mem >= 0.7 then
    table.insert(cmds, { type = "kv.evict_sliding", keep_ratio = 0.5  })
  elseif p_mem >= 0.4 then
    table.insert(cmds, { type = "kv.evict_sliding", keep_ratio = 0.8  })
  elseif p_therm >= 0.2 then
    table.insert(cmds, { type = "throttle",         delay_ms   = 100  })
  end

  return cmds
end
