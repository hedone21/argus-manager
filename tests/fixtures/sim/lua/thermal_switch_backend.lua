-- thermal_switch_backend.lua
--
-- 목적: thermal pressure 상승 시 GPU→CPU 백엔드 전환 또는 throttle 적용.
--   ctx.coef.pressure.thermal (0.0~1.0): 열 압박 수치
--     ≥ 0.8 → SwitchHw("cpu") — GPU 부하 완전 제거
--     ≥ 0.5 → Throttle(delay_ms=150) — 부분 완화
--     ≥ 0.2 → Throttle(delay_ms=100) — 가벼운 완화
--
-- ctx.signal.thermal.temp_c: 현재 온도(°C)
-- ctx.signal.thermal.throttling: bool

local switched_to_cpu = false

function decide(ctx)
  local cmds = {}
  local pressure_therm = ctx.coef and ctx.coef.pressure and ctx.coef.pressure.thermal or 0.0
  local temp_c         = ctx.signal and ctx.signal.thermal and ctx.signal.thermal.temp_c or 0.0

  if pressure_therm >= 0.8 then
    if not switched_to_cpu then
      table.insert(cmds, { type = "switch_hw", device = "cpu" })
      switched_to_cpu = true
    end
    table.insert(cmds, { type = "throttle", delay_ms = 200 })
  elseif pressure_therm >= 0.5 then
    if not switched_to_cpu then
      table.insert(cmds, { type = "switch_hw", device = "cpu" })
      switched_to_cpu = true
    else
      table.insert(cmds, { type = "throttle", delay_ms = 150 })
    end
  elseif pressure_therm >= 0.2 then
    table.insert(cmds, { type = "throttle", delay_ms = 100 })
  end

  return cmds
end
