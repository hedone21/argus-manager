-- partition_adaptive.lua
--
-- 목적: latency / compute pressure 상승 시 partition_ratio를 낮춰
--       memory bandwidth 경합을 해소한다.
--
-- ctx.coef.pressure.latency (0.0~1.0): latency 압박 수치
-- ctx.engine.partition_ratio: 현재 분할 비율
--   pressure_latency ≥ 0.6 → SetPartitionRatio(0.0) — 단일 백엔드로 전환
--   pressure_latency ≥ 0.3 → SetPartitionRatio(0.25) — 분할 비율 축소

function decide(ctx)
  local cmds = {}
  local pressure_lat = ctx.coef and ctx.coef.pressure and ctx.coef.pressure.latency or 0.0
  local cur_ratio    = ctx.engine and ctx.engine.partition_ratio or 0.0

  if pressure_lat >= 0.6 and cur_ratio > 0.0 then
    table.insert(cmds, { type = "set_partition_ratio", ratio = 0.0 })
  elseif pressure_lat >= 0.3 and cur_ratio > 0.25 then
    table.insert(cmds, { type = "set_partition_ratio", ratio = 0.25 })
  end

  return cmds
end
