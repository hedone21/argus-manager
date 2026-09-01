-- partition_adaptive.lua
--
-- Latency-driven compression. A shorter KV cache is less attention work per token, so
-- when time-between-tokens degrades this trades context for throughput.
--
--   ctx.pressure.latency (0..1): TBT degradation against the warm-up baseline
--     >= 0.6 -> keep 50%
--     >= 0.3 -> keep 80%
--
-- The scenario this drives used to adjust a tensor-partition ratio; that command left
-- the contract, and the fixture keeps the scenario's shape (latency pressure ramping on
-- a contended device) with the one action that remains.

function decide(ctx)
  local lat = ctx.pressure and ctx.pressure.latency or 0.0

  if lat >= 0.6 then
    return { type = "kv.compress", budget = 0.5 }
  elseif lat >= 0.3 then
    return { type = "kv.compress", budget = 0.8 }
  end
  return nil
end
