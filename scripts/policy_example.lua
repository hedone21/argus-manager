-- policy_example.lua — the smallest policy that does something.
--
-- Compress in proportion to memory pressure and release when it clears. Start here when
-- porting to a new platform, then move to policy_default.lua once you know which axis
-- binds first on your device.
--
-- Deliberately stateless, so it emits on every tick. That is fine here because it only ever
-- names three values and the manager suppresses a byte-identical repeat -- but note that a
-- signal sitting on 0.85 will alternate 0.70/0.30, and each SWITCH costs the engine a full
-- scoring pass. A policy with a continuous ramp must not be written this way: see the
-- `applied` ratchet in policy_default.lua, which keeps one relief window monotone.

POLICY_META = { name = "example", version = "3.0.0" }

function decide(ctx)
  if not ctx.engine.seen then
    return nil
  end
  local m = ctx.pressure.memory
  if m < 0.70 then
    return { type = "restore_defaults" }
  elseif m < 0.85 then
    return { type = "kv.compress", budget = 0.70 }
  else
    return { type = "kv.compress", budget = 0.30 }
  end
end
