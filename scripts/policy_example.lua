-- policy_example.lua — the smallest policy that does something.
--
-- Compress in proportion to memory pressure and release when it clears. Start here when
-- porting to a new platform, then move to policy_default.lua once you know which axis
-- binds first on your device.

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
