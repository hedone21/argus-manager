-- memory_evict_graduated.lua
--
-- 목적: memory pressure에 따라 KV evict target_ratio를 단계적으로 조정.
--   ctx.coef.pressure.memory (0.0~1.0): 메모리 압박 수치
--     ≥ 0.7 → kv_evict_sliding(keep_ratio=0.5)  — 강력한 eviction
--     ≥ 0.4 → kv_evict_sliding(keep_ratio=0.8)  — 완만한 eviction
--
-- ctx.coef.trigger.mem_low: bool — 메모리 부족 트리거
-- ctx.signal.memory.available / .total: 바이트 단위 실측값
--
-- 참고: LuaPolicy.build_ctx()는 level 문자열 대신 normalized pressure를 노출함.

function decide(ctx)
  local cmds = {}
  local pressure_mem = ctx.coef and ctx.coef.pressure and ctx.coef.pressure.memory or 0.0
  local mem_low      = ctx.coef and ctx.coef.trigger   and ctx.coef.trigger.mem_low  or false

  if pressure_mem >= 0.7 or mem_low then
    table.insert(cmds, { type = "kv.evict_sliding", keep_ratio = 0.5 })
  elseif pressure_mem >= 0.4 then
    table.insert(cmds, { type = "kv.evict_sliding", keep_ratio = 0.8 })
  end

  return cmds
end
