-- policy_default.lua — reference policy for argus-manager.
--
-- The manager decides WHEN relief is needed and HOW MUCH; the engine decides WHICH of
-- its KV cache techniques delivers it. So this script's whole job is to turn per-axis
-- pressure into one number: the fraction of the uncompressed KV cache to keep.
--
-- Written as a per-platform file rather than compiled in because none of the constants
-- below transfer. Which axis binds first, where "under pressure" starts, and how much
-- accuracy the deployment can trade are properties of the device and of whatever else is
-- running on it -- a phone coexisting with a game is not a Jetson coexisting with a
-- detector. Port by editing ENTER/EXIT and B_KV; the manager binary does not change.
--
--   ctx.pressure = { gpu, cpu, memory, thermal, latency, main_app }  -- 0..1
--   ctx.trigger  = { tbt_degraded, mem_low, temp_high }              -- Rust-side hysteresis
--   ctx.engine   = { seen, kv_cache_bytes, kv_cache_budget_bytes, kv_cache_tokens,
--                    tbt_ms, phase, state, kv_fill }
--   ctx.signal   = { memory = {...}, compute = {...}, thermal = {...} }
--   sys.*        = read/meminfo/thermal/gpu_busy/gpu_freq/cpu_freq/foreground_fps
--
-- Return nil, one action table, or an array of them:
--   { type = "kv.compress", budget = 0.5 }   { type = "restore_defaults" }
--   { type = "suspend" }                     { type = "resume" }

POLICY_META = { name = "argus_default", version = "3.0.0" }

-- Per-axis enter/exit thresholds. ENTER is tighter than EXIT, and the gap is the
-- hysteresis that keeps a signal sitting on a threshold from oscillating.
--
-- Memory is the exception: ENTER == EXIT, no gap. The other axes degrade -- a warm SoC
-- throttles, a slow token is still a token -- but crossing the memory limit means the
-- LMK kills the process and the whole conversation goes with it. There is no value in
-- damping a signal whose overshoot is unrecoverable.
local ENTER = { memory = 0.80, gpu = 0.85, thermal = 0.75, latency = 0.30 }
local EXIT  = { memory = 0.80, gpu = 0.60, thermal = 0.55, latency = 0.10 }

-- Budget ramp: how much KV to keep at a given pressure, from the ENTER threshold up.
local B_MAX, B_MIN = 0.90, 0.25

-- Quantize the budget so that pressure drifting within a step does not produce a new
-- directive every cycle. The engine re-submits on a CHANGED budget, so an unquantized
-- ramp would churn the cache once per tick.
local B_STEP = 0.05

-- Axes we can actually act on. A KV budget relieves memory directly and compute
-- indirectly (a shorter cache is less attention work); it does nothing about a hot SoC on
-- its own, but shortening the cache does cut the work that heats it, so thermal is in.
local AXES = { "memory", "gpu", "thermal", "latency" }

-- Which axes are currently being relieved. Persists across calls: this is the state that
-- makes EXIT mean something.
local relieving = {}

-- The budget already in force, or nil outside a relief window.
--
-- A window only ever TIGHTENS. The engine's budget is a fraction of what the context would
-- occupy uncompressed, and eviction is not reversible, so a LOOSER value inside one window
-- names a state the cache is already in -- it asks for nothing. Sending it anyway is not
-- free: the engine scores every candidate technique before it can decide there is nothing
-- to remove, and that scoring is the expensive half of a decision.
--
-- It is also not rare. `decide` runs every tick while relieving, and a pressure signal that
-- moves faster than one B_STEP walks B_kv up and down its ramp; every value that DIFFERS
-- from the last one clears both the manager's byte-identical dedup and the engine's
-- repeat gate. Archived Galaxy S25 runs show exactly this: one 867-tick cell emitted 111
-- distinct budgets (0.5, 0.25, 0.9, 0.85, 0.9, 0.85, 0.6, 0.7, ...) driven by a thermal
-- reading swinging 40.3-63.8 C across a 35->50 C normalization band. B_STEP is ~1.15 C of
-- that band, far below the per-tick swing, so quantizing the budget does not damp it.
--
-- Loosening happens by LEAVING the window (`restore_defaults`), which is the only action
-- that actually gives the cache back. Because the ramp has (B_MAX-B_MIN)/B_STEP + 1 = 14
-- levels and this makes the sequence monotone, one window can now emit at most 14
-- directives no matter how long it lasts or how badly the signal dithers.
local applied

local function normalize(ctx)
  local p = ctx.pressure
  return {
    memory  = p.memory,
    gpu     = p.gpu,
    thermal = p.thermal,
    -- `latency` is the TBT degradation ratio when a baseline exists, 0 before warm-up.
    latency = p.latency,
  }
end

local function most_stressed(p)
  local worst, worst_excess = nil, -math.huge
  for _, a in ipairs(AXES) do
    -- Compare axes by how far past their OWN enter threshold they are, not by raw
    -- pressure: 0.82 memory is an emergency and 0.82 GPU is a busy afternoon.
    local excess = p[a] - ENTER[a]
    if excess > worst_excess then
      worst, worst_excess = a, excess
    end
  end
  return worst
end

local function B_kv(pressure, axis)
  local span = 1.0 - ENTER[axis]
  local over = 0.0
  if span > 0 then
    over = math.min(math.max((pressure - ENTER[axis]) / span, 0.0), 1.0)
  end
  local budget = B_MAX - over * (B_MAX - B_MIN)
  budget = math.floor(budget / B_STEP + 0.5) * B_STEP
  return math.min(math.max(budget, B_MIN), B_MAX)
end

function decide(ctx)
  -- No heartbeat yet: the engine has not said how big its cache can get, so a budget
  -- would be a fraction of an unknown. `seen` exists to make this distinguishable from
  -- an engine whose cache is genuinely empty.
  if not ctx.engine.seen then
    return nil
  end

  local p = normalize(ctx)

  -- Close every window whose axis has fallen back under its own EXIT threshold, and sweep
  -- ALL of them rather than only the one `most_stressed` names. That function answers "which
  -- axis is furthest past its own threshold", and once every axis has headroom the answer is
  -- an arbitrary one of them. Keying the exit check on it therefore LOSES windows: a relieved
  -- axis that cools while some other axis happens to rank worst is never re-examined, so
  -- `relieving` and `applied` latch for the rest of the run. One spike would pin the cache at
  -- that budget forever -- and because the ratchet below refuses anything looser, even a later
  -- spike wanting a LOOSER budget returns nothing. Verified against this script directly.
  local still_relieving = false
  for _, axis in ipairs(AXES) do
    if relieving[axis] then
      if p[axis] < EXIT[axis] then
        relieving[axis] = nil
      else
        still_relieving = true
      end
    end
  end
  if applied and not still_relieving then
    -- Every window has closed. Leaving the window is the only thing that gives the cache back.
    applied = nil
    return { type = "restore_defaults" }
  end

  local a = most_stressed(p)
  if not relieving[a] and p[a] < ENTER[a] then
    -- Headroom on the worst axis means headroom everywhere: run at full accuracy.
    return nil
  end

  relieving[a] = true
  local budget = B_kv(p[a], a)
  if applied and budget >= applied then
    -- Already at least this tight, and eviction does not run backwards. See `applied`.
    return nil
  end
  applied = budget
  return { type = "kv.compress", budget = budget }
end
