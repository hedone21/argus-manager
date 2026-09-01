//! Spec tests that survive the KV-only contract.
//!
//! The rest — thirty `hierarchical`-gated files plus the EWMA-relief and
//! self-utilization suites — tested the DPP selector, the relief table, and heartbeat
//! fields the contract no longer carries. Their subjects are gone, so the tests went with
//! them rather than being re-pointed at something they were not written to check.

// ── MGR-049: Lua sandbox allows IO, blocks OS/PACKAGE/DEBUG ──
#[cfg(feature = "lua")]
#[path = "spec/test_mgr_049_lua_sandbox.rs"]
mod test_mgr_049_lua_sandbox;
