#[cfg(feature = "hierarchical")]
#[path = "spec/helpers.rs"]
mod helpers;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_016.rs"]
mod test_inv_016;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_030_031.rs"]
mod test_inv_030_031;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_032_033.rs"]
mod test_inv_032_033;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_034_036.rs"]
mod test_inv_034_036;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_037_038.rs"]
mod test_inv_037_038;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_039_040.rs"]
mod test_inv_039_040;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_041_042.rs"]
mod test_inv_041_042;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_043_044.rs"]
mod test_inv_043_044;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_046_049.rs"]
mod test_inv_046_049;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_050.rs"]
mod test_inv_050;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_083_085.rs"]
mod test_inv_083_085;

// 비-INV 요구사항 Spec 테스트
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_050_054.rs"]
mod test_mgr_050_054;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_067_072.rs"]
mod test_mgr_067_072;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_alg_010_014.rs"]
mod test_mgr_alg_010_014;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_alg_031_035.rs"]
mod test_mgr_alg_031_035;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_alg_036_051.rs"]
mod test_mgr_alg_036_051;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_alg_042_047.rs"]
mod test_mgr_alg_042_047;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_dat_020_056.rs"]
mod test_mgr_dat_020_056;

// ── 갭 테스트 (MGR-ALG / MGR-060 / MGR-DAT) ──
#[cfg(feature = "hierarchical")]
#[path = "spec/test_inv_016_swap_weights_no_exclusion.rs"]
mod test_inv_016_swap_weights_no_exclusion;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_060_061.rs"]
mod test_mgr_060_061;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_alg_013a_016.rs"]
mod test_mgr_alg_013a_016;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_dat_022_024.rs"]
mod test_mgr_dat_022_024;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_dat_040_action_id_catalog.rs"]
mod test_mgr_dat_040_action_id_catalog;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_dat_054_swap_weights_meta.rs"]
mod test_mgr_dat_054_swap_weights_meta;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_dat_d2o.rs"]
mod test_mgr_dat_d2o;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_mgr_dat_streaming.rs"]
mod test_mgr_dat_streaming;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_swap_weights_dual_path_equivalence.rs"]
mod test_swap_weights_dual_path_equivalence;

// ── SEQ 통합 테스트 ──
#[cfg(feature = "hierarchical")]
#[path = "spec/test_seq_040_064_mgr.rs"]
mod test_seq_040_064_mgr;
#[cfg(feature = "hierarchical")]
#[path = "spec/test_seq_095_098.rs"]
mod test_seq_095_098;

// ── LuaPolicy (2026-04 기본 정책) EWMA Relief Adaptation ──
// MGR-ALG-080~083, MGR-090~093, MGR-DAT-070~074, SEQ-055~057, INV-086~090
// feature gate 없음 — LuaPolicy가 기본 경로.
#[path = "spec/test_mgr_alg_080_083_ewma_relief.rs"]
mod test_mgr_alg_080_083_ewma_relief;

// ── Engine Self-Utilization Heartbeat 노출 (2026-04 Phase 1) ──
// MGR-DAT-075, MGR-DAT-076, MSG-069, INV-091~092
// feature gate 없음 — LuaPolicy ctx.engine 노출 경로.
#[path = "spec/test_mgr_dat_075_076_engine_util.rs"]
mod test_mgr_dat_075_076_engine_util;

// ── MGR-049: Lua sandbox IO 허용 + OS/PACKAGE/DEBUG 차단 ──
#[cfg(feature = "lua")]
#[path = "spec/test_mgr_049_lua_sandbox.rs"]
mod test_mgr_049_lua_sandbox;
