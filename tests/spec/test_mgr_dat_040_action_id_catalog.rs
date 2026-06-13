//! MGR-DAT-040: ActionId 카탈로그 — SwapWeights 등록 검증
//!
//! ActionId::SwapWeights variant 존재, from_str 매핑, all() 목록,
//! primary_domain, serde round-trip, 카탈로그 전체 크기(10개) 회귀 방지.

use std::collections::HashSet;

use argus_manager::types::{ActionId, Domain};

// ── from_str 매핑 ──

/// "swap_weights" → ActionId::SwapWeights 변환이 성공해야 한다.
#[test]
fn test_mgr_dat_040_from_str_swap_weights() {
    let id = ActionId::from_str("swap_weights");
    assert_eq!(
        id,
        Some(ActionId::SwapWeights),
        "from_str(\"swap_weights\") should return Some(SwapWeights)"
    );
}

/// camelCase 입력("swapWeights")은 None을 반환해야 한다.
#[test]
fn test_mgr_dat_040_from_str_camel_case_rejected() {
    let id = ActionId::from_str("swapWeights");
    assert_eq!(
        id, None,
        "from_str(\"swapWeights\") should return None (snake_case only)"
    );
}

/// 빈 문자열은 None을 반환해야 한다.
#[test]
fn test_mgr_dat_040_from_str_empty_is_none() {
    assert_eq!(ActionId::from_str(""), None);
}

// ── all() 목록 ──

/// ActionId::all()이 정확히 10개를 포함해야 한다 (회귀 방지).
#[test]
fn test_mgr_dat_040_all_count_is_ten() {
    assert_eq!(
        ActionId::all().len(),
        10,
        "ActionId::all() must contain exactly 10 variants"
    );
}

/// ActionId::all()에 SwapWeights가 포함되어야 한다.
#[test]
fn test_mgr_dat_040_all_contains_swap_weights() {
    assert!(
        ActionId::all().contains(&ActionId::SwapWeights),
        "ActionId::all() must include SwapWeights"
    );
}

/// ActionId::all()이 정확히 10개 variant를 포함하는지 set 비교.
#[test]
fn test_mgr_dat_040_all_exact_set() {
    let expected: HashSet<ActionId> = [
        ActionId::SwitchHw,
        ActionId::Throttle,
        ActionId::KvOffloadDisk,
        ActionId::KvEvictSliding,
        ActionId::KvEvictH2o,
        ActionId::KvEvictStreaming,
        ActionId::KvMergeD2o,
        ActionId::KvQuantDynamic,
        ActionId::LayerSkip,
        ActionId::SwapWeights,
    ]
    .into_iter()
    .collect();

    let actual: HashSet<ActionId> = ActionId::all().iter().copied().collect();
    assert_eq!(actual, expected, "ActionId::all() set mismatch");
}

// ── primary_domain ──

/// SwapWeights의 primary_domain은 Memory여야 한다.
#[test]
fn test_mgr_dat_040_swap_weights_primary_domain_is_memory() {
    assert_eq!(
        ActionId::SwapWeights.primary_domain(),
        Domain::Memory,
        "SwapWeights primary_domain must be Memory"
    );
}

// ── serde round-trip ──

/// SwapWeights serde JSON round-trip: "swap_weights" 문자열.
#[test]
fn test_mgr_dat_040_serde_roundtrip() {
    let id = ActionId::SwapWeights;
    let json = serde_json::to_string(&id).expect("serialization should succeed");
    assert_eq!(json, r#""swap_weights""#, "serde should produce snake_case");
    let back: ActionId = serde_json::from_str(&json).expect("deserialization should succeed");
    assert_eq!(back, id, "round-trip should preserve value");
}
