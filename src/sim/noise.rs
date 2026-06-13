//! 옵션 Gaussian noise: seed 고정, seed_key별 독립 ChaCha8Rng 스트림.
//!
//! `rng_seed: null`이 기본값이며, 그 경우 noise는 완전히 비활성화(0 반환)된다.
//! seed를 지정하면 같은 seed + 같은 key = 같은 시퀀스가 보장된다.

#![allow(dead_code)]

use rand::RngCore;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use std::collections::HashMap;

use super::config::ScenarioConfig;

// ─────────────────────────────────────────────────────────
// NoiseRng
// ─────────────────────────────────────────────────────────

/// seed_key별 독립 ChaCha8Rng 스트림을 관리하는 노이즈 생성기.
///
/// 같은 `master_seed` + 같은 `seed_key` 조합은 항상 동일한 시퀀스를 생성한다.
pub struct NoiseRng {
    streams: HashMap<String, ChaCha8Rng>,
    master_seed: u64,
}

impl NoiseRng {
    pub fn new(master_seed: u64) -> Self {
        Self {
            streams: HashMap::new(),
            master_seed,
        }
    }

    /// 지정된 `seed_key`로 독립 스트림에서 N(0, sigma²) 분포의 샘플을 반환한다.
    ///
    /// 스트림이 없으면 `hash(master_seed, key)` 기반으로 초기 seed를 생성한다.
    pub fn gaussian(&mut self, seed_key: &str, sigma: f64) -> f64 {
        if sigma == 0.0 {
            return 0.0;
        }
        let rng = self.streams.entry(seed_key.to_string()).or_insert_with(|| {
            let stream_seed = derive_stream_seed(self.master_seed, seed_key);
            ChaCha8Rng::seed_from_u64(stream_seed)
        });

        box_muller(rng) * sigma
    }

    /// sigma=0으로 gaussian을 호출하는 것과 동일하다 (noise 없음).
    pub fn no_noise(&self) -> f64 {
        0.0
    }
}

// ─────────────────────────────────────────────────────────
// 헬퍼 함수
// ─────────────────────────────────────────────────────────

/// master_seed와 key를 결합하여 스트림 seed를 파생한다.
/// FNV-1a 기반 간단한 해시 사용.
fn derive_stream_seed(master_seed: u64, key: &str) -> u64 {
    // FNV-1a 초기값과 곱승자
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;

    let mut hash = FNV_OFFSET;
    // master_seed 바이트를 먼저 넣는다
    for byte in master_seed.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    // key 바이트를 이어서 넣는다
    for byte in key.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Box-Muller 변환으로 표준 정규 분포 N(0,1) 샘플을 생성한다.
///
/// ChaCha8Rng에서 두 개의 균등 분포 난수 u1, u2를 얻어 변환한다.
fn box_muller(rng: &mut ChaCha8Rng) -> f64 {
    use std::f64::consts::TAU;

    // [1, u32::MAX] 범위의 균등 난수 → (0, 1] 범위로 정규화
    let u1 = loop {
        let raw = rng.next_u32();
        if raw != 0 {
            break raw as f64 / u32::MAX as f64;
        }
    };
    let u2 = rng.next_u32() as f64 / u32::MAX as f64;

    // Box-Muller 공식: z = sqrt(-2 * ln(u1)) * cos(2π * u2)
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

// ─────────────────────────────────────────────────────────
// 공용 헬퍼
// ─────────────────────────────────────────────────────────

/// `ScenarioConfig.rng_seed`가 Some이면 NoiseRng를 생성, None이면 None을 반환.
pub fn maybe_create(cfg: &ScenarioConfig) -> Option<NoiseRng> {
    cfg.rng_seed.map(NoiseRng::new)
}

// ─────────────────────────────────────────────────────────
// 단위 테스트
// ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_reproducibility() {
        let n = 20;
        let key = "test.stream";
        let sigma = 1.0;

        let mut rng1 = NoiseRng::new(42);
        let samples1: Vec<f64> = (0..n).map(|_| rng1.gaussian(key, sigma)).collect();

        let mut rng2 = NoiseRng::new(42);
        let samples2: Vec<f64> = (0..n).map(|_| rng2.gaussian(key, sigma)).collect();

        assert_eq!(samples1, samples2, "같은 seed + 같은 key → 동일 시퀀스");
    }

    #[test]
    fn test_noise_disabled_returns_zero_sigma_equivalent() {
        // sigma=0이면 0을 반환한다
        let mut rng = NoiseRng::new(99);
        for _ in 0..10 {
            assert_eq!(rng.gaussian("any", 0.0), 0.0);
        }
    }

    #[test]
    fn test_maybe_create_with_seed() {
        use super::super::config::load_scenario;
        use std::path::PathBuf;

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("sim")
            .join("baseline.yaml");
        let mut cfg = load_scenario(&path).expect("baseline.yaml should load");

        cfg.rng_seed = None;
        assert!(
            maybe_create(&cfg).is_none(),
            "seed None → maybe_create None"
        );

        cfg.rng_seed = Some(42);
        assert!(
            maybe_create(&cfg).is_some(),
            "seed Some → maybe_create Some"
        );
    }

    #[test]
    fn test_noise_independent_streams() {
        let mut rng = NoiseRng::new(42);
        let n = 10;
        let sigma = 2.0;

        let a: Vec<f64> = (0..n).map(|_| rng.gaussian("stream.a", sigma)).collect();

        // 새 rng로 다시 생성하여 비교 (같은 key는 같은 스트림)
        let mut rng2 = NoiseRng::new(42);
        let b: Vec<f64> = (0..n).map(|_| rng2.gaussian("stream.b", sigma)).collect();

        // 서로 다른 key는 다른 스트림 → 시퀀스가 달라야 함
        assert_ne!(a, b, "서로 다른 seed_key는 독립 시퀀스를 가져야 함");
    }

    #[test]
    fn test_noise_same_key_same_sequence_across_instances() {
        // 두 인스턴스에서 같은 key에 대해 동일 시퀀스
        let n = 5;
        let mut rng_a = NoiseRng::new(777);
        let mut rng_b = NoiseRng::new(777);
        for _ in 0..n {
            let va = rng_a.gaussian("key", 1.0);
            let vb = rng_b.gaussian("key", 1.0);
            assert_eq!(va, vb);
        }
    }

    #[test]
    fn test_derive_stream_seed_different_keys() {
        let s1 = derive_stream_seed(42, "key_a");
        let s2 = derive_stream_seed(42, "key_b");
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_derive_stream_seed_different_masters() {
        let s1 = derive_stream_seed(1, "same_key");
        let s2 = derive_stream_seed(2, "same_key");
        assert_ne!(s1, s2);
    }
}
