use std::collections::HashMap;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::types::{ActionId, FEATURE_DIM, FeatureVector, ReliefVector};

use super::ReliefEstimator;

/// 단일 액션에 대한 온라인 선형 회귀 모델 (RLS 업데이트).
///
/// relief = W × φ + b
/// - W: 4 × D 행렬 (4개 relief 차원 × D feature)
/// - b: 4차원 bias
/// - P: D × D 역공분산 행렬 (RLS)
#[derive(Clone, Serialize, Deserialize)]
struct LinearModel {
    /// 4 × D 가중치 행렬 (행: relief 차원, 열: feature)
    weights: Vec<Vec<f32>>,
    /// 4차원 bias
    bias: [f32; 4],
    /// D × D 역공분산 행렬 (RLS P matrix)
    p_matrix: Vec<Vec<f32>>,
    /// 망각 인수 λ
    forgetting_factor: f32,
    /// 누적 관측 횟수
    observation_count: u32,
}

impl LinearModel {
    fn new(feature_dim: usize, forgetting_factor: f32) -> Self {
        // W: 4×D zeros
        let weights = vec![vec![0.0f32; feature_dim]; 4];
        // bias: [0; 4]
        let bias = [0.0f32; 4];
        // P: D×D identity × 100.0 (큰 초기 불확실성)
        let mut p_matrix = vec![vec![0.0f32; feature_dim]; feature_dim];
        for (i, row) in p_matrix.iter_mut().enumerate() {
            row[i] = 100.0;
        }
        Self {
            weights,
            bias,
            p_matrix,
            forgetting_factor,
            observation_count: 0,
        }
    }

    /// 현재 feature φ에 대해 4차원 relief를 예측한다.
    fn predict(&self, phi: &[f32]) -> ReliefVector {
        let mut vals = [0.0f32; 4];
        for (dim, val) in vals.iter_mut().enumerate() {
            let dot: f32 = self.weights[dim]
                .iter()
                .zip(phi.iter())
                .map(|(&w, &x)| w * x)
                .sum();
            *val = dot + self.bias[dim];
        }
        ReliefVector {
            compute: vals[0],
            memory: vals[1],
            thermal: vals[2],
            latency: vals[3],
        }
    }

    /// RLS 업데이트: 실측값 actual을 이용해 모델을 갱신한다.
    ///
    /// P matrix는 feature_dim × feature_dim 크기를 유지하고,
    /// bias는 별도 EMA로 갱신한다.
    fn update(&mut self, phi: &[f32], actual: &ReliefVector) {
        let d = phi.len();
        let lambda = self.forgetting_factor;
        let lr_bias = 0.1f32;

        // actual을 배열로 취급
        let actual_arr = [
            actual.compute,
            actual.memory,
            actual.thermal,
            actual.latency,
        ];

        // --- RLS: k = P × φ / (λ + φᵀ × P × φ) ---

        // Pφ = P × phi (D-벡터)
        let mut p_phi = vec![0.0f32; d];
        for (p_phi_i, row) in p_phi.iter_mut().zip(self.p_matrix.iter()) {
            *p_phi_i = row.iter().zip(phi.iter()).map(|(&p, &x)| p * x).sum();
        }

        // denom = λ + φᵀ × Pφ
        let denom = lambda
            + phi
                .iter()
                .zip(p_phi.iter())
                .map(|(&x, &pp)| x * pp)
                .sum::<f32>();

        // k = Pφ / denom (D-벡터, gain vector)
        let k: Vec<f32> = p_phi.iter().map(|&v| v / denom).collect();

        // 4개 relief 차원 각각 가중치 업데이트
        for (dim, &actual_d) in actual_arr.iter().enumerate() {
            let predicted_d: f32 = self.weights[dim]
                .iter()
                .zip(phi.iter())
                .map(|(&w, &x)| w * x)
                .sum::<f32>()
                + self.bias[dim];
            let error = actual_d - predicted_d;
            for (w, &ki) in self.weights[dim].iter_mut().zip(k.iter()) {
                *w += ki * error;
            }
            // bias: EMA 업데이트
            let predicted_with_new_w: f32 = self.weights[dim]
                .iter()
                .zip(phi.iter())
                .map(|(&w, &x)| w * x)
                .sum::<f32>()
                + self.bias[dim];
            let residual = actual_d - predicted_with_new_w;
            self.bias[dim] += lr_bias * residual;
        }

        // P 업데이트: P = (P - k × φᵀ × P) / λ
        // kφᵀP = k × (φᵀ × P) → (D×1)(1×D)(D×D) = D×D
        // 먼저 φᵀP[j] = Σᵢ phi[i] * P[i][j] 계산
        let mut phi_t_p = vec![0.0f32; d];
        for (i, &phi_i) in phi.iter().enumerate() {
            for (j, phi_t_p_j) in phi_t_p.iter_mut().enumerate() {
                *phi_t_p_j += phi_i * self.p_matrix[i][j];
            }
        }
        // P[i][j] = (P[i][j] - k[i] * phi_t_p[j]) / λ
        for (i, row) in self.p_matrix.iter_mut().enumerate() {
            for (j, p_ij) in row.iter_mut().enumerate() {
                *p_ij = (*p_ij - k[i] * phi_t_p[j]) / lambda;
            }
        }

        self.observation_count += 1;
    }
}

/// 저장 포맷 (JSON 직렬화용 컨테이너)
#[derive(Serialize, Deserialize)]
struct SavedEstimator {
    feature_dim: usize,
    forgetting_factor: f32,
    models: HashMap<String, LinearModel>,
}

/// ActionId별 독립 LinearModel을 관리하는 온라인 선형 추정기.
pub struct OnlineLinearEstimator {
    models: HashMap<ActionId, LinearModel>,
    feature_dim: usize,
    forgetting_factor: f32,
}

impl OnlineLinearEstimator {
    pub fn new(feature_dim: usize, forgetting_factor: f32) -> Self {
        Self {
            models: HashMap::new(),
            feature_dim,
            forgetting_factor,
        }
    }

    /// 기본 설정으로 생성 (FEATURE_DIM=13, λ=0.995)
    pub fn default_config() -> Self {
        Self::new(FEATURE_DIM, 0.995)
    }

    /// 모델이 없으면 초기화한다.
    fn ensure_model(&mut self, action: &ActionId) {
        self.models
            .entry(*action)
            .or_insert_with(|| LinearModel::new(self.feature_dim, self.forgetting_factor));
    }

    /// ActionId를 문자열 키로 변환 (저장 시 사용)
    fn action_to_key(action: &ActionId) -> &'static str {
        action.as_str()
    }

    /// 문자열 키로부터 ActionId 변환 (로드 시 사용)
    fn key_to_action(s: &str) -> Option<ActionId> {
        ActionId::from_str(s)
    }
}

/// 학습 데이터 없을 때 사용하는 도메인 기반 default relief.
///
/// 각 액션의 주 효과를 전문 지식 기반으로 사전 지정한다.
fn default_relief(action: &ActionId) -> ReliefVector {
    match action {
        ActionId::SwitchHw => ReliefVector {
            compute: 0.5,
            memory: 0.0,
            thermal: 0.3,
            latency: -0.1,
        },
        ActionId::Throttle => ReliefVector {
            compute: 0.3,
            memory: 0.0,
            thermal: 0.2,
            latency: -0.3,
        },
        ActionId::KvOffloadDisk => ReliefVector {
            compute: 0.0,
            memory: 0.4,
            thermal: 0.0,
            latency: -0.2,
        },
        ActionId::KvEvictSliding => ReliefVector {
            compute: 0.0,
            memory: 0.7,
            thermal: 0.0,
            latency: 0.0,
        },
        ActionId::KvEvictH2o => ReliefVector {
            compute: 0.0,
            memory: 0.6,
            thermal: 0.0,
            latency: 0.0,
        },
        ActionId::KvEvictStreaming => ReliefVector {
            compute: 0.0,
            memory: 0.7,
            thermal: 0.0,
            latency: 0.0,
        },
        ActionId::KvMergeD2o => ReliefVector {
            compute: 0.0,
            memory: 0.6,
            thermal: 0.0,
            latency: 0.0,
        },
        ActionId::KvQuantDynamic => ReliefVector {
            compute: 0.0,
            memory: 0.3,
            thermal: 0.0,
            latency: 0.0,
        },
        ActionId::LayerSkip => ReliefVector {
            compute: 0.3,
            memory: 0.0,
            thermal: 0.1,
            latency: -0.2,
        },
        ActionId::SwapWeights => ReliefVector {
            compute: 0.0,
            memory: 0.5,
            thermal: 0.0,
            latency: -0.2,
        },
    }
}

impl ReliefEstimator for OnlineLinearEstimator {
    fn predict(&self, action: &ActionId, state: &FeatureVector) -> ReliefVector {
        match self.models.get(action) {
            Some(model) if model.observation_count > 0 => model.predict(&state.values),
            _ => default_relief(action),
        }
    }

    fn observe(&mut self, action: &ActionId, state: &FeatureVector, actual: &ReliefVector) {
        self.ensure_model(action);
        let model = self.models.get_mut(action).unwrap();
        model.update(&state.values, actual);
    }

    fn save(&self, path: &Path) -> io::Result<()> {
        let mut string_models: HashMap<String, LinearModel> = HashMap::new();
        for (action, model) in &self.models {
            string_models.insert(Self::action_to_key(action).to_string(), model.clone());
        }
        let saved = SavedEstimator {
            feature_dim: self.feature_dim,
            forgetting_factor: self.forgetting_factor,
            models: string_models,
        };
        let json = serde_json::to_string_pretty(&saved)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    fn load(&mut self, path: &Path) -> io::Result<()> {
        let json = std::fs::read_to_string(path)?;
        let saved: SavedEstimator = serde_json::from_str(&json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.feature_dim = saved.feature_dim;
        self.forgetting_factor = saved.forgetting_factor;
        self.models.clear();
        for (key, model) in saved.models {
            if let Some(action) = Self::key_to_action(&key) {
                self.models.insert(action, model);
            }
        }
        Ok(())
    }

    fn observation_count(&self, action: &ActionId) -> u32 {
        self.models
            .get(action)
            .map(|m| m.observation_count)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FEATURE_DIM;

    fn make_state(values: [f32; FEATURE_DIM]) -> FeatureVector {
        FeatureVector { values }
    }

    fn zero_state() -> FeatureVector {
        FeatureVector::zeros()
    }

    fn simple_state() -> FeatureVector {
        let mut values = [0.0f32; FEATURE_DIM];
        values[0] = 0.5; // kv_occupancy
        values[1] = 1.0; // is_gpu
        values[2] = 0.3; // token_progress
        FeatureVector { values }
    }

    fn relief(compute: f32, memory: f32, thermal: f32, latency: f32) -> ReliefVector {
        ReliefVector {
            compute,
            memory,
            thermal,
            latency,
        }
    }

    /// 관측 없는 초기 predict는 도메인 기반 default relief를 반환해야 한다.
    #[test]
    fn test_predict_initial_returns_default_relief() {
        let estimator = OnlineLinearEstimator::default_config();
        let state = simple_state();
        let pred = estimator.predict(&ActionId::KvEvictSliding, &state);
        // KvEvictSliding default: compute=0.0, memory=0.7, thermal=0.0, latency=0.0
        assert!(
            pred.compute.abs() < 1e-6,
            "initial compute should be 0.0 (default), got {}",
            pred.compute
        );
        assert!(
            (pred.memory - 0.7).abs() < 1e-6,
            "initial memory should be 0.7 (default), got {}",
            pred.memory
        );
        assert!(
            pred.thermal.abs() < 1e-6,
            "initial thermal should be 0.0 (default), got {}",
            pred.thermal
        );
        assert!(
            pred.latency.abs() < 1e-6,
            "initial latency should be 0.0 (default), got {}",
            pred.latency
        );
    }

    /// 학습 전 default_relief 값이 각 액션에 올바르게 매핑된다.
    #[test]
    fn test_default_relief_values() {
        let estimator = OnlineLinearEstimator::default_config();
        let state = simple_state();

        let sw = estimator.predict(&ActionId::SwitchHw, &state);
        assert!(
            (sw.compute - 0.5).abs() < 1e-6,
            "SwitchHw compute={}",
            sw.compute
        );
        assert!(
            (sw.thermal - 0.3).abs() < 1e-6,
            "SwitchHw thermal={}",
            sw.thermal
        );
        assert!(
            (sw.latency - (-0.1)).abs() < 1e-6,
            "SwitchHw latency={}",
            sw.latency
        );

        let thr = estimator.predict(&ActionId::Throttle, &state);
        assert!(
            (thr.compute - 0.3).abs() < 1e-6,
            "Throttle compute={}",
            thr.compute
        );
        assert!(
            (thr.latency - (-0.3)).abs() < 1e-6,
            "Throttle latency={}",
            thr.latency
        );

        let ls = estimator.predict(&ActionId::LayerSkip, &state);
        assert!(
            (ls.compute - 0.3).abs() < 1e-6,
            "LayerSkip compute={}",
            ls.compute
        );
        assert!(
            (ls.latency - (-0.2)).abs() < 1e-6,
            "LayerSkip latency={}",
            ls.latency
        );
    }

    /// 같은 state+relief를 10회 observe 후 predict 결과가 actual 방향으로 수렴해야 한다.
    #[test]
    fn test_observe_and_predict() {
        let mut estimator = OnlineLinearEstimator::default_config();
        let state = simple_state();
        let actual = relief(0.6, 0.8, 0.3, -0.2);

        for _ in 0..10 {
            estimator.observe(&ActionId::KvEvictSliding, &state, &actual);
        }

        let pred = estimator.predict(&ActionId::KvEvictSliding, &state);

        // 방향이 맞는지 확인: 양수 relief는 양수 예측, 음수는 음수 예측
        assert!(
            pred.compute > 0.0,
            "compute should be positive after training, got {}",
            pred.compute
        );
        assert!(
            pred.memory > 0.0,
            "memory should be positive after training, got {}",
            pred.memory
        );
        assert!(
            pred.thermal > 0.0,
            "thermal should be positive after training, got {}",
            pred.thermal
        );
        assert!(
            pred.latency < 0.0,
            "latency should be negative after training, got {}",
            pred.latency
        );
    }

    /// action A의 observe가 action B의 predict에 영향을 미치지 않아야 한다.
    #[test]
    fn test_multiple_actions_independent() {
        let mut estimator = OnlineLinearEstimator::default_config();
        let state = simple_state();
        let actual = relief(0.5, 0.7, 0.2, -0.1);

        // Action A만 학습
        for _ in 0..20 {
            estimator.observe(&ActionId::KvEvictSliding, &state, &actual);
        }

        // Action B는 아직 학습하지 않음 → default_relief(SwitchHw) 반환
        // SwitchHw: compute=0.5, memory=0.0, thermal=0.3, latency=-0.1
        let pred_b = estimator.predict(&ActionId::SwitchHw, &state);
        assert!(
            (pred_b.compute - 0.5).abs() < 1e-6,
            "untrained SwitchHw should return default compute=0.5, got {}",
            pred_b.compute
        );
        assert!(
            pred_b.memory.abs() < 1e-6,
            "untrained SwitchHw should return default memory=0.0, got {}",
            pred_b.memory
        );
    }

    /// save → load → predict 결과가 동일해야 한다 (roundtrip).
    #[test]
    fn test_save_load_roundtrip() {
        let mut estimator = OnlineLinearEstimator::default_config();
        let state = simple_state();
        let actual = relief(0.4, 0.6, 0.1, -0.3);

        for _ in 0..5 {
            estimator.observe(&ActionId::KvEvictSliding, &state, &actual);
            estimator.observe(&ActionId::SwitchHw, &state, &actual);
        }

        let pred_before = estimator.predict(&ActionId::KvEvictSliding, &state);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        estimator.save(tmp.path()).expect("save should succeed");

        let mut estimator2 = OnlineLinearEstimator::default_config();
        estimator2.load(tmp.path()).expect("load should succeed");

        let pred_after = estimator2.predict(&ActionId::KvEvictSliding, &state);

        assert!(
            (pred_before.compute - pred_after.compute).abs() < 1e-5,
            "compute mismatch: {} vs {}",
            pred_before.compute,
            pred_after.compute
        );
        assert!(
            (pred_before.memory - pred_after.memory).abs() < 1e-5,
            "memory mismatch: {} vs {}",
            pred_before.memory,
            pred_after.memory
        );
        assert!(
            (pred_before.latency - pred_after.latency).abs() < 1e-5,
            "latency mismatch: {} vs {}",
            pred_before.latency,
            pred_after.latency
        );
    }

    /// observe 호출 횟수와 observation_count가 일치해야 한다.
    #[test]
    fn test_observation_count() {
        let mut estimator = OnlineLinearEstimator::default_config();
        let state = zero_state();
        let actual = relief(0.1, 0.2, 0.0, 0.0);

        assert_eq!(estimator.observation_count(&ActionId::Throttle), 0);

        for i in 1..=7 {
            estimator.observe(&ActionId::Throttle, &state, &actual);
            assert_eq!(
                estimator.observation_count(&ActionId::Throttle),
                i,
                "count should be {i}"
            );
        }

        // 다른 액션은 별도 카운트
        assert_eq!(estimator.observation_count(&ActionId::SwitchHw), 0);
    }

    /// 동일 state에 반복 observe 후 예측이 actual에 점차 수렴하는지 확인.
    #[test]
    fn test_convergence_direction() {
        let mut estimator = OnlineLinearEstimator::default_config();
        // 모든 feature가 1인 state
        let state = make_state([1.0f32; FEATURE_DIM]);
        let actual = relief(0.8, 0.5, 0.0, -0.4);

        for _ in 0..30 {
            estimator.observe(&ActionId::KvEvictH2o, &state, &actual);
        }

        let pred = estimator.predict(&ActionId::KvEvictH2o, &state);
        // 30회 학습 후 예측이 actual에 50% 이상 수렴해야 한다
        assert!(
            (pred.compute - actual.compute).abs() < actual.compute * 0.5,
            "compute did not converge: pred={}, actual={}",
            pred.compute,
            actual.compute
        );
        assert!(
            (pred.memory - actual.memory).abs() < actual.memory * 0.5,
            "memory did not converge: pred={}, actual={}",
            pred.memory,
            actual.memory
        );
    }
}
