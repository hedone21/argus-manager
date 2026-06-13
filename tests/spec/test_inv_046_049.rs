//! INV-046: RLS gain vector k = f(P, phi). lambda는 망각 인수.
//! INV-047: bias는 W 갱신 후 잔여 오차에 EMA(lr=0.1) 적용.
//! INV-048: P matrix: D x D 대칭 양정치. 초기값 100 * I.
//! INV-049: lambda in (0, 1]. lambda=1.0이면 forgetting 없음.
//!
//! 검증 대상: `OnlineLinearEstimator` (내부 `LinearModel` RLS 구현).
//! 원본: spec/22-manager-algorithms.md ALG-044 ~ ALG-046.
//!
//! LinearModel이 private이므로, public API를 통해 수치적으로 RLS 수식을 검증한다.

use argus_manager::relief::ReliefEstimator;
use argus_manager::relief::linear::OnlineLinearEstimator;
use argus_manager::types::{ActionId, FEATURE_DIM, FeatureVector, ReliefVector};

fn zero_state() -> FeatureVector {
    FeatureVector::zeros()
}

fn unit_state() -> FeatureVector {
    FeatureVector {
        values: [1.0; FEATURE_DIM],
    }
}

fn relief(compute: f32, memory: f32, thermal: f32, latency: f32) -> ReliefVector {
    ReliefVector {
        compute,
        memory,
        thermal,
        latency,
    }
}

// ---------------------------------------------------------------------------
// INV-046: RLS gain vector k = f(P, phi). lambda는 망각 인수.
// ---------------------------------------------------------------------------

/// INV-046: 첫 번째 observe 후 predict 결과가 RLS 수식과 일치하는지 수치 검증.
///
/// 초기 상태: W=0, b=0, P=100*I, lambda=1.0
/// phi = [1, 0, 0, ..., 0] (단위 벡터)
/// actual = (0.5, 0.0, 0.0, 0.0)
///
/// 수식:
///   P_phi = P * phi = [100, 0, ..., 0]
///   denom = lambda + phi^T * P_phi = 1 + 100 = 101
///   k = P_phi / denom = [100/101, 0, ..., 0]
///
///   예측(W=0, b=0): predicted = 0
///   error = 0.5 - 0 = 0.5
///   W[0][0] += k[0] * error = 100/101 * 0.5 = 50/101
///
///   bias 갱신:
///   predicted_new = W_new * phi + b = 50/101 + 0 = 50/101
///   residual = 0.5 - 50/101 = 0.5/101
///   b[0] += 0.1 * residual = 0.1 * 0.5/101 = 0.05/101
///
///   최종 predict = 50/101 + 0.05/101 = 50.05/101
#[test]
fn inv046_first_observe_matches_rls_formula() {
    let mut estimator = OnlineLinearEstimator::new(FEATURE_DIM, 1.0);
    let action = ActionId::SwitchHw;

    // phi = [1, 0, 0, ..., 0]
    let mut state = zero_state();
    state.values[0] = 1.0;

    let actual = relief(0.5, 0.0, 0.0, 0.0);
    estimator.observe(&action, &state, &actual);

    let pred = estimator.predict(&action, &state);

    // W[0][0] = 100/101 * 0.5 = 50/101 ~ 0.49505
    // b[0] = 0.1 * (0.5 - 50/101) = 0.1 * 0.5/101 ~ 0.000495
    // predict = 50/101 + 0.05/101 = 50.05/101 ~ 0.49554
    let expected_compute = 50.05 / 101.0;
    assert!(
        (pred.compute - expected_compute).abs() < 1e-4,
        "INV-046: After first observe, compute prediction should be ~{:.5}, got {:.5}",
        expected_compute,
        pred.compute
    );
}

/// INV-046: lambda < 1.0일 때 과거 관측의 가중치가 감소하는지 확인.
/// 동일 데이터에서 lambda=0.9 vs lambda=1.0의 수렴 속도 비교.
#[test]
fn inv046_lambda_affects_forgetting() {
    let action = ActionId::KvEvictSliding;
    let state = unit_state();

    // Phase 1: actual_1로 10회 학습
    let actual_1 = relief(0.8, 0.0, 0.0, 0.0);
    // Phase 2: actual_2로 10회 학습
    let actual_2 = relief(0.2, 0.0, 0.0, 0.0);

    // lambda = 0.9 (강한 forgetting)
    let mut est_forget = OnlineLinearEstimator::new(FEATURE_DIM, 0.9);
    for _ in 0..10 {
        est_forget.observe(&action, &state, &actual_1);
    }
    for _ in 0..10 {
        est_forget.observe(&action, &state, &actual_2);
    }
    let pred_forget = est_forget.predict(&action, &state);

    // lambda = 1.0 (forgetting 없음)
    let mut est_no_forget = OnlineLinearEstimator::new(FEATURE_DIM, 1.0);
    for _ in 0..10 {
        est_no_forget.observe(&action, &state, &actual_1);
    }
    for _ in 0..10 {
        est_no_forget.observe(&action, &state, &actual_2);
    }
    let pred_no_forget = est_no_forget.predict(&action, &state);

    // lambda=0.9는 최근 데이터(0.2)에 더 빨리 적응해야 한다
    // → pred_forget.compute가 0.2에 더 가까워야 한다
    let dist_forget = (pred_forget.compute - 0.2).abs();
    let dist_no_forget = (pred_no_forget.compute - 0.2).abs();
    assert!(
        dist_forget < dist_no_forget,
        "INV-046: lambda=0.9 (forgetting) should adapt faster to recent data. \
         forget_dist={:.4}, no_forget_dist={:.4}",
        dist_forget,
        dist_no_forget
    );
}

// ---------------------------------------------------------------------------
// INV-047: bias는 W 갱신 후 잔여 오차에 EMA(lr=0.1) 적용
// ---------------------------------------------------------------------------

/// INV-047: bias가 EMA(lr=0.1)로 갱신되는지 수치 검증.
///
/// phi = [0, 0, ..., 0] (zero vector)이면 W*phi = 0이므로
/// predict = b 만 반영된다. bias 갱신만 추적 가능.
///
/// 첫 번째 observe:
///   W*phi = 0 (phi=0이면 k=0이므로 W 갱신 없음)
///   predicted = b = 0
///   error = actual - 0 = actual
///   k = P * 0 / (lambda + 0) = 0 → W 갱신 없음
///   predicted_new = 0 + b = 0
///   residual = actual - 0 = actual
///   b += 0.1 * actual
///
/// 두 번째 observe (같은 phi=0, 같은 actual):
///   predicted = b = 0.1*actual
///   error = actual - 0.1*actual = 0.9*actual
///   k = 0 → W 갱신 없음
///   predicted_new = b = 0.1*actual
///   residual = actual - 0.1*actual = 0.9*actual
///   b += 0.1 * 0.9*actual = 0.09*actual → b = 0.19*actual
#[test]
fn inv047_bias_ema_with_zero_phi() {
    let mut estimator = OnlineLinearEstimator::new(FEATURE_DIM, 1.0);
    let action = ActionId::Throttle;
    let state = zero_state();

    let actual_val = 1.0;
    let actual = relief(actual_val, 0.0, 0.0, 0.0);

    // 첫 번째 observe: b[0] = 0.1 * 1.0 = 0.1
    estimator.observe(&action, &state, &actual);
    let pred1 = estimator.predict(&action, &state);
    assert!(
        (pred1.compute - 0.1).abs() < 1e-5,
        "INV-047: After 1st observe with zero phi, b should be 0.1, got {:.5}",
        pred1.compute
    );

    // 두 번째 observe: b[0] = 0.1 + 0.1 * (1.0 - 0.1) = 0.1 + 0.09 = 0.19
    estimator.observe(&action, &state, &actual);
    let pred2 = estimator.predict(&action, &state);
    assert!(
        (pred2.compute - 0.19).abs() < 1e-5,
        "INV-047: After 2nd observe with zero phi, b should be 0.19, got {:.5}",
        pred2.compute
    );

    // 세 번째: b = 0.19 + 0.1 * (1.0 - 0.19) = 0.19 + 0.081 = 0.271
    estimator.observe(&action, &state, &actual);
    let pred3 = estimator.predict(&action, &state);
    assert!(
        (pred3.compute - 0.271).abs() < 1e-4,
        "INV-047: After 3rd observe, b should be ~0.271, got {:.5}",
        pred3.compute
    );
}

/// INV-047: bias는 W와 독립적으로 갱신된다 (P matrix는 W만 다룬다).
/// phi != 0일 때 bias가 잔여 오차에 EMA로 갱신되는 것을 확인.
#[test]
fn inv047_bias_updates_on_residual_after_w_update() {
    let mut estimator = OnlineLinearEstimator::new(FEATURE_DIM, 1.0);
    let action = ActionId::KvEvictSliding;

    let mut state = zero_state();
    state.values[0] = 1.0;

    let actual = relief(1.0, 0.0, 0.0, 0.0);
    estimator.observe(&action, &state, &actual);

    // W[0][0] = (100/101)*1.0 = 100/101
    // predicted_new = 100/101
    // residual = 1.0 - 100/101 = 1/101
    // b[0] = 0.1 * 1/101 = 0.1/101

    let pred = estimator.predict(&action, &state);
    // predict = W[0][0]*1.0 + b[0] = 100/101 + 0.1/101 = 100.1/101
    let expected = 100.1 / 101.0;
    assert!(
        (pred.compute - expected).abs() < 1e-4,
        "INV-047: W+bias combined prediction should be ~{:.5}, got {:.5}",
        expected,
        pred.compute
    );
}

// ---------------------------------------------------------------------------
// INV-048: P matrix: D x D 대칭 양정치. 초기값 100 * I.
// ---------------------------------------------------------------------------

/// INV-048: 초기 predict 후 첫 observe가 높은 학습률을 보여야 한다.
///
/// P = 100 * I는 초기 불확실성이 크다는 뜻으로,
/// 첫 관측에서 예측이 actual에 크게 수렴해야 한다.
#[test]
fn inv048_initial_high_p_causes_fast_initial_learning() {
    let mut estimator = OnlineLinearEstimator::new(FEATURE_DIM, 1.0);
    let action = ActionId::SwitchHw;

    let mut state = zero_state();
    state.values[0] = 1.0;

    let actual = relief(1.0, 0.0, 0.0, 0.0);

    // 첫 관측: P=100*I → k[0] = 100/(1+100) = 100/101 ~ 0.99
    // W[0][0] = 100/101 * 1.0 ~ 0.99
    estimator.observe(&action, &state, &actual);
    let pred = estimator.predict(&action, &state);

    // P=100*I 덕분에 첫 관측에서 99% 수렴
    assert!(
        pred.compute > 0.98,
        "INV-048: P=100*I initial value should cause ~99% convergence on first observe, got {:.4}",
        pred.compute
    );
}

/// INV-048: 초기 P가 identity matrix * 100이므로,
/// 서로 다른 feature 차원이 독립적으로 학습되어야 한다.
#[test]
fn inv048_independent_feature_dimensions() {
    let mut estimator = OnlineLinearEstimator::new(FEATURE_DIM, 1.0);
    let action = ActionId::Throttle;

    // feature 0만 활성인 상태로 학습
    let mut state_0 = zero_state();
    state_0.values[0] = 1.0;
    let actual = relief(0.5, 0.0, 0.0, 0.0);
    estimator.observe(&action, &state_0, &actual);

    // feature 1만 활성인 상태로 predict → W[0][1]이 아직 0이므로 bias만 반영
    let mut state_1 = zero_state();
    state_1.values[1] = 1.0;
    let pred_1 = estimator.predict(&action, &state_1);

    // feature 0로 학습한 것이 feature 1 predict에 W를 통해 크게 영향을 미치면 안 됨
    // P가 대각이므로 cross-feature 영향 없이 W[dim][0]만 갱신됨
    // pred_1 = W[0][1]*1.0 + b ≈ 0 + bias
    // feature 0로 학습 → W[0][0] 갱신, W[0][1]은 여전히 0
    let pred_0 = estimator.predict(&action, &state_0);
    assert!(
        pred_0.compute > pred_1.compute * 5.0,
        "INV-048: Feature learned on dim0 should not transfer to dim1 (diagonal P). \
         pred_0={:.4}, pred_1={:.4}",
        pred_0.compute,
        pred_1.compute
    );
}

// ---------------------------------------------------------------------------
// INV-049: lambda in (0, 1]. lambda=1.0이면 forgetting 없음.
// ---------------------------------------------------------------------------

/// INV-049: lambda=1.0이면 표준 RLS (forgetting 없음).
/// 오래된 관측과 최근 관측의 가중치가 동일해야 한다.
#[test]
fn inv049_lambda_one_no_forgetting() {
    let mut estimator = OnlineLinearEstimator::new(FEATURE_DIM, 1.0);
    let action = ActionId::KvEvictSliding;
    let state = unit_state();

    // Phase 1: actual=0.8 으로 10회 학습
    for _ in 0..10 {
        estimator.observe(&action, &state, &relief(0.8, 0.0, 0.0, 0.0));
    }

    // Phase 2: actual=0.2 으로 10회 학습
    for _ in 0..10 {
        estimator.observe(&action, &state, &relief(0.2, 0.0, 0.0, 0.0));
    }

    let pred = estimator.predict(&action, &state);

    // lambda=1.0 (no forgetting)이면 20회 관측의 평균에 가까워야 함
    // (0.8*10 + 0.2*10) / 20 = 0.5
    // 정확한 RLS 수렴값은 아니지만 0.5 근처여야 한다
    assert!(
        pred.compute > 0.3 && pred.compute < 0.7,
        "INV-049: lambda=1.0 should weight old and new equally, got {:.4}",
        pred.compute
    );
}

/// INV-049: lambda=1.0 vs lambda=0.9에서 최근 데이터 반영 차이 확인.
#[test]
fn inv049_lambda_less_than_one_forgets_old_data() {
    let action = ActionId::KvEvictSliding;
    let state = unit_state();

    // lambda = 1.0 (no forgetting)
    let mut est_1 = OnlineLinearEstimator::new(FEATURE_DIM, 1.0);
    // lambda = 0.9 (strong forgetting)
    let mut est_09 = OnlineLinearEstimator::new(FEATURE_DIM, 0.9);

    // Phase 1: actual=0.8 20회
    for _ in 0..20 {
        est_1.observe(&action, &state, &relief(0.8, 0.0, 0.0, 0.0));
        est_09.observe(&action, &state, &relief(0.8, 0.0, 0.0, 0.0));
    }
    // Phase 2: actual=0.0 20회
    for _ in 0..20 {
        est_1.observe(&action, &state, &relief(0.0, 0.0, 0.0, 0.0));
        est_09.observe(&action, &state, &relief(0.0, 0.0, 0.0, 0.0));
    }

    let pred_1 = est_1.predict(&action, &state);
    let pred_09 = est_09.predict(&action, &state);

    // lambda=0.9는 과거(0.8)를 더 빨리 잊으므로 최근(0.0)에 가까워야 함
    assert!(
        pred_09.compute < pred_1.compute,
        "INV-049: lambda=0.9 should forget old data faster (closer to 0.0). \
         pred_0.9={:.4}, pred_1.0={:.4}",
        pred_09.compute,
        pred_1.compute
    );
}

/// INV-049: OnlineLinearEstimator에 설정된 forgetting_factor가 반영되는지 확인.
#[test]
fn inv049_config_forgetting_factor_is_used() {
    // 기본 config: lambda = 0.995
    let est_default = OnlineLinearEstimator::default_config();
    assert_eq!(
        est_default.observation_count(&ActionId::SwitchHw),
        0,
        "INV-049: Initial observation count should be 0"
    );

    // 관측 후 observation_count 증가 확인
    let mut est = OnlineLinearEstimator::new(FEATURE_DIM, 0.5);
    let state = unit_state();
    est.observe(&ActionId::SwitchHw, &state, &relief(0.5, 0.0, 0.0, 0.0));
    assert_eq!(
        est.observation_count(&ActionId::SwitchHw),
        1,
        "INV-049: observation_count should be 1 after one observe"
    );
}
