use std::io;
use std::path::Path;

use crate::types::{ActionId, FeatureVector, ReliefVector};

pub mod linear;

/// Strategy 패턴: Relief 추정기 인터페이스.
///
/// 각 ActionId에 대해 현재 시스템 상태(FeatureVector)를 입력으로 받아
/// 해당 액션을 적용했을 때 예측되는 압력 완화량(ReliefVector)을 반환한다.
pub trait ReliefEstimator: Send + Sync {
    /// 해당 액션을 현재 state에서 적용했을 때의 예측 relief.
    fn predict(&self, action: &ActionId, state: &FeatureVector) -> ReliefVector;

    /// 실측 관측으로 모델 업데이트.
    fn observe(&mut self, action: &ActionId, state: &FeatureVector, actual: &ReliefVector);

    /// 디스크에 저장.
    fn save(&self, path: &Path) -> io::Result<()>;

    /// 디스크에서 복원.
    fn load(&mut self, path: &Path) -> io::Result<()>;

    /// 해당 액션의 관측 횟수.
    fn observation_count(&self, action: &ActionId) -> u32;
}
