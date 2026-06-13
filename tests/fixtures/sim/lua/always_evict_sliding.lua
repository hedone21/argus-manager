-- always_evict_sliding.lua
--
-- 목적: 모든 signal에 대해 항상 kv_evict_sliding을 반환한다.
-- 단일 커맨드를 반환하므로 ObservationContext가 생성된다.
-- EWMA relief 학습 경로 검증용 결정론적 스크립트.

function decide(ctx)
  return {{ type = "kv.evict_sliding", keep_ratio = 0.7 }}
end
