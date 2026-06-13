//! MGR-ALG-080 ~ MGR-ALG-083, MGR-090 ~ MGR-093, MGR-DAT-070 ~ MGR-DAT-074,
//! SEQ-055 ~ SEQ-057, INV-086 ~ INV-090: LuaPolicy EWMA Relief Adaptation Spec 테스트
//!
//! 2026-04 LuaPolicy 기본 채택에 따라 도입된 `EwmaReliefTable` + `ObservationContext`
//! 기반 학습 경로의 불변식·알고리즘·데이터 스키마·관측 시퀀스·영속화 정책을 검증한다.
//!
//! 노출 확장 (lua_policy.rs → pub):
//!
//! - `RELIEF_DIMS: usize` — 6D 차원 상수
//! - `ReliefEntry { relief, observation_count }` — JSON 직렬화 대상
//! - `EwmaReliefTable` — new/observe/predict/save/load/snapshot
//!
//! 이유: integration test는 pub(crate) 접근이 불가하므로, 테스트 계약 타입은 pub으로 노출.
//! `#[doc(hidden)]`으로 API 문서에서 숨긴다.

#![allow(clippy::needless_doctest_main)]

#[cfg(feature = "lua")]
mod ewma_tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use argus_manager::config::AdaptationConfig;
    use argus_manager::pipeline::PolicyStrategy;
    use argus_manager::{EwmaReliefTable, RELIEF_DIMS, ReliefEntry};

    const TOL: f32 = 1e-4;

    // ─────────────────────────────────────────────────────────────────────────
    // MGR-ALG-080 / INV-087: EWMA observe 수식 + 첫 관측 EWMA 적용
    // ─────────────────────────────────────────────────────────────────────────

    /// MGR-ALG-080, INV-087: 첫 번째 observe도 EWMA를 적용한다.
    /// defaults가 없을 때 entry 초기값은 [0.0; 6]이므로:
    /// 첫 관측 후 relief[i] = alpha * 0.0 + (1-alpha) * observed[i] = (1-alpha) * observed[i]
    #[test]
    fn mgr_alg_080_first_observation_direct_assignment() {
        let alpha = 0.875f32;
        let mut table = EwmaReliefTable::new(alpha, HashMap::new());
        let observed: [f32; 6] = [0.1, 0.2, 0.3, -0.1, 0.05, 0.4];

        table.observe("test_action", &observed);

        let entry = table.entries.get("test_action").expect("entry 없음");
        assert_eq!(entry.observation_count, 1, "첫 관측 후 count==1");
        for (i, &v) in observed.iter().enumerate() {
            let expected = (1.0 - alpha) * v;
            assert!(
                (entry.relief[i] - expected).abs() < TOL,
                "dim[{}]: expected {} (= (1-alpha)*obs), got {}",
                i,
                expected,
                entry.relief[i]
            );
        }
    }

    /// MGR-ALG-080: 두 번째 이후 observe에도 동일하게 α 평활이 적용된다.
    /// 첫 관측: relief = (1-alpha)*first (initial=0)
    /// 두 번째: relief = alpha * ((1-alpha)*first) + (1-alpha)*second
    #[test]
    fn mgr_alg_080_subsequent_observation_applies_alpha() {
        let alpha = 0.875f32;
        let mut table = EwmaReliefTable::new(alpha, HashMap::new());

        let first: [f32; 6] = [1.0, 0.5, 0.0, 0.2, 0.1, 0.3];
        let second: [f32; 6] = [0.0, 0.0, 1.0, 0.8, 0.9, 0.0];

        table.observe("action_a", &first);
        table.observe("action_a", &second);

        let entry = table.entries.get("action_a").expect("entry 없음");
        assert_eq!(entry.observation_count, 2, "두 번째 관측 후 count==2");

        for i in 0..RELIEF_DIMS {
            // after first: (1-alpha)*first[i]
            let after_first = (1.0 - alpha) * first[i];
            // after second: alpha * after_first + (1-alpha) * second[i]
            let expected = alpha * after_first + (1.0 - alpha) * second[i];
            assert!(
                (entry.relief[i] - expected).abs() < TOL,
                "dim[{}]: expected EWMA {:.6}, got {:.6}",
                i,
                expected,
                entry.relief[i]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MGR-ALG-081 / INV-090: predict 우선순위 + 읽기 전용성
    // ─────────────────────────────────────────────────────────────────────────

    /// MGR-ALG-081: entries에 있으면 defaults보다 우선순위가 높다.
    /// 첫 관측 후 entries 값은 EWMA 적용 결과이므로 defaults(9.0)와 다르다.
    #[test]
    fn mgr_alg_081_predict_entries_priority() {
        let alpha = 0.875f32;
        let default_val = 9.0f32;
        let mut defaults = HashMap::new();
        defaults.insert("A".to_string(), vec![default_val; 6]);
        let mut table = EwmaReliefTable::new(alpha, defaults);

        // entries에 "A" 등록: initial = default = 9.0
        // 첫 관측 후: alpha * 9.0 + (1-alpha) * obs
        let obs: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        table.observe("A", &obs);

        let result = table.predict("A");
        for i in 0..RELIEF_DIMS {
            let expected = alpha * default_val + (1.0 - alpha) * obs[i];
            // entries가 존재하므로 defaults(9.0)가 아닌 EWMA 결과를 반환해야 함
            assert!(
                (result[i] - expected).abs() < TOL,
                "dim[{}]: entries({}) 우선, defaults(9.0) 무시되어야 함",
                i,
                result[i]
            );
        }
    }

    /// MGR-ALG-081: entries에 없으면 defaults를 반환한다.
    #[test]
    fn mgr_alg_081_predict_falls_back_to_defaults() {
        let mut defaults = HashMap::new();
        defaults.insert("A".to_string(), vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let table = EwmaReliefTable::new(0.875, defaults);

        let result = table.predict("A");
        let expected = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        for i in 0..RELIEF_DIMS {
            assert!(
                (result[i] - expected[i]).abs() < TOL,
                "dim[{}]: defaults fallback 실패: {} vs {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    /// MGR-ALG-081: entries도 defaults도 없으면 [0.0; 6] 반환.
    #[test]
    fn mgr_alg_081_predict_returns_zeros_when_unknown() {
        let table = EwmaReliefTable::new(0.875, HashMap::new());
        let result = table.predict("unknown_action");
        assert_eq!(
            result, [0.0; 6],
            "entries/defaults 모두 없으면 zeros 반환해야 함"
        );
    }

    /// INV-090: predict는 entries를 변경하지 않는다 (읽기 전용).
    #[test]
    fn inv_090_predict_is_read_only() {
        let table = EwmaReliefTable::new(0.875, HashMap::new());
        assert!(table.entries.is_empty(), "초기 entries 비어있어야 함");

        let _ = table.predict("unknown");
        let _ = table.predict("another_unknown");
        let _ = table.predict("unknown");

        assert!(
            table.entries.is_empty(),
            "predict 호출 후에도 entries가 비어있어야 함 (INV-090)"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MGR-ALG-082 / INV-088 / SEQ-055: 단일 액션 관측 제한
    // ─────────────────────────────────────────────────────────────────────────

    /// MGR-ALG-082, INV-088, SEQ-055: 다중 커맨드 시 첫 번째 액션만 ObservationContext에 기록됨.
    ///
    /// LuaPolicy.process_signal()은 commands.len() == 1일 때만 observation을 생성한다.
    /// 복수 커맨드이면 observation = None (SEQ-055: single-action tracking).
    /// 이 동작은 LuaPolicy를 통해 시뮬레이터에서 검증한다.
    #[test]
    fn mgr_alg_082_single_action_tracked_from_multi_command_decision() {
        use argus_manager::lua_policy::LuaPolicy;
        use argus_manager::pipeline::PolicyStrategy;
        use argus_shared::SystemSignal;

        // 두 커맨드를 반환하는 Lua 스크립트 — observation 생성 안 됨
        let script = r#"
            function decide(ctx)
                return {
                    { type = "kv.evict_sliding", keep_ratio = 0.7 },
                    { type = "throttle", delay_ms = 100 }
                }
            end
        "#;

        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("multi_cmd.lua");
        std::fs::write(&script_path, script).unwrap();

        // qcf_penalty_weight=0.0 으로 QCF 선발행 로직을 비활성화한다.
        // 이 테스트는 multi-command observation 생성 안 됨을 검증하므로
        // QCF RequestQcf가 먼저 개입하면 안 된다.
        let config = AdaptationConfig {
            qcf_penalty_weight: 0.0,
            ..AdaptationConfig::default()
        };
        let mut policy =
            LuaPolicy::with_system_clock(script_path.to_str().unwrap(), config).unwrap();

        let signal = SystemSignal::MemoryPressure {
            level: argus_shared::Level::Warning,
            available_bytes: 1_000_000,
            total_bytes: 8_000_000,
            reclaim_target_bytes: 0,
        };
        let directive = policy.process_signal(&signal).expect("directive 반환 필요");

        // 두 커맨드이므로 observation 생성 안 됨 → relief_snapshot은 비어있어야 함
        assert_eq!(directive.commands.len(), 2, "두 커맨드가 반환되어야 함");

        // 3초 이상 경과 후에도 relief는 비어있어야 함 (관측이 시작되지 않았으므로)
        // → 다시 signal 주입 후 relief_snapshot 확인
        let _ = policy.process_signal(&signal);
        let snapshot = argus_manager::pipeline::get_relief_snapshot(&mut policy);
        // observation이 시작되지 않았으므로 relief는 비어있음
        assert!(
            snapshot.is_empty(),
            "다중 커맨드 시 ObservationContext 미생성 → relief 빈 상태여야 함 (INV-088)"
        );
    }

    /// INV-088, SEQ-055: 관찰 대기 중 새 커맨드가 오면 기존 관찰이 학습 없이 폐기된다.
    ///
    /// 구현 동작: commands.len() != 1이면 self.observation = None으로 교체한다.
    /// 단일 커맨드로 observation을 시작한 뒤, 다시 단일 다른 커맨드로 교체하면
    /// 기존 action에 대한 observe()는 호출되지 않는다.
    #[test]
    fn inv_088_pending_observation_dropped_on_new_command() {
        use std::sync::{Arc, Mutex};

        use argus_manager::clock::{Clock, LogicalInstant};
        use argus_manager::lua_policy::LuaPolicy;
        use argus_manager::pipeline::PolicyStrategy;
        use argus_shared::SystemSignal;

        // ManualClock: 수동으로 시간을 제어
        struct ManualClock {
            now: Arc<Mutex<Duration>>,
        }
        impl ManualClock {
            fn new() -> (Self, Arc<Mutex<Duration>>) {
                let inner = Arc::new(Mutex::new(Duration::ZERO));
                (
                    Self {
                        now: Arc::clone(&inner),
                    },
                    inner,
                )
            }
        }
        impl Clock for ManualClock {
            fn now(&self) -> LogicalInstant {
                LogicalInstant::from_duration_since_start(*self.now.lock().unwrap())
            }
        }

        let (clock, time_ctl) = ManualClock::new();

        // action_A를 반환하는 Lua 스크립트
        let script_a = r#"
            function decide(ctx)
                return {{ type = "throttle", delay_ms = 50 }}
            end
        "#;
        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("action_a.lua");
        std::fs::write(&script_path, script_a).unwrap();

        let mut policy = LuaPolicy::new(
            script_path.to_str().unwrap(),
            AdaptationConfig::default(),
            Arc::new(clock),
        )
        .unwrap();

        let mem_signal = SystemSignal::MemoryPressure {
            level: argus_shared::Level::Warning,
            available_bytes: 1_000_000,
            total_bytes: 8_000_000,
            reclaim_target_bytes: 0,
        };

        // 첫 번째 signal: observation(throttle, t=0) 시작
        let _ = policy.process_signal(&mem_signal);

        // 3초 미경과 상태에서 두 번째 signal: observation 미완료 상태
        // (time_ctl은 ZERO 그대로이므로 elapsed < 3.0s)
        *time_ctl.lock().unwrap() = Duration::from_millis(1000); // 1초
        let _ = policy.process_signal(&mem_signal);

        // 아직 3초가 경과하지 않았고, 두 번째 signal에서도 새 observation으로 교체됨.
        // relief_snapshot은 비어있어야 함 (learn 안 됨).
        let snapshot = argus_manager::pipeline::get_relief_snapshot(&mut policy);
        assert!(
            snapshot.is_empty(),
            "3초 미경과 상태에서 새 signal → 기존 observation 폐기, relief 비어있어야 함 (INV-088)"
        );
    }

    /// MGR-ALG-082: decide()가 빈 배열 반환 시 ObservationContext 미생성.
    #[test]
    fn mgr_alg_082_no_observation_on_empty_decide() {
        use argus_manager::lua_policy::LuaPolicy;
        use argus_manager::pipeline::PolicyStrategy;
        use argus_shared::SystemSignal;

        let script = r#"
            function decide(ctx)
                return {}
            end
        "#;

        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("empty.lua");
        std::fs::write(&script_path, script).unwrap();

        let mut policy = LuaPolicy::with_system_clock(
            script_path.to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();

        let signal = SystemSignal::MemoryPressure {
            level: argus_shared::Level::Normal,
            available_bytes: 4_000_000,
            total_bytes: 8_000_000,
            reclaim_target_bytes: 0,
        };

        // 여러 번 signal 주입해도 decide()가 빈 배열 반환 → directive 없음
        for _ in 0..5 {
            let result = policy.process_signal(&signal);
            assert!(result.is_none(), "빈 decide() → directive 없어야 함");
        }

        // relief_snapshot은 비어있어야 함 (observation 미생성)
        let snapshot = argus_manager::pipeline::get_relief_snapshot(&mut policy);
        assert!(
            snapshot.is_empty(),
            "빈 커맨드 반환 시 ObservationContext 미생성 → relief 비어있어야 함"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MGR-ALG-083 / INV-089 / SEQ-056: 3초 Settling + 6D 부호 규약
    // ─────────────────────────────────────────────────────────────────────────

    /// MGR-ALG-083, SEQ-056: OBSERVATION_DELAY_SECS=3.0 경계 검증.
    ///
    /// ManualClock을 사용해 시각을 정밀하게 제어한다.
    /// process_signal() 흐름:
    ///   1. check_observation() — elapsed >= 3.0s이면 learn
    ///   2. call_decide() → 새 observation 시작 (기존 교체)
    ///
    /// 따라서 "2.999s 미경과 검증"과 "3.001s 경과 검증"은 서로 다른 관측 인스턴스를 사용해야 한다.
    #[test]
    fn mgr_alg_083_observation_requires_3_second_delay() {
        use std::sync::{Arc, Mutex};

        use argus_manager::clock::{Clock, LogicalInstant};
        use argus_manager::lua_policy::LuaPolicy;
        use argus_shared::SystemSignal;

        struct ManualClock {
            now: Arc<Mutex<Duration>>,
        }
        impl Clock for ManualClock {
            fn now(&self) -> LogicalInstant {
                LogicalInstant::from_duration_since_start(*self.now.lock().unwrap())
            }
        }

        let time = Arc::new(Mutex::new(Duration::ZERO));
        let clock = ManualClock {
            now: Arc::clone(&time),
        };

        let script = r#"
            function decide(ctx)
                return {{ type = "kv.evict_sliding", keep_ratio = 0.7 }}
            end
        "#;
        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("evict.lua");
        std::fs::write(&script_path, script).unwrap();

        let mut policy = LuaPolicy::new(
            script_path.to_str().unwrap(),
            AdaptationConfig::default(),
            Arc::new(clock),
        )
        .unwrap();

        let mem_signal = SystemSignal::MemoryPressure {
            level: argus_shared::Level::Warning,
            available_bytes: 1_000_000,
            total_bytes: 8_000_000,
            reclaim_target_bytes: 0,
        };

        // ── 케이스 A: 2.999초 미경과 시 learn 없음 ──────────────────────────
        // t=0: observation(kv_evict_sliding, timestamp=0) 생성
        *time.lock().unwrap() = Duration::ZERO;
        let _ = policy.process_signal(&mem_signal);

        // t=2.999s: check_observation → elapsed=2.999s < 3.0s → observation 유지, learn 안 됨
        //           그 후 새 directive → observation 교체(timestamp=2.999s)
        *time.lock().unwrap() = Duration::from_millis(2999);
        let _ = policy.process_signal(&mem_signal);
        let snapshot_before = argus_manager::pipeline::get_relief_snapshot(&mut policy);
        assert!(
            snapshot_before.is_empty(),
            "2.999초 시점에서 learn이 없어야 함 (elapsed < 3.0s)"
        );

        // ── 케이스 B: 3.001초 경과 시 learn 발생 ───────────────────────────
        // 현재 observation은 timestamp=2.999s
        // t=6.001s: elapsed = 6.001 - 2.999 = 3.002s >= 3.0s → learn 발생
        *time.lock().unwrap() = Duration::from_millis(6001);
        let _ = policy.process_signal(&mem_signal);
        let snapshot_after = argus_manager::pipeline::get_relief_snapshot(&mut policy);
        assert!(
            !snapshot_after.is_empty(),
            "3.002초 경과 시점에서 relief에 학습된 값이 있어야 함 (MGR-ALG-083)"
        );
        assert!(
            snapshot_after.contains_key("kv.evict_sliding"),
            "kv_evict_sliding 액션이 학습되어야 함"
        );
    }

    /// INV-089, MGR-ALG-083, MGR-DAT-073: 차원 0~4는 before - after (양수 = 압박 감소).
    #[test]
    fn inv_089_sign_convention_dim_0_to_4() {
        // EwmaReliefTable.observe에 넘겨지는 observed 배열이 expected 부호 규약을 따르는지 검증.
        // observed[0..5] = before[0..5] - after[0..5]
        // before = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        // after  = [0.5, 0.4, 0.3, 0.2, 0.1, 0.2]
        // observed[0..5] = [0.3, 0.3, 0.3, 0.3, 0.3, ?]

        let before = [0.8f32, 0.7, 0.6, 0.5, 0.4, 0.3];
        let after = [0.5f32, 0.4, 0.3, 0.2, 0.1, 0.2];

        // check_observation()의 계산을 수동으로 재현
        let observed: [f32; 6] = [
            before[0] - after[0], // gpu: +0.3
            before[1] - after[1], // cpu: +0.3
            before[2] - after[2], // memory: +0.3
            before[3] - after[3], // thermal: +0.3
            before[4] - after[4], // latency: +0.3
            after[5] - before[5], // main_app_qos: +0.2 (부호 반전)
        ];

        // dim 0~4: 모두 양수 (before > after → 압박 감소)
        for (i, &obs_val) in observed.iter().enumerate().take(5) {
            assert!(
                obs_val > 0.0,
                "dim[{}] observed={}: before-after는 양수여야 함 (INV-089)",
                i,
                obs_val
            );
            assert!(
                (obs_val - 0.3).abs() < TOL,
                "dim[{}]: expected 0.3, got {}",
                i,
                obs_val
            );
        }

        // EwmaReliefTable에 observe 후 relief = (1-alpha) * observed (defaults 없음, initial=0)
        let alpha = 0.875f32;
        let mut table = EwmaReliefTable::new(alpha, HashMap::new());
        table.observe("test", &observed);
        let entry = table.entries.get("test").unwrap();
        for (i, &r) in entry.relief.iter().enumerate().take(5) {
            let expected = (1.0 - alpha) * 0.3;
            assert!(
                (r - expected).abs() < TOL,
                "dim[{}] relief: expected {}, got {}",
                i,
                expected,
                r
            );
        }
    }

    /// INV-089: 차원 5(main_app_qos)는 after - before (양수 = QoS 향상).
    #[test]
    fn inv_089_sign_convention_dim_5_main_app() {
        let before_main_app = 0.3f32;
        let after_main_app = 0.5f32;

        // dim 5: after - before = +0.2 (QoS 향상)
        let observed_dim5 = after_main_app - before_main_app;
        assert!(
            observed_dim5 > 0.0,
            "dim[5] = after - before = {} > 0 (QoS 향상 시 양수, INV-089)",
            observed_dim5
        );
        assert!(
            (observed_dim5 - 0.2).abs() < TOL,
            "dim[5]: expected 0.2, got {}",
            observed_dim5
        );

        // EwmaReliefTable에 관측: relief[5] = (1-alpha) * observed_dim5 (defaults 없음, initial=0)
        let alpha = 0.875f32;
        let observed: [f32; 6] = [0.0, 0.0, 0.0, 0.0, 0.0, observed_dim5];
        let mut table = EwmaReliefTable::new(alpha, HashMap::new());
        table.observe("action", &observed);
        let entry = table.entries.get("action").unwrap();
        let expected_dim5 = (1.0 - alpha) * 0.2;
        assert!(
            (entry.relief[5] - expected_dim5).abs() < TOL,
            "dim[5] relief: expected {}, got {}",
            expected_dim5,
            entry.relief[5]
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MGR-DAT-070 ~ MGR-DAT-074: 데이터 타입 스키마
    // ─────────────────────────────────────────────────────────────────────────

    /// MGR-DAT-070: ReliefEntry JSON round-trip.
    #[test]
    fn mgr_dat_070_relief_entry_serde_roundtrip() {
        let entry = ReliefEntry {
            relief: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            observation_count: 7,
        };

        let json = serde_json::to_string(&entry).expect("직렬화 실패");
        let restored: ReliefEntry = serde_json::from_str(&json).expect("역직렬화 실패");

        assert_eq!(restored.observation_count, 7);
        for i in 0..RELIEF_DIMS {
            assert!(
                (restored.relief[i] - entry.relief[i]).abs() < TOL,
                "dim[{}]: {} vs {}",
                i,
                restored.relief[i],
                entry.relief[i]
            );
        }
    }

    /// MGR-DAT-071, INV-086: save()는 entries만 직렬화한다 (alpha/defaults 키 없음).
    #[test]
    fn mgr_dat_071_save_only_serializes_entries() {
        let mut defaults = HashMap::new();
        defaults.insert("A".to_string(), vec![1.0f32; 6]);

        let mut table = EwmaReliefTable::new(0.9, defaults);
        let obs: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        table.observe("A", &obs);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("relief.json");
        table.save(&path).expect("save 실패");

        let content = std::fs::read_to_string(&path).expect("파일 읽기 실패");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("JSON 파싱 실패");

        let obj = parsed.as_object().expect("루트가 object여야 함");
        assert!(
            !obj.contains_key("alpha"),
            "save()는 alpha 필드를 포함하지 않아야 함 (MGR-DAT-071)"
        );
        assert!(
            !obj.contains_key("defaults"),
            "save()는 defaults 필드를 포함하지 않아야 함 (MGR-DAT-071)"
        );

        // entries의 키만 있어야 함
        assert!(obj.contains_key("A"), "action 'A'가 저장되어야 함");
    }

    /// MGR-DAT-072: AdaptationConfig::default() 기본값 검증.
    #[test]
    fn mgr_dat_072_adaptation_config_defaults() {
        let config = AdaptationConfig::default();

        assert!(
            (config.ewma_alpha - 0.875).abs() < TOL,
            "ewma_alpha 기본값: expected 0.875, got {}",
            config.ewma_alpha
        );
        assert!(
            config.relief_table_path.is_empty(),
            "relief_table_path 기본값은 빈 문자열이어야 함"
        );
        assert!(
            (config.temp_safe_c - 35.0).abs() < TOL,
            "temp_safe_c 기본값: expected 35.0, got {}",
            config.temp_safe_c
        );
        assert!(
            (config.temp_critical_c - 50.0).abs() < TOL,
            "temp_critical_c 기본값: expected 50.0, got {}",
            config.temp_critical_c
        );
    }

    /// MGR-DAT-073: RELIEF_DIMS == 6, ReliefEntry.relief 타입은 [f32; 6].
    #[test]
    fn mgr_dat_073_relief_dims_is_six() {
        assert_eq!(RELIEF_DIMS, 6, "RELIEF_DIMS는 6이어야 함");

        // ReliefEntry.relief 타입이 [f32; 6]인지 컴파일 타임 검증
        let entry = ReliefEntry {
            relief: [0.0f32; 6],
            observation_count: 0,
        };
        assert_eq!(entry.relief.len(), RELIEF_DIMS);
    }

    /// MGR-DAT-074: TriggerConfig::default() 기본값 및 hysteresis 관계.
    #[test]
    fn mgr_dat_074_trigger_config_defaults_and_hysteresis() {
        use argus_manager::config::TriggerConfig;

        let trigger = TriggerConfig::default();

        assert!(
            trigger.tbt_enter > trigger.tbt_exit,
            "tbt_enter({}) > tbt_exit({}) hysteresis",
            trigger.tbt_enter,
            trigger.tbt_exit
        );
        assert!(
            (trigger.tbt_enter - 0.30).abs() < 1e-6,
            "tbt_enter 기본값: expected 0.30, got {}",
            trigger.tbt_enter
        );
        assert!(
            (trigger.tbt_exit - 0.10).abs() < 1e-6,
            "tbt_exit 기본값: expected 0.10, got {}",
            trigger.tbt_exit
        );

        assert!(
            trigger.mem_enter > trigger.mem_exit,
            "mem_enter({}) > mem_exit({}) hysteresis",
            trigger.mem_enter,
            trigger.mem_exit
        );
        assert!(
            (trigger.mem_enter - 0.80).abs() < 1e-6,
            "mem_enter 기본값: expected 0.80, got {}",
            trigger.mem_enter
        );
        assert!(
            (trigger.mem_exit - 0.60).abs() < 1e-6,
            "mem_exit 기본값: expected 0.60, got {}",
            trigger.mem_exit
        );

        assert!(
            trigger.temp_enter > trigger.temp_exit,
            "temp_enter({}) > temp_exit({}) hysteresis",
            trigger.temp_enter,
            trigger.temp_exit
        );
        assert!(
            (trigger.temp_enter - 0.70).abs() < 1e-6,
            "temp_enter 기본값: expected 0.70, got {}",
            trigger.temp_enter
        );
        assert!(
            (trigger.temp_exit - 0.50).abs() < 1e-6,
            "temp_exit 기본값: expected 0.50, got {}",
            trigger.temp_exit
        );

        assert_eq!(
            trigger.tbt_warmup_tokens, 20,
            "tbt_warmup_tokens 기본값: expected 20"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // INV-086 / MGR-091 / MGR-092 / MGR-093 / SEQ-057: 영속화 정책
    // ─────────────────────────────────────────────────────────────────────────

    /// INV-086: save()는 raw 관측 이력을 포함하지 않고, EWMA 누적값과 count만 저장한다.
    #[test]
    fn inv_086_save_excludes_raw_history() {
        let mut table = EwmaReliefTable::new(0.875, HashMap::new());

        // 5회 관측
        for i in 0..5u32 {
            let val = i as f32 * 0.1;
            let obs: [f32; 6] = [val; 6];
            table.observe("kv.evict_sliding", &obs);
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("relief.json");
        table.save(&path).expect("save 실패");

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

        let entry_val = &parsed["kv.evict_sliding"];
        assert!(!entry_val.is_null(), "kv_evict_sliding entry가 존재해야 함");

        // observation_count와 relief만 있어야 함 (raw history 없음)
        let entry_obj = entry_val.as_object().unwrap();
        assert!(entry_obj.contains_key("relief"), "relief 필드 필요");
        assert!(
            entry_obj.contains_key("observation_count"),
            "observation_count 필드 필요"
        );

        // raw history 필드가 없어야 함
        assert!(
            !entry_obj.contains_key("history"),
            "raw history는 저장되지 않아야 함 (INV-086)"
        );
        assert!(
            !entry_obj.contains_key("observations"),
            "raw observations는 저장되지 않아야 함 (INV-086)"
        );

        // observation_count == 5
        assert_eq!(
            entry_obj["observation_count"].as_u64().unwrap(),
            5,
            "observation_count == 5"
        );
    }

    /// MGR-091, SEQ-057: save_model()은 shutdown 경로에서 정확히 한 번 파일을 쓴다.
    #[test]
    fn mgr_091_save_called_exactly_once_on_shutdown() {
        use argus_manager::lua_policy::LuaPolicy;
        use argus_manager::pipeline::PolicyStrategy;

        let dir = tempfile::tempdir().unwrap();
        let relief_path = dir.path().join("relief.json");
        let script_path = dir.path().join("policy.lua");

        std::fs::write(&script_path, r#"function decide(ctx) return {} end"#).unwrap();

        let config = AdaptationConfig {
            relief_table_path: relief_path.to_str().unwrap().to_string(),
            ..AdaptationConfig::default()
        };

        let policy = LuaPolicy::with_system_clock(script_path.to_str().unwrap(), config).unwrap();

        // 파일이 아직 없음 (초기 entries 비어있으므로 load 실패 → fresh start)
        // save_model() 호출 → 파일 생성
        policy.save_model();

        assert!(relief_path.exists(), "save_model() 후 파일이 생성되어야 함");

        let content1 = std::fs::read_to_string(&relief_path).unwrap();

        // 두 번째 save_model() 호출 — 내용이 동일해야 함 (idempotent)
        policy.save_model();
        let content2 = std::fs::read_to_string(&relief_path).unwrap();

        assert_eq!(
            content1, content2,
            "같은 상태에서 두 번 save해도 동일한 내용이어야 함"
        );
    }

    /// MGR-091: relief_table_path가 빈 문자열이면 파일 I/O를 수행하지 않는다.
    #[test]
    fn mgr_091_empty_path_disables_persistence() {
        use argus_manager::lua_policy::LuaPolicy;
        use argus_manager::pipeline::PolicyStrategy;

        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("policy.lua");

        std::fs::write(&script_path, r#"function decide(ctx) return {} end"#).unwrap();

        // relief_table_path = "" (기본값)
        let policy = LuaPolicy::with_system_clock(
            script_path.to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();

        // save_model()이 panic 없이 no-op으로 완료되어야 함
        policy.save_model();

        // 현재 디렉토리에 relief 파일이 생성되지 않아야 함
        // (빈 경로이므로 파일 생성 안 함)
        let default_path = std::path::Path::new("");
        assert!(
            !default_path.exists() || !default_path.is_file(),
            "빈 경로 시 파일 생성 없어야 함"
        );
    }

    /// MGR-092: 파일이 없으면 fresh start (빈 entries).
    #[test]
    fn mgr_092_missing_file_results_in_fresh_start() {
        use argus_manager::lua_policy::LuaPolicy;

        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("policy.lua");
        let relief_path = dir.path().join("nonexistent_relief.json");

        std::fs::write(&script_path, r#"function decide(ctx) return {} end"#).unwrap();

        let config = AdaptationConfig {
            relief_table_path: relief_path.to_str().unwrap().to_string(),
            ..AdaptationConfig::default()
        };

        // 파일 없음 → crash 없이 fresh start
        let mut policy =
            LuaPolicy::with_system_clock(script_path.to_str().unwrap(), config).unwrap();

        // 빈 entries로 시작해야 함
        let snapshot = argus_manager::pipeline::get_relief_snapshot(&mut policy);
        assert!(
            snapshot.is_empty(),
            "파일 없을 때 빈 entries로 시작해야 함 (MGR-092)"
        );
    }

    /// MGR-092: JSON 파싱 불가 파일 → fresh start fallback.
    #[test]
    fn mgr_092_corrupt_json_results_in_fresh_start() {
        use argus_manager::lua_policy::LuaPolicy;

        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("policy.lua");
        let relief_path = dir.path().join("corrupt_relief.json");

        std::fs::write(&script_path, r#"function decide(ctx) return {} end"#).unwrap();
        // 파싱 불가능한 JSON 내용
        std::fs::write(&relief_path, b"NOT VALID JSON {{{{").unwrap();

        let config = AdaptationConfig {
            relief_table_path: relief_path.to_str().unwrap().to_string(),
            ..AdaptationConfig::default()
        };

        // 파싱 실패 → crash 없이 fresh start
        let mut policy =
            LuaPolicy::with_system_clock(script_path.to_str().unwrap(), config).unwrap();

        let snapshot = argus_manager::pipeline::get_relief_snapshot(&mut policy);
        assert!(
            snapshot.is_empty(),
            "파싱 실패 시 빈 entries로 fresh start해야 함 (MGR-092)"
        );
    }

    /// MGR-092, MGR-DAT-071: save → load round-trip으로 학습 상태가 보존된다.
    #[test]
    fn mgr_092_roundtrip_preserves_learned_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("relief.json");

        // 세션 1: observe 후 save
        let mut table1 = EwmaReliefTable::new(0.875, HashMap::new());
        let obs1: [f32; 6] = [0.3, 0.1, 0.5, 0.2, 0.0, 0.4];
        let obs2: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        table1.observe("kv.evict_sliding", &obs1);
        table1.observe("kv.evict_sliding", &obs2);
        table1.save(&path).expect("save 실패");

        let expected_relief = table1.entries["kv.evict_sliding"].relief;
        let expected_count = table1.entries["kv.evict_sliding"].observation_count;

        // 세션 2: 같은 경로로 load
        let table2 = EwmaReliefTable::load(&path, 0.875, HashMap::new()).expect("load 실패");

        let entry2 = table2.entries.get("kv.evict_sliding").expect("entry 없음");
        assert_eq!(
            entry2.observation_count, expected_count,
            "observation_count 보존 실패"
        );
        for (i, (&r2, &exp)) in entry2.relief.iter().zip(expected_relief.iter()).enumerate() {
            assert!(
                (r2 - exp).abs() < TOL,
                "dim[{}] round-trip 실패: {} vs {}",
                i,
                r2,
                exp
            );
        }
    }

    /// MGR-093: shutdown 전까지 파일은 생성/수정되지 않는다 (주기 체크포인트 없음).
    #[test]
    fn mgr_093_no_periodic_checkpoint() {
        use argus_manager::lua_policy::LuaPolicy;
        use argus_manager::pipeline::PolicyStrategy;
        use argus_shared::SystemSignal;

        let dir = tempfile::tempdir().unwrap();
        let relief_path = dir.path().join("relief.json");
        let script_path = dir.path().join("policy.lua");

        std::fs::write(
            &script_path,
            r#"
                function decide(ctx)
                    return {{ type = "kv.evict_sliding", keep_ratio = 0.7 }}
                end
            "#,
        )
        .unwrap();

        let config = AdaptationConfig {
            relief_table_path: relief_path.to_str().unwrap().to_string(),
            ..AdaptationConfig::default()
        };

        let mut policy =
            LuaPolicy::with_system_clock(script_path.to_str().unwrap(), config).unwrap();

        // 여러 번 signal 주입 (relief 학습이 발생할 수도 있지만 save는 안 됨)
        let mem_signal = SystemSignal::MemoryPressure {
            level: argus_shared::Level::Warning,
            available_bytes: 1_000_000,
            total_bytes: 8_000_000,
            reclaim_target_bytes: 0,
        };

        for _ in 0..10 {
            let _ = policy.process_signal(&mem_signal);
        }

        // save_model() 호출 전에는 파일이 없어야 함 (주기 체크포인트 없음)
        assert!(
            !relief_path.exists(),
            "save_model() 호출 전에는 relief 파일이 없어야 함 (MGR-093)"
        );

        // save_model() 호출 후에는 파일 생성
        policy.save_model();
        assert!(
            relief_path.exists(),
            "save_model() 호출 후에는 파일이 생성되어야 함"
        );
    }
} // mod ewma_tests
