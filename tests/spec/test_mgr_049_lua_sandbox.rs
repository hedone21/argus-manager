//! MGR-049: Lua VM 샌드박스 IO 허용 + OS/PACKAGE/DEBUG 차단 Spec 테스트
//!
//! LuaPolicy sandbox 표면을 검증한다:
//! - `io` 라이브러리가 활성화되어 파일 RW가 가능하다.
//! - `os`, `package`(require 포함), `debug` 라이브러리는 차단(nil)된다.
//! - 4MB 메모리 한도가 유지된다.
//! - `reload_script` 후에도 sandbox 표면이 동일하게 유지된다.

#![allow(clippy::needless_doctest_main)]

#[cfg(feature = "lua")]
mod sandbox_tests {
    use std::io::Write;

    use argus_manager::config::AdaptationConfig;
    use argus_manager::lua_policy::LuaPolicy;
    use argus_manager::pipeline::{PolicyStrategy, ReloadablePolicy};
    use argus_shared::{Level, SystemSignal};

    // ─── Helpers ──────────────────────────────────────────────────────────────

    fn write_lua(body: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".lua").tempfile().unwrap();
        f.write_all(body.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    fn new_policy(script_body: &str) -> (LuaPolicy, tempfile::NamedTempFile) {
        let f = write_lua(script_body);
        let p = LuaPolicy::with_system_clock(
            f.path().to_str().unwrap(),
            AdaptationConfig {
                qcf_penalty_weight: 0.0,
                ..AdaptationConfig::default()
            },
        )
        .unwrap();
        (p, f)
    }

    fn dummy_signal() -> SystemSignal {
        SystemSignal::MemoryPressure {
            level: Level::Normal,
            available_bytes: 4 * 1024 * 1024 * 1024,
            total_bytes: 8 * 1024 * 1024 * 1024,
            reclaim_target_bytes: 0,
        }
    }

    // ─── MGR-049-1: io 라이브러리 활성화 ─────────────────────────────────────

    /// MGR-049: `io.open` / `io.write` / `io.close`로 파일 RW가 성공해야 한다.
    #[test]
    fn test_mgr_049_io_lib_available() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let tmp_path = tmp.path().to_str().unwrap().to_owned();

        // top-level에서 io.open으로 쓰기, 결과를 글로벌에 저장
        let script = format!(
            r#"
local f = io.open("{path}", "w")
if f then
    f:write("mgr049")
    f:close()
    io_write_ok = true
else
    io_write_ok = false
end

function decide(ctx) return {{}} end
"#,
            path = tmp_path
        );

        let (mut policy, _f) = new_policy(&script);

        // process_signal 호출로 스크립트 실행 트리거
        let _ = policy.process_signal(&dummy_signal());

        // io 글로벌에서 결과 확인
        let io_ok: bool = policy.lua().globals().get("io_write_ok").unwrap_or(false);
        assert!(io_ok, "io.open/write/close 호출이 성공해야 한다");

        // 파일에 실제로 쓰였는지 확인
        let contents = std::fs::read_to_string(&tmp_path).unwrap_or_default();
        assert_eq!(
            contents, "mgr049",
            "io.write로 파일에 내용이 기록되어야 한다"
        );
    }

    // ─── MGR-049-2: os 라이브러리 차단 ───────────────────────────────────────

    /// MGR-049: `os` 글로벌이 nil이어야 한다.
    #[test]
    fn test_mgr_049_os_lib_blocked() {
        let script = r#"
os_is_nil = (type(os) == "nil")
function decide(ctx) return {} end
"#;
        let (mut policy, _f) = new_policy(script);
        let _ = policy.process_signal(&dummy_signal());

        let result: bool = policy.lua().globals().get("os_is_nil").unwrap_or(false);
        assert!(result, "os 글로벌은 nil이어야 한다 (OS 라이브러리 차단)");
    }

    // ─── MGR-049-3: package / require 차단 ───────────────────────────────────

    /// MGR-049: `require`와 `package` 글로벌이 nil이어야 한다.
    #[test]
    fn test_mgr_049_package_blocked() {
        let script = r#"
require_is_nil = (type(require) == "nil")
package_is_nil = (type(package) == "nil")
function decide(ctx) return {} end
"#;
        let (mut policy, _f) = new_policy(script);
        let _ = policy.process_signal(&dummy_signal());

        let require_nil: bool = policy
            .lua()
            .globals()
            .get("require_is_nil")
            .unwrap_or(false);
        let package_nil: bool = policy
            .lua()
            .globals()
            .get("package_is_nil")
            .unwrap_or(false);
        assert!(require_nil, "require 글로벌은 nil이어야 한다");
        assert!(package_nil, "package 글로벌은 nil이어야 한다");
    }

    // ─── MGR-049-4: debug 라이브러리 차단 ────────────────────────────────────

    /// MGR-049: `debug` 글로벌이 nil이어야 한다.
    #[test]
    fn test_mgr_049_debug_blocked() {
        let script = r#"
debug_is_nil = (type(debug) == "nil")
function decide(ctx) return {} end
"#;
        let (mut policy, _f) = new_policy(script);
        let _ = policy.process_signal(&dummy_signal());

        let debug_nil: bool = policy.lua().globals().get("debug_is_nil").unwrap_or(false);
        assert!(debug_nil, "debug 글로벌은 nil이어야 한다");
    }

    // ─── MGR-049-5: 4MB 메모리 한도 유지 ─────────────────────────────────────

    /// MGR-049: 4MB 초과 할당 시 에러, 작은 할당은 성공해야 한다.
    #[test]
    fn test_mgr_049_memory_limit_4mb_preserved() {
        // 작은 테이블 할당 — 성공해야 한다
        let script_small = r#"
local t = {}
for i = 1, 100 do t[i] = i end
small_alloc_ok = true
function decide(ctx) return {} end
"#;
        let (mut policy, _f) = new_policy(script_small);
        let _ = policy.process_signal(&dummy_signal());

        let small_ok: bool = policy
            .lua()
            .globals()
            .get("small_alloc_ok")
            .unwrap_or(false);
        assert!(small_ok, "소규모 할당은 성공해야 한다");

        // 4MB 초과 할당 — 스크립트 로드 시 에러가 발생해야 한다
        // 5MB 문자열 생성 시도 (string.rep은 메모리 한도 적용 대상)
        let script_large = r#"
local big = string.rep("x", 5 * 1024 * 1024)
function decide(ctx) return {} end
"#;
        let large_tmp = write_lua(script_large);
        let result = LuaPolicy::with_system_clock(
            large_tmp.path().to_str().unwrap(),
            AdaptationConfig {
                qcf_penalty_weight: 0.0,
                ..AdaptationConfig::default()
            },
        );
        assert!(
            result.is_err(),
            "4MB를 초과하는 할당은 LuaPolicy 생성 실패를 유발해야 한다"
        );
    }

    // ─── MGR-049-6: reload_script 후 sandbox 동일성 ──────────────────────────

    /// MGR-049: reload_script 후에도 io 허용 + os 차단이 동일하게 유지된다.
    #[test]
    fn test_mgr_049_reload_script_preserves_sandbox() {
        // 초기 스크립트: 빈 decide
        let initial = write_lua("function decide(ctx) return {} end");
        let mut policy = LuaPolicy::with_system_clock(
            initial.path().to_str().unwrap(),
            AdaptationConfig {
                qcf_penalty_weight: 0.0,
                ..AdaptationConfig::default()
            },
        )
        .unwrap();

        // reload 대상 스크립트: io 사용 + os 차단 확인
        let tmp_out = tempfile::NamedTempFile::new().unwrap();
        let tmp_path = tmp_out.path().to_str().unwrap().to_owned();

        let reload_script = format!(
            r#"
-- io: 파일 쓰기 가능해야 함
local f = io.open("{path}", "w")
if f then
    f:write("reload_ok")
    f:close()
    io_ok = true
else
    io_ok = false
end

-- os: nil이어야 함
os_blocked = (type(os) == "nil")

function decide(ctx) return {{}} end
"#,
            path = tmp_path
        );
        let reload_file = write_lua(&reload_script);

        // reload_script를 PolicyStrategy trait으로 호출
        policy
            .reload_script(reload_file.path())
            .expect("reload_script 실패");

        // reload 후 process_signal 호출로 스크립트 실행
        let _ = policy.process_signal(&dummy_signal());

        // io 파일 쓰기 성공 확인
        let contents = std::fs::read_to_string(&tmp_path).unwrap_or_default();
        assert_eq!(contents, "reload_ok", "reload 후 io.write가 동작해야 한다");

        // os 차단 확인
        let os_blocked: bool = policy.lua().globals().get("os_blocked").unwrap_or(false);
        assert!(os_blocked, "reload 후 os 글로벌은 여전히 nil이어야 한다");
    }
}
