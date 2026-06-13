//! GPU telemetry provider abstraction.
//!
//! Manager가 GPU frequency/utilization을 읽는 방식을 플랫폼별로 추상화한다.
//! Android (Adreno/Mali) — sysfs 단일 파일
//! Jetson (Tegra GV11b, JetPack 5.x) — devfreq cur_freq 파일 + `tegrastats` 서브프로세스 스트리밍
//! x86 호스트 — Null (항상 None)
//!
//! util과 freq을 분리한 이유: Jetson에서 두 소스는 완전히 다른 메커니즘이라 실패 도메인이
//! 독립적이다. 하나의 read() 튜플로 묶으면 한쪽 실패가 다른 쪽을 오염시킨다.
//!
//! Factory (`build_provider`)가 `GpuBackend` 설정을 받아 실제 구현을 반환하며,
//! `GpuBackend::Auto`일 때만 `detect_backend`를 통해 플랫폼을 자동 감지한다.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::thread::JoinHandle;

use crate::config::GpuBackend;

/// Adreno/Mali sysfs 후보 경로 — 존재하는 첫 파일을 사용한다.
pub(crate) const SYSFS_UTIL_CANDIDATES: &[&str] = &[
    "/sys/kernel/gpu/gpu_busy_percentage",
    "/sys/devices/platform/kgsl-3d0.0/kgsl/kgsl-3d0/gpu_busy_percentage",
    "/sys/class/misc/mali0/device/utilization",
];

pub(crate) const SYSFS_FREQ_CANDIDATES: &[&str] = &["/sys/class/kgsl/kgsl-3d0/gpuclk"];

/// GPU telemetry를 제공하는 추상화.
///
/// `&mut self`를 받는 이유: `JetsonGpuProvider`가 내부적으로 child 프로세스/atomic을
/// 소유하는 상태있는 구현이며, 향후 rate-limit 캐싱 등을 삽입할 수 있도록 한다.
pub trait GpuTelemetryProvider: Send {
    /// 현재 GPU 사용률 (0~100%). None = 미지원 또는 측정 불가.
    fn util_pct(&mut self) -> Option<f64>;

    /// 현재 GPU 주파수 (Hz). None = 미지원.
    fn freq_hz(&mut self) -> Option<u64>;

    /// 로그/디버그용 단문 식별자.
    fn describe(&self) -> &str;
}

// ---- Null provider ----------------------------------------------------------

/// GPU가 없거나 미지원 플랫폼용 — 항상 None.
pub struct NullGpuProvider;

impl GpuTelemetryProvider for NullGpuProvider {
    fn util_pct(&mut self) -> Option<f64> {
        None
    }
    fn freq_hz(&mut self) -> Option<u64> {
        None
    }
    fn describe(&self) -> &str {
        "null"
    }
}

// ---- Sysfs provider (Adreno / Mali) ----------------------------------------

/// sysfs 단일 파일에서 0~100 정수를 읽는 provider.
/// freq은 옵션 경로 (Adreno kgsl-3d0/gpuclk) 하나만 지원.
pub struct SysfsGpuProvider {
    util_paths: Vec<String>,
    freq_paths: Vec<String>,
    desc: String,
}

impl SysfsGpuProvider {
    /// 기본 후보 목록으로 생성 (Adreno/Mali 순차 시도).
    pub fn with_defaults() -> Self {
        Self {
            util_paths: SYSFS_UTIL_CANDIDATES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            freq_paths: SYSFS_FREQ_CANDIDATES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            desc: "sysfs(adreno|mali)".into(),
        }
    }

    /// 커스텀 util 경로 하나만 사용 (backward-compat for `gpu_sysfs_path`).
    pub fn with_custom_util(path: String) -> Self {
        Self {
            util_paths: vec![path.clone()],
            freq_paths: SYSFS_FREQ_CANDIDATES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            desc: format!("sysfs(custom={path})"),
        }
    }
}

impl GpuTelemetryProvider for SysfsGpuProvider {
    fn util_pct(&mut self) -> Option<f64> {
        for path in &self.util_paths {
            if let Ok(content) = std::fs::read_to_string(path)
                && let Ok(v) = content.trim().parse::<f64>()
            {
                return Some(v.clamp(0.0, 100.0));
            }
        }
        None
    }

    fn freq_hz(&mut self) -> Option<u64> {
        for path in &self.freq_paths {
            if let Ok(content) = std::fs::read_to_string(path)
                && let Ok(v) = content.trim().parse::<u64>()
            {
                return Some(v);
            }
        }
        None
    }

    fn describe(&self) -> &str {
        &self.desc
    }
}

// ---- Jetson provider (Tegra) -----------------------------------------------

/// `tegrastats` child에서 line을 읽는 source. 테스트에서 `VecLineSource`로 교체 가능.
pub(crate) trait LineSource: Send {
    /// Blocking line read. EOF/에러 시 None → reader 스레드 종료.
    fn next_line(&mut self) -> Option<String>;
}

/// `tegrastats --interval 300` child 프로세스 래퍼.
struct TegrastatsChild {
    child: std::process::Child,
    reader: BufReader<std::process::ChildStdout>,
}

impl TegrastatsChild {
    fn spawn(bin: &str) -> anyhow::Result<Self> {
        use std::process::{Command, Stdio};
        let mut child = Command::new(bin)
            .args(["--interval", "300"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .stdin(Stdio::null())
            .spawn()
            .map_err(|e| anyhow::anyhow!("spawn '{bin}' failed: {e}"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("tegrastats: missing stdout pipe"))?;
        Ok(Self {
            child,
            reader: BufReader::new(stdout),
        })
    }
}

impl LineSource for TegrastatsChild {
    fn next_line(&mut self) -> Option<String> {
        let mut buf = String::new();
        match self.reader.read_line(&mut buf) {
            Ok(0) => None, // EOF
            Ok(_) => Some(buf),
            Err(_) => None,
        }
    }
}

impl Drop for TegrastatsChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct JetsonGpuProvider {
    freq_path: Option<PathBuf>,
    util_percent: Arc<AtomicU8>,
    util_seen: Arc<AtomicBool>, // 최소 1 sample 수신 여부
    shutdown: Arc<AtomicBool>,
    reader: Option<JoinHandle<()>>,
    desc: String,
}

impl JetsonGpuProvider {
    /// 기본 생성자 — tegrastats 바이너리 spawn 시도. 실패 시 Err.
    pub fn new(freq_path: Option<PathBuf>, tegrastats_bin: &str) -> anyhow::Result<Self> {
        let child = TegrastatsChild::spawn(tegrastats_bin)?;
        Ok(Self::new_with_source(freq_path, Box::new(child), "jetson(gv11b+tegrastats)").0)
    }

    /// freq만 지원하는 경량 생성자 — tegrastats 바이너리 없을 때 폴백.
    pub fn freq_only(freq_path: PathBuf) -> Self {
        Self {
            freq_path: Some(freq_path),
            util_percent: Arc::new(AtomicU8::new(0)),
            util_seen: Arc::new(AtomicBool::new(false)),
            shutdown: Arc::new(AtomicBool::new(false)),
            reader: None,
            desc: "jetson(gv11b,freq-only)".into(),
        }
    }

    /// 테스트/확장용: 임의의 LineSource와 descriptor로 생성.
    ///
    /// 두 번째 값으로 reader 스레드가 atomic에 load한 뒤 external code에서 관찰할 수 있도록
    /// `Arc<AtomicU8>`를 함께 반환한다 (테스트에서 첫 sample 도달 확인용).
    pub(crate) fn new_with_source(
        freq_path: Option<PathBuf>,
        mut source: Box<dyn LineSource>,
        desc: &str,
    ) -> (Self, Arc<AtomicU8>) {
        let util_percent = Arc::new(AtomicU8::new(0));
        let util_seen = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));

        let util_percent_bg = Arc::clone(&util_percent);
        let util_seen_bg = Arc::clone(&util_seen);
        let shutdown_bg = Arc::clone(&shutdown);

        let reader = std::thread::spawn(move || {
            while !shutdown_bg.load(Ordering::Relaxed) {
                match source.next_line() {
                    Some(line) => {
                        if let Some(pct) = parse_gr3d_freq(&line) {
                            util_percent_bg.store(pct, Ordering::Relaxed);
                            util_seen_bg.store(true, Ordering::Relaxed);
                        }
                    }
                    None => break, // EOF or I/O error
                }
            }
        });

        let util_observe = Arc::clone(&util_percent);
        let provider = Self {
            freq_path,
            util_percent,
            util_seen,
            shutdown,
            reader: Some(reader),
            desc: desc.to_string(),
        };
        (provider, util_observe)
    }
}

impl GpuTelemetryProvider for JetsonGpuProvider {
    fn util_pct(&mut self) -> Option<f64> {
        self.reader.as_ref()?;
        if !self.util_seen.load(Ordering::Relaxed) {
            return None; // 첫 sample 이전 → 값 신뢰 불가
        }
        Some(self.util_percent.load(Ordering::Relaxed) as f64)
    }

    fn freq_hz(&mut self) -> Option<u64> {
        let path = self.freq_path.as_ref()?;
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
    }

    fn describe(&self) -> &str {
        &self.desc
    }
}

impl Drop for JetsonGpuProvider {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.reader.take() {
            // LineSource가 Drop될 때 child가 kill → read_line이 EOF로 풀려 스레드 종료.
            // 다만 reader 스레드가 source를 소유하므로 shutdown flag만으로는 다음 line을
            // 기다리며 blocking 중일 수 있다 → best-effort join. panic 방지용 ok().
            let _ = handle.join();
        }
    }
}

/// tegrastats line에서 `GR3D_FREQ XX%`의 XX를 추출 (0~100).
fn parse_gr3d_freq(line: &str) -> Option<u8> {
    let key = "GR3D_FREQ ";
    let start = line.find(key)? + key.len();
    let tail = &line[start..];
    let num_end = tail.find('%')?;
    let num_str = tail[..num_end].trim();
    let v: u32 = num_str.parse().ok()?;
    Some(v.min(100) as u8)
}

/// 테스트/시뮬레이터용 편의 함수 — GPU가 없는 공유 provider를 반환한다.
pub fn shared_null() -> std::sync::Arc<std::sync::Mutex<Box<dyn GpuTelemetryProvider>>> {
    std::sync::Arc::new(std::sync::Mutex::new(Box::new(NullGpuProvider)))
}

// ---- Detection + factory ----------------------------------------------------

/// sysfs root를 받아 사용 가능한 백엔드를 결정한다. 테스트에선 tempdir을 주입.
pub fn detect_backend(sys_root: &Path) -> GpuBackend {
    // 1. Jetson (Tegra) — devfreq 하위에 gv11b/ga10b 등 이름 엔트리
    let devfreq = sys_root.join("class/devfreq");
    if let Ok(entries) = std::fs::read_dir(&devfreq) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_s = name.to_string_lossy();
            if name_s.contains("gv11b") || name_s.contains("ga10b") {
                return GpuBackend::Jetson;
            }
        }
    }
    // 2. Adreno/Mali sysfs
    for p in SYSFS_UTIL_CANDIDATES {
        // sys_root 기반 경로로 변환: "/sys/..." → "<sys_root>/..."
        let rel = p.strip_prefix("/sys/").unwrap_or(p);
        if sys_root.join(rel).exists() {
            return GpuBackend::Sysfs;
        }
    }
    // 3. Nothing
    GpuBackend::Null
}

/// `GpuBackend` 설정을 실제 provider로 빌드한다. Auto는 `/sys`에 대해 감지한다.
pub fn build_provider(backend: &GpuBackend) -> Box<dyn GpuTelemetryProvider> {
    let resolved = match backend {
        GpuBackend::Auto => detect_backend(Path::new("/sys")),
        other => other.clone(),
    };
    match resolved {
        GpuBackend::Null => Box::new(NullGpuProvider),
        GpuBackend::Sysfs | GpuBackend::Auto => Box::new(SysfsGpuProvider::with_defaults()),
        GpuBackend::CustomSysfs { path } => Box::new(SysfsGpuProvider::with_custom_util(path)),
        GpuBackend::Jetson => {
            let freq_path = discover_jetson_freq_path(Path::new("/sys"));
            match JetsonGpuProvider::new(freq_path.clone(), "tegrastats") {
                Ok(p) => Box::new(p),
                Err(e) => {
                    log::warn!(
                        "tegrastats not available ({e}); falling back to Jetson freq-only mode"
                    );
                    match freq_path {
                        Some(fp) => Box::new(JetsonGpuProvider::freq_only(fp)),
                        None => Box::new(NullGpuProvider),
                    }
                }
            }
        }
        GpuBackend::JetsonExplicit {
            freq_path,
            tegrastats_bin,
        } => {
            let fp = PathBuf::from(&freq_path);
            let bin = tegrastats_bin.as_deref().unwrap_or("tegrastats");
            match JetsonGpuProvider::new(Some(fp.clone()), bin) {
                Ok(p) => Box::new(p),
                Err(e) => {
                    log::warn!("tegrastats spawn failed ({e}); freq-only mode");
                    Box::new(JetsonGpuProvider::freq_only(fp))
                }
            }
        }
    }
}

/// `/sys/class/devfreq/<name>/cur_freq` 경로 중 gv11b/ga10b 포함 엔트리를 찾는다.
fn discover_jetson_freq_path(sys_root: &Path) -> Option<PathBuf> {
    let devfreq = sys_root.join("class/devfreq");
    let entries = std::fs::read_dir(&devfreq).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if name_s.contains("gv11b") || name_s.contains("ga10b") {
            return Some(devfreq.join(&*name_s).join("cur_freq"));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    // ---- parse_gr3d_freq ----
    #[test]
    fn parses_gr3d_freq_basic() {
        let line = "04-23-2026 RAM 6421/30991MB CPU [0%@2265] EMC_FREQ 0% GR3D_FREQ 42% GPU@37C";
        assert_eq!(parse_gr3d_freq(line), Some(42));
    }

    #[test]
    fn parses_gr3d_freq_edge_values() {
        assert_eq!(parse_gr3d_freq("... GR3D_FREQ 0% ..."), Some(0));
        assert_eq!(parse_gr3d_freq("... GR3D_FREQ 100% ..."), Some(100));
    }

    #[test]
    fn parse_gr3d_freq_clamps_over_100() {
        assert_eq!(parse_gr3d_freq("... GR3D_FREQ 150% ..."), Some(100));
    }

    #[test]
    fn parse_gr3d_freq_missing_returns_none() {
        assert_eq!(parse_gr3d_freq("RAM 1/2 CPU 5%"), None);
    }

    // ---- detect_backend ----
    #[test]
    fn detect_prefers_jetson_when_gv11b_devfreq_present() {
        let root = TempDir::new().unwrap();
        let devfreq = root.path().join("class/devfreq/17000000.gv11b");
        fs::create_dir_all(&devfreq).unwrap();
        fs::write(devfreq.join("cur_freq"), "1377000000\n").unwrap();

        assert!(matches!(detect_backend(root.path()), GpuBackend::Jetson));
    }

    #[test]
    fn detect_falls_back_to_sysfs_when_adreno_candidate_exists() {
        let root = TempDir::new().unwrap();
        // /sys/kernel/gpu/gpu_busy_percentage → relative "kernel/gpu/gpu_busy_percentage"
        let p = root.path().join("kernel/gpu");
        fs::create_dir_all(&p).unwrap();
        fs::write(p.join("gpu_busy_percentage"), "50\n").unwrap();

        assert!(matches!(detect_backend(root.path()), GpuBackend::Sysfs));
    }

    #[test]
    fn detect_returns_null_when_nothing_present() {
        let root = TempDir::new().unwrap();
        assert!(matches!(detect_backend(root.path()), GpuBackend::Null));
    }

    // ---- SysfsGpuProvider ----
    #[test]
    fn sysfs_provider_reads_util_from_file() {
        let f = tempfile::NamedTempFile::new().unwrap();
        fs::write(f.path(), "67\n").unwrap();
        let mut p = SysfsGpuProvider {
            util_paths: vec![f.path().to_string_lossy().into_owned()],
            freq_paths: vec![],
            desc: "t".into(),
        };
        assert_eq!(p.util_pct(), Some(67.0));
    }

    #[test]
    fn sysfs_provider_clamps_out_of_range() {
        let f = tempfile::NamedTempFile::new().unwrap();
        fs::write(f.path(), "150\n").unwrap();
        let mut p = SysfsGpuProvider {
            util_paths: vec![f.path().to_string_lossy().into_owned()],
            freq_paths: vec![],
            desc: "t".into(),
        };
        assert_eq!(p.util_pct(), Some(100.0));
    }

    #[test]
    fn sysfs_provider_returns_none_when_no_path_matches() {
        let mut p = SysfsGpuProvider {
            util_paths: vec!["/nonexistent/gpu_busy".into()],
            freq_paths: vec![],
            desc: "t".into(),
        };
        assert_eq!(p.util_pct(), None);
    }

    #[test]
    fn sysfs_provider_reads_freq_when_present() {
        let f = tempfile::NamedTempFile::new().unwrap();
        fs::write(f.path(), "585000000\n").unwrap();
        let mut p = SysfsGpuProvider {
            util_paths: vec![],
            freq_paths: vec![f.path().to_string_lossy().into_owned()],
            desc: "t".into(),
        };
        assert_eq!(p.freq_hz(), Some(585000000));
    }

    #[test]
    fn sysfs_provider_freq_none_when_absent() {
        let mut p = SysfsGpuProvider {
            util_paths: vec![],
            freq_paths: vec!["/nonexistent/freq".into()],
            desc: "t".into(),
        };
        assert_eq!(p.freq_hz(), None);
    }

    // ---- NullGpuProvider ----
    #[test]
    fn null_provider_always_none() {
        let mut p = NullGpuProvider;
        assert_eq!(p.util_pct(), None);
        assert_eq!(p.freq_hz(), None);
    }

    // ---- JetsonGpuProvider with mock LineSource ----
    struct VecLineSource {
        lines: std::sync::Mutex<std::collections::VecDeque<String>>,
    }

    impl VecLineSource {
        fn new(lines: &[&str]) -> Self {
            Self {
                lines: std::sync::Mutex::new(lines.iter().map(|s| s.to_string()).collect()),
            }
        }
    }

    impl LineSource for VecLineSource {
        fn next_line(&mut self) -> Option<String> {
            let mut guard = self.lines.lock().unwrap();
            if let Some(s) = guard.pop_front() {
                Some(s)
            } else {
                // 고갈되면 잠깐 sleep → 테스트 안정화 (shutdown 도달까지)
                drop(guard);
                std::thread::sleep(Duration::from_millis(5));
                Some(String::new())
            }
        }
    }

    fn wait_util(observe: &Arc<AtomicU8>, expected: u8, max_ms: u64) -> bool {
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(max_ms) {
            if observe.load(Ordering::Relaxed) == expected {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }

    #[test]
    fn jetson_provider_parses_gr3d_line() {
        let src = VecLineSource::new(&["RAM 1/2 CPU [0%] GR3D_FREQ 42% GPU@37C"]);
        let (mut p, obs) = JetsonGpuProvider::new_with_source(None, Box::new(src), "test");
        assert!(wait_util(&obs, 42, 500), "sample never reached atomic");
        assert_eq!(p.util_pct(), Some(42.0));
        assert_eq!(p.freq_hz(), None);
    }

    #[test]
    fn jetson_provider_ignores_malformed_lines() {
        let src = VecLineSource::new(&["no match here", "also no", "\n"]);
        let (mut p, _obs) = JetsonGpuProvider::new_with_source(None, Box::new(src), "test");
        std::thread::sleep(Duration::from_millis(50));
        // util_seen이 false이므로 None 반환
        assert_eq!(p.util_pct(), None);
    }

    #[test]
    fn jetson_provider_reads_freq_file() {
        let f = tempfile::NamedTempFile::new().unwrap();
        fs::write(f.path(), "1377000000\n").unwrap();
        let src = VecLineSource::new(&[]);
        let (mut p, _obs) =
            JetsonGpuProvider::new_with_source(Some(f.path().to_path_buf()), Box::new(src), "test");
        assert_eq!(p.freq_hz(), Some(1377000000));
    }

    #[test]
    fn jetson_provider_drop_joins_cleanly() {
        let src = VecLineSource::new(&["GR3D_FREQ 10%"]);
        let (p, _obs) = JetsonGpuProvider::new_with_source(None, Box::new(src), "test");
        drop(p); // 패닉 없이 리턴하면 성공
    }

    #[test]
    fn jetson_freq_only_provider() {
        let f = tempfile::NamedTempFile::new().unwrap();
        fs::write(f.path(), "828750000\n").unwrap();
        let mut p = JetsonGpuProvider::freq_only(f.path().to_path_buf());
        assert_eq!(p.util_pct(), None);
        assert_eq!(p.freq_hz(), Some(828750000));
    }

    #[test]
    fn discover_jetson_freq_path_finds_gv11b() {
        let root = TempDir::new().unwrap();
        let dir = root.path().join("class/devfreq/17000000.gv11b");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cur_freq"), "1\n").unwrap();
        let found = discover_jetson_freq_path(root.path());
        assert_eq!(found, Some(dir.join("cur_freq")));
    }

    #[test]
    fn discover_jetson_freq_path_returns_none_when_no_devfreq() {
        let root = TempDir::new().unwrap();
        assert_eq!(discover_jetson_freq_path(root.path()), None);
    }
}
