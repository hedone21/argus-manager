/// 시스템 운영 모드
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperatingMode {
    Normal,
    Warning,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operating_mode_ordering() {
        assert!(OperatingMode::Normal < OperatingMode::Warning);
        assert!(OperatingMode::Warning < OperatingMode::Critical);
        assert!(OperatingMode::Critical > OperatingMode::Normal);
    }
}
