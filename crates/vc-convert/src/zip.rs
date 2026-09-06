//! Bounded PTH ZIP access. ZIP/ZIP64 structure and CRC validation belong to `zip`,
//! not a second parser here. Only requested entries are inflated, never extracted.
use std::io::{Cursor, Read};
use std::path::Path;

use ::zip::CompressionMethod;
use anyhow::{anyhow, bail, Context, Result};

// These bound archive I/O, not subsequent pickle objects, widened tensors or ONNX
// graphs. Keep the file and in-memory entry points on the same input limit.
#[derive(Clone, Copy)]
struct Limits {
    input: u64,
    entries: usize,
    total: u64,
    entry: u64,
    pickle: u64,
}
const LIMITS: Limits = Limits {
    input: 2 * 1024 * 1024 * 1024,
    entries: 100_000,
    total: 2 * 1024 * 1024 * 1024,
    entry: 1024 * 1024 * 1024,
    pickle: 64 * 1024 * 1024,
};

/// Do not reserve using untrusted metadata or use unbounded read_to_end. Probe
/// one byte beyond the remaining budget to distinguish exact EOF from overflow.
fn read_bounded(reader: &mut impl Read, limit: u64, consumed: &mut u64) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let remaining = limit.saturating_sub(out.len() as u64);
        let want = remaining.saturating_add(1).min(buf.len() as u64) as usize;
        let n = match reader.read(&mut buf[..want]) {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            other => other?,
        };
        if n == 0 {
            return Ok(out);
        }
        *consumed = consumed
            .checked_add(n as u64)
            .ok_or_else(|| anyhow!("read size overflow"))?;
        if n as u64 > remaining {
            bail!("read exceeds limit of {limit} bytes");
        }
        out.try_reserve(n)
            .context("unable to allocate read buffer")?;
        out.extend_from_slice(&buf[..n]);
    }
}

pub(crate) fn read_pth_file(path: &Path) -> Result<Vec<u8>> {
    read_file_with_limit(path, LIMITS.input)
}
fn read_file_with_limit(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let read = || -> Result<Vec<u8>> {
        let mut file = std::fs::File::open(path)?;
        if file.metadata()?.len() > limit {
            bail!("input exceeds limit of {limit} bytes");
        }
        read_bounded(&mut file, limit, &mut 0)
    };
    read().with_context(|| format!("failed to read PTH {}", path.display()))
}

pub(crate) struct ZipArchive<'a> {
    inner: ::zip::ZipArchive<Cursor<&'a [u8]>>,
    limits: Limits,
    consumed: u64,
}
impl<'a> ZipArchive<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        Self::with_limits(data, LIMITS)
    }

    fn with_limits(data: &'a [u8], limits: Limits) -> Result<Self> {
        if data.len() as u64 > limits.input {
            bail!("PTH input exceeds limit of {} bytes", limits.input);
        }
        let mut inner =
            ::zip::ZipArchive::new(Cursor::new(data)).context("invalid PTH ZIP archive")?;
        if inner.len() > limits.entries {
            bail!("ZIP entry count exceeds limit of {}", limits.entries);
        }
        let mut total = 0u64;
        for i in 0..inner.len() {
            // Raw access inspects metadata without inflating even unused entries.
            let file = inner
                .by_index_raw(i)
                .context("invalid ZIP entry metadata")?;
            if file.size() > limits.entry {
                bail!(
                    "ZIP entry {} exceeds limit of {} bytes",
                    file.name(),
                    limits.entry
                );
            }
            total = total
                .checked_add(file.size())
                .ok_or_else(|| anyhow!("ZIP declared size overflow at {}", file.name()))?;
            if total > limits.total {
                bail!(
                    "ZIP declared total exceeds limit of {} bytes at {}",
                    limits.total,
                    file.name()
                );
            }
        }
        Ok(Self {
            inner,
            limits,
            consumed: 0,
        })
    }
    pub fn entry_names(&self) -> impl Iterator<Item = &str> {
        (0..self.inner.len()).filter_map(|i| self.inner.name_for_index(i))
    }
    pub fn has_entry(&self, name: &str) -> bool {
        self.inner.index_for_name(name).is_some()
    }
    pub fn read_pickle(&mut self, name: &str) -> Result<Vec<u8>> {
        self.read_entry(name, self.limits.pickle.min(self.limits.entry))
    }
    pub fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        self.read_entry(name, self.limits.entry)
    }
    fn read_entry(&mut self, name: &str, entry_limit: u64) -> Result<Vec<u8>> {
        let mut read = || -> Result<Vec<u8>> {
            let index = self
                .inner
                .index_for_name(name)
                .ok_or_else(|| anyhow!("entry not found"))?;
            {
                let file = self.inner.by_index_raw(index)?;
                if file.encrypted() {
                    bail!("encrypted entries are unsupported");
                }
                if !matches!(
                    file.compression(),
                    CompressionMethod::Stored | CompressionMethod::Deflated
                ) {
                    bail!(
                        "unsupported ZIP compression method {:?}",
                        file.compression()
                    );
                }
            }
            let mut file = self.inner.by_index(index)?;
            let remaining = self
                .limits
                .total
                .checked_sub(self.consumed)
                .ok_or_else(|| anyhow!("ZIP cumulative read limit exceeded"))?;
            let limit = entry_limit.min(remaining);
            if file.size() > limit {
                bail!("declared size exceeds entry or remaining cumulative limit of {limit} bytes");
            }
            let expected = file.size();
            // Read through EOF so zip's CRC check executes. Include repeated reads
            // in the budget; storage references must not bypass the archive cap.
            let out = read_bounded(&mut file, limit, &mut self.consumed)?;
            if out.len() as u64 != expected {
                bail!("ZIP size mismatch: read {}, expected {expected}", out.len());
            }
            Ok(out)
        };
        read().with_context(|| format!("failed to read ZIP entry {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::zip::{write::SimpleFileOptions, ZipWriter};
    use std::io::Write;

    fn fixture(method: CompressionMethod, large: bool) -> Vec<u8> {
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        if large {
            // Force an actual ZIP64 end record without allocating gigabytes or
            // creating thousands of entries; large_file covers local size fields.
            writer.set_raw_zip64_extensible_data_sector(Box::new([]));
        }
        for name in ["archive/data.pkl", "archive/data/0"] {
            writer
                .start_file(
                    name,
                    SimpleFileOptions::default()
                        .compression_method(method)
                        .large_file(large)
                        .with_alignment(64),
                )
                .unwrap();
            writer.write_all(b"abcdefgh").unwrap();
        }
        writer.finish().unwrap().into_inner()
    }
    fn small(data: &[u8]) -> Limits {
        Limits {
            input: data.len() as u64,
            entries: 2,
            total: 16,
            entry: 8,
            pickle: 8,
        }
    }
    fn offset(data: &[u8], signature: u32) -> usize {
        data.windows(4)
            .position(|w| w == signature.to_le_bytes())
            .unwrap()
    }
    fn put32(data: &mut [u8], pos: usize, n: u32) {
        data[pos..pos + 4].copy_from_slice(&n.to_le_bytes());
    }
    fn rejects(data: &[u8]) {
        // A panic naturally fails this test; the public parser must return Err.
        assert!(crate::parse_pth(data).is_err());
    }

    #[test]
    fn reads_stored_deflated_and_zip64_with_local_alignment() {
        for method in [CompressionMethod::Stored, CompressionMethod::Deflated] {
            for large in [false, true] {
                let data = fixture(method, large);
                if large {
                    assert!(data.windows(4).any(|w| w == 0x06064b50u32.to_le_bytes()));
                }
                let mut archive = ZipArchive::with_limits(&data, small(&data)).unwrap();
                assert_eq!(
                    archive.entry_names().collect::<Vec<_>>(),
                    ["archive/data.pkl", "archive/data/0"]
                );
                assert!(archive.has_entry("archive/data/0"));
                assert_eq!(
                    archive.read_pickle("archive/data.pkl").unwrap(),
                    b"abcdefgh"
                );
                assert_eq!(archive.read("archive/data/0").unwrap(), b"abcdefgh");
            }
        }
    }
    #[test]
    fn malformed_zip64_entry_count_returns_error() {
        let mut data = vec![0u8; 98];
        put32(&mut data, 0, 0x06064b50);
        data[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
        put32(&mut data, 56, 0x07064b50);
        put32(&mut data, 76, 0x06054b50);
        data[86..88].copy_from_slice(&u16::MAX.to_le_bytes());
        rejects(&data);
    }
    #[test]
    fn rejects_truncation_and_invalid_offsets() {
        let original = fixture(CompressionMethod::Stored, false);
        for end in [0, 4, 30, original.len() - 1] {
            rejects(&original[..end]);
        }
        let mut data = original;
        let cd = offset(&data, 0x02014b50);
        put32(&mut data, cd + 42, u32::MAX - 1);
        rejects(&data);
    }
    #[test]
    fn rejects_crc_corruption_with_entry_context() {
        for method in [CompressionMethod::Stored, CompressionMethod::Deflated] {
            let mut data = fixture(method, false);
            let cd = offset(&data, 0x02014b50);
            put32(&mut data, 14, 123);
            put32(&mut data, cd + 16, 123);
            let mut archive = ZipArchive::parse(&data).unwrap();
            let error = archive.read_pickle("archive/data.pkl").unwrap_err();
            assert!(format!("{error:#}").contains("archive/data.pkl"));
        }
    }
    #[test]
    fn rejects_unsupported_compression_and_encryption() {
        for encrypted in [false, true] {
            let mut data = fixture(CompressionMethod::Stored, false);
            let cd = offset(&data, 0x02014b50);
            if encrypted {
                data[6] |= 1;
                data[cd + 8] |= 1;
            } else {
                data[8..10].copy_from_slice(&99u16.to_le_bytes());
                data[cd + 10..cd + 12].copy_from_slice(&99u16.to_le_bytes());
            }
            rejects(&data);
        }
    }
    #[test]
    fn checks_input_entry_count_and_declared_size_limits() {
        let data = fixture(CompressionMethod::Stored, false);
        let limits = small(&data);
        assert!(ZipArchive::with_limits(&data, limits).is_ok());
        for too_small in [
            Limits {
                input: limits.input - 1,
                ..limits
            },
            Limits {
                entries: 1,
                ..limits
            },
            Limits {
                total: 15,
                ..limits
            },
            Limits { entry: 7, ..limits },
        ] {
            assert!(ZipArchive::with_limits(&data, too_small).is_err());
        }
        let mut archive = ZipArchive::with_limits(
            &data,
            Limits {
                pickle: 7,
                ..limits
            },
        )
        .unwrap();
        assert!(archive.read_pickle("archive/data.pkl").is_err());
    }
    #[test]
    fn repeated_reads_use_cumulative_budget() {
        let data = fixture(CompressionMethod::Deflated, false);
        let mut archive = ZipArchive::with_limits(&data, small(&data)).unwrap();
        assert_eq!(archive.read("archive/data/0").unwrap().len(), 8);
        assert_eq!(archive.read("archive/data/0").unwrap().len(), 8);
        assert!(archive.read("archive/data/0").is_err());
    }
    #[test]
    fn rejects_declared_size_mismatch() {
        for method in [CompressionMethod::Stored, CompressionMethod::Deflated] {
            for size in [1, 9] {
                let mut data = fixture(method, false);
                let cd = offset(&data, 0x02014b50);
                put32(&mut data, 22, size);
                put32(&mut data, cd + 24, size);
                rejects(&data);
                if let Ok(mut archive) = ZipArchive::parse(&data) {
                    assert!(archive.read_pickle("archive/data.pkl").is_err());
                }
            }
        }
    }
    #[test]
    fn bounded_reader_checks_actual_bytes_and_counter_overflow() {
        let mut count = 0;
        assert_eq!(
            read_bounded(&mut &b"abcd"[..], 4, &mut count).unwrap(),
            b"abcd"
        );
        assert_eq!(count, 4);
        assert!(read_bounded(&mut &b"abcde"[..], 4, &mut count).is_err());
        let mut overflowing = u64::MAX;
        assert!(read_bounded(&mut &b"a"[..], 1, &mut overflowing).is_err());
        assert!(read_bounded(&mut &b""[..], 0, &mut 0).unwrap().is_empty());
    }
    #[test]
    fn file_input_is_bounded_before_reading() {
        let path =
            std::env::temp_dir().join(format!("vc-convert-zip-limit-{}.pth", std::process::id()));
        std::fs::write(&path, b"abcd").unwrap();
        let result = (
            read_file_with_limit(&path, 4),
            read_file_with_limit(&path, 3),
        );
        std::fs::remove_file(&path).unwrap();
        assert_eq!(result.0.unwrap(), b"abcd");
        assert!(result.1.is_err());
    }
}
