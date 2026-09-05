//! Minimal ZIP reader for PyTorch `.pth` archives.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — where `src/pth-parser.ts` delegates to `fflate.unzipSync`,
//! this module reads the archive directly so stored entries (PyTorch ≥1.6
//! always writes `ZIP_STORED`) come back as zero-copy slices of the input.
//!
//! Scope: read-only, central-directory driven, methods 0 (stored) and
//! 8 (deflate), with ZIP64 offsets/sizes (PyTorch's miniz writer emits ZIP64
//! fields for large checkpoints). Anything else errors — a `.pth` never needs
//! encryption, spanning, or exotic compression.

use std::borrow::Cow;

use anyhow::{anyhow, bail, Context, Result};

const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD64_LOCATOR_SIG: u32 = 0x0706_4b50;
const EOCD64_SIG: u32 = 0x0606_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;
/// EOCD is 22 bytes plus a comment of at most 65535 bytes.
const EOCD_SEARCH_MAX: usize = 22 + 65_535;

pub(crate) struct ZipEntry {
    pub name: String,
    method: u16,
    compressed_size: u64,
    uncompressed_size: u64,
    local_header_offset: u64,
}

pub(crate) struct ZipArchive<'a> {
    data: &'a [u8],
    entries: Vec<ZipEntry>,
}

fn u16_at(data: &[u8], pos: usize) -> Result<u16> {
    let bytes: [u8; 2] = data
        .get(pos..pos + 2)
        .ok_or_else(|| anyhow!("ZIP truncated at offset {pos}"))?
        .try_into()
        .expect("slice length checked");
    Ok(u16::from_le_bytes(bytes))
}

fn u32_at(data: &[u8], pos: usize) -> Result<u32> {
    let bytes: [u8; 4] = data
        .get(pos..pos + 4)
        .ok_or_else(|| anyhow!("ZIP truncated at offset {pos}"))?
        .try_into()
        .expect("slice length checked");
    Ok(u32::from_le_bytes(bytes))
}

fn u64_at(data: &[u8], pos: usize) -> Result<u64> {
    let bytes: [u8; 8] = data
        .get(pos..pos + 8)
        .ok_or_else(|| anyhow!("ZIP truncated at offset {pos}"))?
        .try_into()
        .expect("slice length checked");
    Ok(u64::from_le_bytes(bytes))
}

impl<'a> ZipArchive<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let eocd = find_eocd(data)?;
        let mut entry_count = u64::from(u16_at(data, eocd + 10)?);
        let mut cd_offset = u64::from(u32_at(data, eocd + 16)?);

        // ZIP64: the 16-bit/32-bit EOCD fields saturate and the real values
        // live in the ZIP64 EOCD record, found via the locator just before
        // the EOCD.
        if entry_count == 0xFFFF || cd_offset == 0xFFFF_FFFF {
            let locator = eocd
                .checked_sub(20)
                .ok_or_else(|| anyhow!("ZIP64 archive missing EOCD64 locator"))?;
            if u32_at(data, locator)? != EOCD64_LOCATOR_SIG {
                bail!("ZIP64 archive missing EOCD64 locator");
            }
            let eocd64 = usize::try_from(u64_at(data, locator + 8)?)
                .map_err(|_| anyhow!("ZIP64 EOCD offset out of range"))?;
            if u32_at(data, eocd64)? != EOCD64_SIG {
                bail!("invalid ZIP64 end-of-central-directory record");
            }
            entry_count = u64_at(data, eocd64 + 32)?;
            cd_offset = u64_at(data, eocd64 + 48)?;
        }

        let mut pos = usize::try_from(cd_offset)
            .map_err(|_| anyhow!("central directory offset out of range"))?;
        let mut entries = Vec::with_capacity(usize::try_from(entry_count).unwrap_or(0));
        for _ in 0..entry_count {
            if u32_at(data, pos)? != CENTRAL_SIG {
                bail!("invalid central directory entry at offset {pos}");
            }
            let method = u16_at(data, pos + 10)?;
            let mut compressed_size = u64::from(u32_at(data, pos + 20)?);
            let mut uncompressed_size = u64::from(u32_at(data, pos + 24)?);
            let name_len = usize::from(u16_at(data, pos + 28)?);
            let extra_len = usize::from(u16_at(data, pos + 30)?);
            let comment_len = usize::from(u16_at(data, pos + 32)?);
            let mut local_header_offset = u64::from(u32_at(data, pos + 42)?);
            let name_bytes = data
                .get(pos + 46..pos + 46 + name_len)
                .ok_or_else(|| anyhow!("ZIP truncated in entry name"))?;
            let name = String::from_utf8_lossy(name_bytes).into_owned();

            // ZIP64 extended-information extra field: only the fields whose
            // 32-bit form saturated are present, in this fixed order.
            let mut extra_pos = pos + 46 + name_len;
            let extra_end = extra_pos + extra_len;
            while extra_pos + 4 <= extra_end {
                let id = u16_at(data, extra_pos)?;
                let size = usize::from(u16_at(data, extra_pos + 2)?);
                if id == 0x0001 {
                    let mut field = extra_pos + 4;
                    if uncompressed_size == 0xFFFF_FFFF {
                        uncompressed_size = u64_at(data, field)?;
                        field += 8;
                    }
                    if compressed_size == 0xFFFF_FFFF {
                        compressed_size = u64_at(data, field)?;
                        field += 8;
                    }
                    if local_header_offset == 0xFFFF_FFFF {
                        local_header_offset = u64_at(data, field)?;
                    }
                }
                extra_pos += 4 + size;
            }

            entries.push(ZipEntry {
                name,
                method,
                compressed_size,
                uncompressed_size,
                local_header_offset,
            });
            pos = extra_end + comment_len;
        }

        Ok(ZipArchive { data, entries })
    }

    pub fn entry_names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.name.as_str())
    }

    pub fn has_entry(&self, name: &str) -> bool {
        self.entries.iter().any(|e| e.name == name)
    }

    /// Read one entry. Stored entries borrow from the archive buffer;
    /// deflated entries allocate.
    pub fn read(&self, name: &str) -> Result<Cow<'a, [u8]>> {
        let entry = self
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| anyhow!("ZIP entry not found: {name}"))?;

        // The local header repeats name/extra with potentially different extra
        // fields (PyTorch pads the extra field to 64-byte-align tensor data),
        // so the data offset must come from the local header, not the central
        // directory.
        let lho = usize::try_from(entry.local_header_offset)
            .map_err(|_| anyhow!("local header offset out of range"))?;
        if u32_at(self.data, lho)? != LOCAL_SIG {
            bail!("invalid local file header for ZIP entry {name}");
        }
        let name_len = usize::from(u16_at(self.data, lho + 26)?);
        let extra_len = usize::from(u16_at(self.data, lho + 28)?);
        let data_start = lho + 30 + name_len + extra_len;
        let comp_len = usize::try_from(entry.compressed_size)
            .map_err(|_| anyhow!("compressed size out of range"))?;
        let raw = self
            .data
            .get(data_start..data_start + comp_len)
            .ok_or_else(|| anyhow!("ZIP truncated in data of entry {name}"))?;

        match entry.method {
            0 => Ok(Cow::Borrowed(raw)),
            8 => {
                let inflated = miniz_oxide::inflate::decompress_to_vec(raw)
                    .map_err(|e| anyhow!("failed to inflate ZIP entry {name}: {e}"))?;
                if inflated.len() as u64 != entry.uncompressed_size {
                    bail!(
                        "ZIP entry {name} inflated to {} bytes, expected {}",
                        inflated.len(),
                        entry.uncompressed_size
                    );
                }
                Ok(Cow::Owned(inflated))
            }
            other => bail!("unsupported ZIP compression method {other} for entry {name}"),
        }
    }
}

fn find_eocd(data: &[u8]) -> Result<usize> {
    let search_start = data.len().saturating_sub(EOCD_SEARCH_MAX);
    let mut pos = data
        .len()
        .checked_sub(22)
        .context("file too small for ZIP")?;
    loop {
        if u32_at(data, pos)? == EOCD_SIG {
            return Ok(pos);
        }
        if pos == search_start {
            bail!("ZIP end-of-central-directory record not found");
        }
        pos -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a single-entry ZIP by hand (mirrors onnx_meta's hand-assembled
    /// protobuf test style).
    fn build_zip(name: &str, payload: &[u8], method: u16, stored: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        // Local header
        out.extend_from_slice(&LOCAL_SIG.to_le_bytes());
        out.extend_from_slice(&[20, 0]); // version needed
        out.extend_from_slice(&[0, 0]); // flags
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]); // time+date
        out.extend_from_slice(&[0, 0, 0, 0]); // crc32 (unchecked)
        out.extend_from_slice(&(stored.len() as u32).to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&[4, 0]); // extra len: 4-byte pad like torch alignment
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&[0xEE, 0xEE, 0, 0]); // dummy extra field header
        let data_offset = out.len();
        assert_eq!(data_offset, 30 + name.len() + 4);
        out.extend_from_slice(stored);
        // Central directory
        let cd_offset = out.len();
        out.extend_from_slice(&CENTRAL_SIG.to_le_bytes());
        out.extend_from_slice(&[20, 0, 20, 0]); // versions
        out.extend_from_slice(&[0, 0]); // flags
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]); // time+date
        out.extend_from_slice(&[0, 0, 0, 0]); // crc32
        out.extend_from_slice(&(stored.len() as u32).to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&[0, 0]); // extra len
        out.extend_from_slice(&[0, 0]); // comment len
        out.extend_from_slice(&[0, 0]); // disk number
        out.extend_from_slice(&[0, 0]); // internal attrs
        out.extend_from_slice(&[0, 0, 0, 0]); // external attrs
        out.extend_from_slice(&(0u32).to_le_bytes()); // local header offset
        out.extend_from_slice(name.as_bytes());
        let cd_size = out.len() - cd_offset;
        // EOCD
        out.extend_from_slice(&EOCD_SIG.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]); // disk numbers
        out.extend_from_slice(&1u16.to_le_bytes()); // entries on disk
        out.extend_from_slice(&1u16.to_le_bytes()); // total entries
        out.extend_from_slice(&(cd_size as u32).to_le_bytes());
        out.extend_from_slice(&(cd_offset as u32).to_le_bytes());
        out.extend_from_slice(&[0, 0]); // comment len
        out
    }

    #[test]
    fn reads_stored_entry_zero_copy() {
        let payload = b"hello tensor data";
        let zip = build_zip("archive/data/0", payload, 0, payload);
        let archive = ZipArchive::parse(&zip).unwrap();
        assert!(archive.has_entry("archive/data/0"));
        let read = archive.read("archive/data/0").unwrap();
        assert_eq!(&*read, payload);
        assert!(matches!(read, Cow::Borrowed(_)));
    }

    #[test]
    fn reads_deflated_entry() {
        let payload = vec![42u8; 4096];
        let compressed = miniz_oxide::deflate::compress_to_vec(&payload, 6);
        let zip = build_zip("data.pkl", &payload, 8, &compressed);
        let archive = ZipArchive::parse(&zip).unwrap();
        let read = archive.read("data.pkl").unwrap();
        assert_eq!(&*read, payload.as_slice());
        assert!(matches!(read, Cow::Owned(_)));
    }

    #[test]
    fn rejects_unknown_compression_method() {
        let payload = b"x";
        let zip = build_zip("f", payload, 99, payload);
        let archive = ZipArchive::parse(&zip).unwrap();
        let err = archive.read("f").unwrap_err().to_string();
        assert!(
            err.contains("unsupported ZIP compression method 99"),
            "{err}"
        );
    }

    #[test]
    fn rejects_non_zip() {
        assert!(ZipArchive::parse(b"not a zip file at all......").is_err());
        assert!(ZipArchive::parse(b"").is_err());
    }
}
