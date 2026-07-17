//! Shard file framing (design/16 §3.5.2): magic, then a varint-length-
//! prefixed header message, then a stream of varint-length-prefixed episode
//! messages until EOF. Streaming append — an episode is buffered in memory
//! (games are ~100–400 records, §3.5.4) and flushed whole, so a crash can
//! truncate at most the episode being written, never corrupt earlier ones.

use super::schema::{Episode, OBS_FORMAT_VERSION, OBS_MAGIC, ShardHeader};
use super::wire::{FieldWriter, read_varint, write_varint};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

pub struct ShardWriter {
    out: BufWriter<File>,
}

impl ShardWriter {
    /// Create a shard and write its header. A header without provenance is
    /// unwritable (§3.4) — that invariant lives here on the write side too,
    /// not only in the decoder.
    pub fn create(path: impl AsRef<Path>, header: &ShardHeader) -> io::Result<Self> {
        if header.provenance.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "refusing to write a shard without provenance (design/16 §3.4)",
            ));
        }
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(&OBS_MAGIC)?;
        let mut message = FieldWriter::new();
        header.encode(&mut message);
        let bytes = message.into_bytes();
        write_varint(&mut out, bytes.len() as u64)?;
        out.write_all(&bytes)?;
        Ok(Self { out })
    }

    pub fn append(&mut self, episode: &Episode) -> io::Result<()> {
        let mut message = FieldWriter::new();
        episode.encode(&mut message);
        let bytes = message.into_bytes();
        write_varint(&mut self.out, bytes.len() as u64)?;
        self.out.write_all(&bytes)
    }

    /// Push the `BufWriter`'s buffer to the OS without consuming the writer, so
    /// a long-lived writer (the in-search recorder appends one episode per
    /// game across a whole match) makes each flushed episode crash-durable
    /// rather than holding it until `finish`. Whole episodes only ever land at
    /// a message boundary (§3.5.2), so a flush here never splits one.
    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }

    pub fn finish(mut self) -> io::Result<()> {
        self.out.flush()
    }
}

pub struct ShardReader {
    input: BufReader<File>,
    header: ShardHeader,
}

impl ShardReader {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut input = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 6];
        input.read_exact(&mut magic)?;
        if magic != OBS_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not an obs shard (bad magic)",
            ));
        }
        let header_bytes = read_message(&mut input)?
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "shard has no header"))?;
        let header = ShardHeader::decode(&header_bytes)?;
        // Additive changes never bump the version (§3.5.2), so a HIGHER
        // version than ours is by definition a breaking wire change: refuse
        // rather than misread.
        if header.format_version > OBS_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "shard format_version {} is newer than this reader's {}",
                    header.format_version, OBS_FORMAT_VERSION
                ),
            ));
        }
        Ok(Self { input, header })
    }

    pub fn header(&self) -> &ShardHeader {
        &self.header
    }

    /// The next episode, or `None` at clean EOF. A partial trailing episode
    /// (crash mid-append) surfaces as an error, not a silent truncation.
    pub fn next_episode(&mut self) -> io::Result<Option<Episode>> {
        match read_message(&mut self.input)? {
            Some(bytes) => Ok(Some(Episode::decode(&bytes)?)),
            None => Ok(None),
        }
    }
}

/// One varint-length-prefixed message; `None` on clean EOF at a boundary.
fn read_message(input: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    // Distinguish "clean EOF" (no length byte at all) from truncation.
    let mut first = [0u8; 1];
    if input.read(&mut first)? == 0 {
        return Ok(None);
    }
    let len = if first[0] & 0x80 == 0 {
        u64::from(first[0])
    } else {
        // Multi-byte varint: stitch the already-consumed byte back on.
        let mut rest = Vec::with_capacity(10);
        rest.push(first[0]);
        loop {
            let mut byte = [0u8; 1];
            input.read_exact(&mut byte)?;
            rest.push(byte[0]);
            if byte[0] & 0x80 == 0 {
                break;
            }
        }
        read_varint(&mut rest.as_slice())?
    };
    let len = usize::try_from(len)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "message length overflows"))?;
    // Bound the allocation BEFORE trusting the length: a flipped bit in a
    // length prefix must surface as InvalidData, not an allocation abort
    // (D33 round 1, reproduced with a crafted 14-byte shard). 256 MiB is
    // ~100x the largest sane episode.
    const MAX_MESSAGE_LEN: usize = 256 * 1024 * 1024;
    if len > MAX_MESSAGE_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("message length {len} exceeds the {MAX_MESSAGE_LEN}-byte cap"),
        ));
    }
    let mut bytes = vec![0u8; len];
    input.read_exact(&mut bytes)?;
    Ok(Some(bytes))
}
