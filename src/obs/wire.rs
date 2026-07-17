//! The TLV wire layer of the Observation Store (design/16 §3.5.2).
//!
//! Shaped after the protobuf wire model — a public fact, reimplemented here
//! with no protobuf dependency (D15/D27) — because its single load-bearing
//! property is exactly what a permanent store needs: a reader that meets an
//! unknown tag skips it by wire-type, so fields and episode kinds added years
//! from now never break a reader built today.
//!
//! Wire-types: `0` varint (LEB128; signed values are zigzag-encoded first),
//! `1` fixed64, `2` length-delimited (bytes / nested message / packed
//! repeated), `5` fixed32. A field key is `(tag << 3) | wire_type`, itself a
//! varint. Tags are APPEND-ONLY: never renumbered, never reused, never
//! repurposed (deprecated tags are tombstoned in the schema module).

use std::io::{self, Read, Write};

/// Wire-type discriminants (design/16 §3.5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireType {
    Varint,
    Fixed64,
    LengthDelimited,
    Fixed32,
}

impl WireType {
    fn from_bits(bits: u64) -> io::Result<Self> {
        match bits {
            0 => Ok(WireType::Varint),
            1 => Ok(WireType::Fixed64),
            2 => Ok(WireType::LengthDelimited),
            5 => Ok(WireType::Fixed32),
            other => Err(bad_data(format!("unknown wire type {other}"))),
        }
    }

    fn bits(self) -> u64 {
        match self {
            WireType::Varint => 0,
            WireType::Fixed64 => 1,
            WireType::LengthDelimited => 2,
            WireType::Fixed32 => 5,
        }
    }
}

fn bad_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

// ─── Varint primitives ─────────────────────────────────────────────────────

/// LEB128 varint. At most 10 bytes for a u64.
pub fn write_varint(out: &mut impl Write, mut value: u64) -> io::Result<()> {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            return out.write_all(&[byte]);
        }
        out.write_all(&[byte | 0x80])?;
    }
}

pub fn read_varint(input: &mut impl Read) -> io::Result<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let mut byte = [0u8; 1];
        input.read_exact(&mut byte)?;
        let byte = byte[0];
        if shift == 63 && byte > 1 {
            return Err(bad_data("varint overflows u64".into()));
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift > 63 {
            return Err(bad_data("varint longer than 10 bytes".into()));
        }
    }
}

/// Zigzag: maps signed to unsigned so small negatives stay small on the wire.
pub fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

pub fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

// ─── Field-level writer ────────────────────────────────────────────────────

/// Serializes one message into a buffer; nested messages build their own
/// `FieldWriter` and embed via [`FieldWriter::bytes`].
#[derive(Default)]
pub struct FieldWriter {
    buf: Vec<u8>,
}

impl FieldWriter {
    pub fn new() -> Self {
        Self::default()
    }

    fn key(&mut self, tag: u32, wire: WireType) {
        // Writing into a Vec cannot fail.
        write_varint(&mut self.buf, (u64::from(tag) << 3) | wire.bits()).expect("vec write");
    }

    /// Unsigned varint field.
    pub fn uint(&mut self, tag: u32, value: u64) -> &mut Self {
        self.key(tag, WireType::Varint);
        write_varint(&mut self.buf, value).expect("vec write");
        self
    }

    /// Signed (zigzag) varint field.
    pub fn sint(&mut self, tag: u32, value: i64) -> &mut Self {
        self.uint(tag, zigzag(value))
    }

    /// Bool as varint 0/1.
    pub fn boolean(&mut self, tag: u32, value: bool) -> &mut Self {
        self.uint(tag, u64::from(value))
    }

    /// Fixed 64-bit field (hashes/keys: constant-size beats varint here).
    pub fn fixed64(&mut self, tag: u32, value: u64) -> &mut Self {
        self.key(tag, WireType::Fixed64);
        self.buf.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Length-delimited raw bytes / nested message / UTF-8 string.
    pub fn bytes(&mut self, tag: u32, value: &[u8]) -> &mut Self {
        self.key(tag, WireType::LengthDelimited);
        write_varint(&mut self.buf, value.len() as u64).expect("vec write");
        self.buf.extend_from_slice(value);
        self
    }

    pub fn string(&mut self, tag: u32, value: &str) -> &mut Self {
        self.bytes(tag, value.as_bytes())
    }

    /// Packed repeated unsigned varints (one length-delimited field).
    pub fn packed_uints(&mut self, tag: u32, values: impl IntoIterator<Item = u64>) -> &mut Self {
        let mut packed = Vec::new();
        for value in values {
            write_varint(&mut packed, value).expect("vec write");
        }
        self.bytes(tag, &packed)
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }
}

// ─── Field-level reader ────────────────────────────────────────────────────

/// Iterates `(tag, value)` pairs over one message's bytes. Unknown tags are
/// the CALLER's business: `FieldValue` always carries enough to skip.
pub struct FieldReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

/// One decoded field. Length-delimited payloads borrow from the message.
pub enum FieldValue<'a> {
    Varint(u64),
    Fixed64(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> FieldValue<'a> {
    pub fn as_uint(&self) -> io::Result<u64> {
        match self {
            FieldValue::Varint(v) => Ok(*v),
            _ => Err(bad_data("expected varint field".into())),
        }
    }

    pub fn as_sint(&self) -> io::Result<i64> {
        Ok(unzigzag(self.as_uint()?))
    }

    pub fn as_bool(&self) -> io::Result<bool> {
        Ok(self.as_uint()? != 0)
    }

    pub fn as_fixed64(&self) -> io::Result<u64> {
        match self {
            FieldValue::Fixed64(v) => Ok(*v),
            _ => Err(bad_data("expected fixed64 field".into())),
        }
    }

    pub fn as_bytes(&self) -> io::Result<&'a [u8]> {
        match self {
            FieldValue::Bytes(b) => Ok(b),
            _ => Err(bad_data("expected length-delimited field".into())),
        }
    }

    pub fn as_string(&self) -> io::Result<String> {
        String::from_utf8(self.as_bytes()?.to_vec())
            .map_err(|_| bad_data("field is not UTF-8".into()))
    }

    pub fn as_packed_uints(&self) -> io::Result<Vec<u64>> {
        let mut bytes = self.as_bytes()?;
        let mut values = Vec::new();
        while !bytes.is_empty() {
            values.push(read_varint(&mut bytes)?);
        }
        Ok(values)
    }
}

impl<'a> FieldReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// The next `(tag, value)`, or `None` at end of message. Decoding — not
    /// interpreting — every wire-type is what makes unknown-tag skip safe.
    #[allow(clippy::should_implement_trait)] // io::Result payload; not Iterator
    pub fn next(&mut self) -> io::Result<Option<(u32, FieldValue<'a>)>> {
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let mut rest = &self.bytes[self.pos..];
        let before = rest.len();
        let key = read_varint(&mut rest)?;
        let tag = u32::try_from(key >> 3).map_err(|_| bad_data("tag overflows u32".into()))?;
        let wire = WireType::from_bits(key & 0x7)?;
        let value = match wire {
            WireType::Varint => FieldValue::Varint(read_varint(&mut rest)?),
            WireType::Fixed64 => {
                if rest.len() < 8 {
                    return Err(bad_data("truncated fixed64".into()));
                }
                let (head, tail) = rest.split_at(8);
                let value = u64::from_le_bytes(head.try_into().expect("8 bytes"));
                rest = tail;
                FieldValue::Fixed64(value)
            }
            WireType::LengthDelimited => {
                let len = usize::try_from(read_varint(&mut rest)?)
                    .map_err(|_| bad_data("length overflows usize".into()))?;
                if rest.len() < len {
                    return Err(bad_data("truncated length-delimited field".into()));
                }
                let (head, tail) = rest.split_at(len);
                rest = tail;
                FieldValue::Bytes(head)
            }
            WireType::Fixed32 => {
                if rest.len() < 4 {
                    return Err(bad_data("truncated fixed32".into()));
                }
                let (head, tail) = rest.split_at(4);
                let value = u32::from_le_bytes(head.try_into().expect("4 bytes"));
                rest = tail;
                FieldValue::Fixed32(value)
            }
        };
        self.pos += before - rest.len();
        Ok(Some((tag, value)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_round_trips_boundaries() {
        for value in [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX] {
            let mut buf = Vec::new();
            write_varint(&mut buf, value).unwrap();
            assert_eq!(read_varint(&mut buf.as_slice()).unwrap(), value);
        }
    }

    #[test]
    fn zigzag_round_trips_extremes() {
        for value in [0i64, -1, 1, i64::MIN, i64::MAX, -123_456, 123_456] {
            assert_eq!(unzigzag(zigzag(value)), value);
        }
        // Small negatives stay small — the whole point of zigzag.
        assert_eq!(zigzag(-1), 1);
        assert_eq!(zigzag(1), 2);
    }

    #[test]
    fn fields_round_trip_every_wire_type() {
        let mut writer = FieldWriter::new();
        writer
            .uint(1, 42)
            .sint(2, -7)
            .boolean(3, true)
            .fixed64(4, 0xDEAD_BEEF_CAFE_F00D)
            .string(5, "archon")
            .packed_uints(6, [1u64, 128, 300]);
        let bytes = writer.into_bytes();

        let mut reader = FieldReader::new(&bytes);
        let (tag, value) = reader.next().unwrap().unwrap();
        assert_eq!((tag, value.as_uint().unwrap()), (1, 42));
        let (tag, value) = reader.next().unwrap().unwrap();
        assert_eq!((tag, value.as_sint().unwrap()), (2, -7));
        let (tag, value) = reader.next().unwrap().unwrap();
        assert_eq!((tag, value.as_bool().unwrap()), (3, true));
        let (tag, value) = reader.next().unwrap().unwrap();
        assert_eq!(
            (tag, value.as_fixed64().unwrap()),
            (4, 0xDEAD_BEEF_CAFE_F00D)
        );
        let (tag, value) = reader.next().unwrap().unwrap();
        assert_eq!((tag, value.as_string().unwrap()), (5, "archon".to_string()));
        let (tag, value) = reader.next().unwrap().unwrap();
        assert_eq!(value.as_packed_uints().unwrap(), vec![1, 128, 300]);
        assert_eq!(tag, 6);
        assert!(reader.next().unwrap().is_none());
    }

    #[test]
    fn unknown_tags_are_skippable_by_construction() {
        // A reader that recognizes only tag 1 must cleanly step over tags it
        // has never heard of, whatever their wire type — the §3.5.2 contract.
        let mut writer = FieldWriter::new();
        writer
            .uint(99, 7) // future varint field
            .fixed64(100, 123) // future fixed64 field
            .string(101, "future") // future length-delimited field
            .uint(1, 42); // the one known field, last
        let bytes = writer.into_bytes();

        let mut reader = FieldReader::new(&bytes);
        let mut known = None;
        while let Some((tag, value)) = reader.next().unwrap() {
            if tag == 1 {
                known = Some(value.as_uint().unwrap());
            } // any other tag: decoded and dropped — the skip
        }
        assert_eq!(known, Some(42));
    }

    #[test]
    fn truncated_payloads_error_cleanly() {
        let mut writer = FieldWriter::new();
        writer.string(1, "archon");
        let bytes = writer.into_bytes();
        // Chop mid-payload: reader must error, not panic or hang.
        let mut reader = FieldReader::new(&bytes[..bytes.len() - 2]);
        assert!(reader.next().is_err());
    }
}
