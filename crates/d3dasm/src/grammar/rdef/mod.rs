//! `.d3dasm` text codecs for the RDEF (resource definitions) chunk.
//!
//! Two forms, both lossless:
//!
//! * [`hlsl`] — the preferred editable HLSL reconstruction.
//! * [`rdef_to_text`] / [`rdef_from_text`] — a flat `key=value` fallback for
//!   RDEF shapes the HLSL form can't express.
//!
//! The byte parser ([`dxbc::chunks::rdef::parse_rdef`]) and the IR types live in
//! `dxbc`.

pub mod hlsl;

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

use dxbc::chunks::rdef::{
    CBufferDef, CBufferVariable, MemberDesc, ResourceBinding, ResourceDef, TypeDesc,
};

/// Serialize an RDEF to a flat, editable `key=value` text form. Returns `None`
/// for shapes the codec does not model (e.g. nested struct members).
pub fn rdef_to_text(rd: &ResourceDef<'_>) -> Option<String> {
    use core::fmt::Write as _;

    fn write_type(o: &mut String, t: &TypeDesc<'_>) {
        let _ = write!(
            o,
            " class={} base={} rows={} cols={} elements={}",
            t.class, t.var_type, t.rows, t.columns, t.elements
        );
        if !t.name.is_empty() {
            let _ = write!(o, " typename={}", t.name);
        }
        if let Some(e) = &t.sm5_extra {
            let _ = write!(o, " sm5={:x},{:x},{:x},{:x}", e[0], e[1], e[2], e[3]);
        }
    }

    // Nested struct members (a member whose own type has members) are not yet
    // supported by the text codec; defer those RDEFs to raw hex.
    for cb in &rd.constant_buffers {
        for v in &cb.variables {
            for m in &v.var_type.members {
                if !m.member_type.members.is_empty() {
                    return None;
                }
            }
        }
    }

    let mut o = String::new();
    let _ = writeln!(o, "version=0x{:08x}", rd.target_version);
    let _ = writeln!(o, "flags=0x{:x}", rd.compile_flags);
    let _ = writeln!(o, "creator={}", rd.creator);
    if let Some(rd11) = &rd.rd11_extra {
        let _ = write!(o, "rd11=");
        for (i, x) in rd11.iter().enumerate() {
            let _ = write!(o, "{}{x:x}", if i == 0 { "" } else { "," });
        }
        let _ = writeln!(o);
    }
    for b in &rd.bindings {
        let _ = writeln!(
            o,
            "binding {} input={} return={} dim={} samples={} slot={} count={} flags={:x}",
            b.name,
            b.input_type,
            b.return_type,
            b.dimension,
            b.num_samples,
            b.bind_point,
            b.bind_count,
            b.flags
        );
    }
    for cb in &rd.constant_buffers {
        let _ = writeln!(
            o,
            "cbuffer {} size={} flags={:x} kind={}",
            cb.name, cb.size, cb.flags, cb.cb_type
        );
        for v in &cb.variables {
            let _ = write!(
                o,
                "  var {} offset={} size={} flags={:x}",
                v.name, v.offset, v.size, v.flags
            );
            write_type(&mut o, &v.var_type);
            if let Some(ts) = v.texture_start {
                let _ = write!(o, " tex={},{}", ts as i32, v.texture_size.unwrap_or(0));
            }
            if let Some(ss) = v.sampler_start {
                let _ = write!(o, " samp={},{}", ss as i32, v.sampler_size.unwrap_or(0));
            }
            if !v.default_value.is_empty() {
                let _ = write!(o, " default=");
                for byte in v.default_value.iter() {
                    let _ = write!(o, "{byte:02x}");
                }
            }
            let _ = writeln!(o);
            for m in &v.var_type.members {
                let _ = write!(o, "    member {} offset={}", m.name, m.offset);
                write_type(&mut o, &m.member_type);
                let _ = writeln!(o);
            }
        }
    }
    Some(o)
}

/// Parse the text produced by [`rdef_to_text`] back into an owned RDEF.
pub fn rdef_from_text(text: &str) -> Option<ResourceDef<'static>> {
    use alloc::collections::BTreeMap;

    fn kv<'a>(it: impl Iterator<Item = &'a str>) -> BTreeMap<&'a str, &'a str> {
        let mut m = BTreeMap::new();
        for tok in it {
            if let Some((k, val)) = tok.split_once('=') {
                m.insert(k, val);
            }
        }
        m
    }
    fn dec(m: &BTreeMap<&str, &str>, k: &str) -> Option<u32> {
        m.get(k)?.parse().ok()
    }
    fn hexv(m: &BTreeMap<&str, &str>, k: &str) -> Option<u32> {
        u32::from_str_radix(m.get(k)?, 16).ok()
    }
    fn hex_bytes(h: &str) -> Option<Vec<u8>> {
        if h.len() % 2 != 0 {
            return None;
        }
        let b = h.as_bytes();
        let mut out = Vec::with_capacity(h.len() / 2);
        let mut i = 0;
        while i < b.len() {
            let hi = (b[i] as char).to_digit(16)?;
            let lo = (b[i + 1] as char).to_digit(16)?;
            out.push((hi * 16 + lo) as u8);
            i += 2;
        }
        Some(out)
    }
    fn parse_type(m: &BTreeMap<&str, &str>) -> Option<TypeDesc<'static>> {
        let sm5_extra = match m.get("sm5") {
            Some(s) => {
                let mut it = s.split(',');
                let mut a = [0u32; 4];
                for slot in &mut a {
                    *slot = u32::from_str_radix(it.next()?, 16).ok()?;
                }
                Some(a)
            }
            None => None,
        };
        Some(TypeDesc {
            class: dec(m, "class")? as u16,
            var_type: dec(m, "base")? as u16,
            rows: dec(m, "rows")? as u16,
            columns: dec(m, "cols")? as u16,
            elements: dec(m, "elements")? as u16,
            members: Vec::new(),
            sm5_extra,
            name: match m.get("typename") {
                Some(s) => Cow::Owned(String::from(*s)),
                None => Cow::Borrowed(""),
            },
        })
    }

    let mut rd = ResourceDef {
        constant_buffers: Vec::new(),
        bindings: Vec::new(),
        creator: Cow::Owned(String::new()),
        target_version: 0,
        compile_flags: 0,
        rd11_extra: None,
    };

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Header tags (`key=value`).
        let hex0x = |s: &str| u32::from_str_radix(s.trim().strip_prefix("0x").unwrap_or(s.trim()), 16);
        if let Some(v) = line.strip_prefix("version=") {
            rd.target_version = hex0x(v).ok()?;
            continue;
        }
        if let Some(v) = line.strip_prefix("flags=") {
            rd.compile_flags = hex0x(v).ok()?;
            continue;
        }
        if let Some(v) = line.strip_prefix("creator=") {
            rd.creator = Cow::Owned(String::from(v));
            continue;
        }
        if let Some(v) = line.strip_prefix("rd11=") {
            let mut a = [0u32; 8];
            let mut it = v.split(',');
            for slot in &mut a {
                *slot = u32::from_str_radix(it.next()?.trim(), 16).ok()?;
            }
            rd.rd11_extra = Some(a);
            continue;
        }

        let (head, rest) = line.split_once(' ').unwrap_or((line, ""));
        match head {
            "binding" => {
                let mut f = rest.split_whitespace();
                let name = f.next()?;
                let m = kv(f);
                rd.bindings.push(ResourceBinding {
                    name: Cow::Owned(String::from(name)),
                    input_type: dec(&m, "input")?,
                    return_type: dec(&m, "return")?,
                    dimension: dec(&m, "dim")?,
                    num_samples: dec(&m, "samples")?,
                    bind_point: dec(&m, "slot")?,
                    bind_count: dec(&m, "count")?,
                    flags: hexv(&m, "flags")?,
                });
            }
            "cbuffer" => {
                let mut f = rest.split_whitespace();
                let name = f.next()?;
                let m = kv(f);
                rd.constant_buffers.push(CBufferDef {
                    name: Cow::Owned(String::from(name)),
                    variables: Vec::new(),
                    size: dec(&m, "size")?,
                    flags: hexv(&m, "flags")?,
                    cb_type: dec(&m, "kind")?,
                });
            }
            "member" => {
                let mut f = rest.split_whitespace();
                let name = f.next()?;
                let m = kv(f);
                let member_type = parse_type(&m)?;
                let offset = dec(&m, "offset")?;
                let var = rd.constant_buffers.last_mut()?.variables.last_mut()?;
                var.var_type.members.push(MemberDesc {
                    name: Cow::Owned(String::from(name)),
                    member_type,
                    offset,
                });
            }
            "var" => {
                let mut f = rest.split_whitespace();
                let name = f.next()?;
                let m = kv(f);
                let pair = |s: &str| -> Option<(u32, u32)> {
                    let (a, b) = s.split_once(',')?;
                    Some((a.parse::<i32>().ok()? as u32, b.parse().ok()?))
                };
                let (ts, tz) = match m.get("tex") {
                    Some(s) => {
                        let (a, b) = pair(s)?;
                        (Some(a), Some(b))
                    }
                    None => (None, None),
                };
                let (ss, sz) = match m.get("samp") {
                    Some(s) => {
                        let (a, b) = pair(s)?;
                        (Some(a), Some(b))
                    }
                    None => (None, None),
                };
                let default = match m.get("default") {
                    Some(h) => hex_bytes(h)?,
                    None => Vec::new(),
                };
                let var_type = parse_type(&m)?;
                let cb = rd.constant_buffers.last_mut()?;
                cb.variables.push(CBufferVariable {
                    name: Cow::Owned(String::from(name)),
                    offset: dec(&m, "offset")?,
                    size: dec(&m, "size")?,
                    flags: hexv(&m, "flags")?,
                    var_type,
                    default_value: Cow::Owned(default),
                    texture_start: ts,
                    texture_size: tz,
                    sampler_start: ss,
                    sampler_size: sz,
                });
            }
            _ => return None,
        }
    }
    Some(rd)
}
