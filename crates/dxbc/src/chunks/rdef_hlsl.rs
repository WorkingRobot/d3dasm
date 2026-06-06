//! Editable HLSL reconstruction of an RDEF chunk.
//!
//! Renders resource bindings as HLSL declarations and constant buffers / struct
//! types as HLSL blocks with `packoffset`, so the resource interface reads and
//! edits like source. Anything HLSL can't express (the `used` flag, default
//! values, SM5 extra dwords, structured-buffer strides, binding flags) rides
//! along as small inline annotations, so the form stays byte-exact.
//!
//! The container layer prefers this form but verifies it round-trips; if a
//! particular RDEF can't be reproduced exactly here it falls back to the
//! explicit `key=value` form in [`super::rdef`].

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use super::rdef::{
    BIND_FLAG_COMPARISON_SAMPLER, BIND_FLAG_TEX_COMP_0, BIND_FLAG_TEX_COMP_1, BIND_FLAG_USED,
    BIND_FLAG_USER_PACKED, CBufferDef, CBufferVariable, MemberDesc, ResourceBinding, ResourceDef,
    SLOT_UNUSED, TypeDesc, hlsl_type_name, parse_hlsl_type,
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn is_sm5(target_version: u32) -> bool {
    ((target_version >> 8) & 0xFF) >= 5
}

fn base_size(var_type: u16) -> u32 {
    match var_type {
        39 => 8,  // double
        _ => 4,   // bool/int/uint/float
    }
}

/// Natural byte size of a non-array scalar/vector/matrix type.
fn natural_size(td: &TypeDesc<'_>) -> Option<u32> {
    if td.elements != 0 || !td.members.is_empty() {
        return None;
    }
    let b = base_size(td.var_type);
    Some(match td.class {
        0 => b,
        1 => b * td.columns as u32,
        2 | 3 => b * td.rows as u32 * td.columns as u32,
        _ => return None,
    })
}

/// Byte offset → `packoffset(cR[.comp])` (None if not 4-byte aligned).
fn packoffset_str(offset: u32) -> Option<String> {
    if offset % 4 != 0 {
        return None;
    }
    let reg = offset / 16;
    let comp = (offset % 16) / 4;
    Some(if comp == 0 {
        alloc::format!("c{reg}")
    } else {
        alloc::format!("c{reg}.{}", ['x', 'y', 'z', 'w'][comp as usize])
    })
}

fn packoffset_parse(s: &str) -> Option<u32> {
    let s = s.strip_prefix('c')?;
    let (reg, comp) = match s.split_once('.') {
        Some((r, c)) => (
            r.parse::<u32>().ok()?,
            match c {
                "x" => 0,
                "y" => 1,
                "z" => 2,
                "w" => 3,
                _ => return None,
            },
        ),
        None => (s.parse::<u32>().ok()?, 0),
    };
    Some(reg * 16 + comp * 4)
}

/// Binding flags → `userPacked|comparisonSampler|...` (or hex for unknown bits).
fn bind_flags_str(f: u32) -> String {
    let mut parts: Vec<&str> = Vec::new();
    let mut known = 0u32;
    for (bit, name) in [
        (BIND_FLAG_USER_PACKED, "userPacked"),
        (BIND_FLAG_USED, "used"),
        (BIND_FLAG_COMPARISON_SAMPLER, "comparisonSampler"),
        (BIND_FLAG_TEX_COMP_0, "texComp0"),
        (BIND_FLAG_TEX_COMP_1, "texComp1"),
    ] {
        if f & bit != 0 {
            parts.push(name);
            known |= bit;
        }
    }
    let extra = f & !known;
    let mut s = parts.join("|");
    if extra != 0 {
        if !s.is_empty() {
            s.push('|');
        }
        let _ = write!(s, "{extra:#x}");
    }
    s
}

fn bind_flags_from(s: &str) -> Option<u32> {
    let mut f = 0u32;
    for tok in s.split('|') {
        f |= match tok {
            "userPacked" => BIND_FLAG_USER_PACKED,
            "used" => BIND_FLAG_USED,
            "comparisonSampler" => BIND_FLAG_COMPARISON_SAMPLER,
            "texComp0" => BIND_FLAG_TEX_COMP_0,
            "texComp1" => BIND_FLAG_TEX_COMP_1,
            other => {
                let other = other.strip_prefix("0x").unwrap_or(other);
                u32::from_str_radix(other, 16).ok()?
            }
        };
    }
    Some(f)
}

/// Texture/UAV dimension → HLSL type stem.
fn dim_to_hlsl(d: u32) -> Option<&'static str> {
    Some(match d {
        1 => "Buffer",
        2 => "Texture1D",
        3 => "Texture2D",
        4 => "Texture2DMS",
        5 => "Texture3D",
        6 => "TextureCube",
        7 => "Texture1DArray",
        8 => "Texture2DArray",
        9 => "Texture2DMSArray",
        10 => "TextureCubeArray",
        _ => return None,
    })
}

fn dim_from_hlsl(s: &str) -> Option<u32> {
    Some(match s {
        "Buffer" => 1,
        "Texture1D" => 2,
        "Texture2D" => 3,
        "Texture2DMS" => 4,
        "Texture3D" => 5,
        "TextureCube" => 6,
        "Texture1DArray" => 7,
        "Texture2DArray" => 8,
        "Texture2DMSArray" => 9,
        "TextureCubeArray" => 10,
        _ => return None,
    })
}

/// Resource return type → HLSL element spelling (None when there is none).
fn ret_to_hlsl(r: u32) -> Option<&'static str> {
    Some(match r {
        1 => "unorm4",
        2 => "snorm4",
        3 => "int4",
        4 => "uint4",
        5 => "float4",
        6 => "mixed",
        7 => "double4",
        _ => return None,
    })
}

fn ret_from_hlsl(s: &str) -> Option<u32> {
    Some(match s {
        "unorm4" => 1,
        "snorm4" => 2,
        "int4" => 3,
        "uint4" => 4,
        "float4" => 5,
        "mixed" => 6,
        "double4" => 7,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Type references
// ---------------------------------------------------------------------------

/// HLSL spelling of a variable/member type reference (e.g. `float4`,
/// `float4x4`, `float4[8]`, a struct name, or `Name[3]`). The `row_major`
/// qualifier is emitted separately by the caller.
fn type_ref(td: &TypeDesc<'_>) -> Option<String> {
    if !td.members.is_empty() || td.class == 5 {
        // Struct reference: name with optional array suffix.
        if td.name.is_empty() {
            return None;
        }
        return Some(if td.elements > 0 {
            alloc::format!("{}[{}]", td.name, td.elements)
        } else {
            String::from(td.name.as_ref())
        });
    }
    hlsl_type_name(td)
}

// ---------------------------------------------------------------------------
// Emit
// ---------------------------------------------------------------------------

/// Look up the element struct name for a structured-buffer binding (cosmetic).
fn struct_elem_name<'a>(b: &ResourceBinding<'_>, rd: &'a ResourceDef<'_>) -> Option<&'a str> {
    let cb = rd.constant_buffers.iter().find(|c| c.name == b.name)?;
    let v = cb.variables.first()?;
    if v.var_type.members.is_empty() {
        None
    } else {
        Some(v.var_type.name.as_ref())
    }
}

/// `(typespec, register-class char)` for a binding; None if unmodelled.
fn binding_typespec(b: &ResourceBinding<'_>, rd: &ResourceDef<'_>) -> Option<(String, char)> {
    let elem = || struct_elem_name(b, rd).unwrap_or("");
    let wrap = |kw: &str| {
        let e = elem();
        if e.is_empty() {
            String::from(kw)
        } else {
            alloc::format!("{kw}<{e}>")
        }
    };
    let texture = |prefix: &str| -> Option<String> {
        let dim = dim_to_hlsl(b.dimension)?;
        Some(match ret_to_hlsl(b.return_type) {
            Some(r) => alloc::format!("{prefix}{dim}<{r}>"),
            None => alloc::format!("{prefix}{dim}"),
        })
    };
    Some(match b.input_type {
        0 => (String::from("cbuffer"), 'b'),
        1 => (String::from("tbuffer"), 't'),
        2 => (texture("")?, 't'),
        3 => (String::from("SamplerState"), 's'),
        4 => (texture("RW")?, 'u'),
        5 => (wrap("StructuredBuffer"), 't'),
        6 => (wrap("RWStructuredBuffer"), 'u'),
        7 => (String::from("ByteAddressBuffer"), 't'),
        8 => (String::from("RWByteAddressBuffer"), 'u'),
        9 => (wrap("AppendStructuredBuffer"), 'u'),
        10 => (wrap("ConsumeStructuredBuffer"), 'u'),
        11 => (wrap("RWStructuredBuffer"), 'u'),
        _ => return None,
    })
}

fn is_texture(input_type: u32) -> bool {
    input_type == 2 || input_type == 4
}

fn emit_binding(o: &mut String, b: &ResourceBinding<'_>, rd: &ResourceDef<'_>) -> Option<()> {
    let (spec, reg) = binding_typespec(b, rd)?;
    let _ = write!(o, "{spec} {} : register({reg}{})", b.name, b.bind_point);
    if b.bind_count != 1 {
        let _ = write!(o, "[{}]", b.bind_count);
    }
    // Annotations for fields the typespec does not capture.
    if !is_texture(b.input_type) && b.dimension != 0 {
        let _ = write!(o, " dim={}", b.dimension);
    }
    if !is_texture(b.input_type) && b.return_type != 0 {
        let _ = write!(o, " ret={}", b.return_type);
    }
    if b.num_samples == 0xFFFF_FFFF {
        o.push_str(" samples=-1");
    } else if b.num_samples != 0 {
        let _ = write!(o, " samples={}", b.num_samples);
    }
    if b.flags != 0 {
        let _ = write!(o, " flags={}", bind_flags_str(b.flags));
    }
    o.push_str(";\n");
    Some(())
}

fn emit_var(o: &mut String, v: &CBufferVariable<'_>, indent: &str, sm5: bool) -> Option<()> {
    let t = &v.var_type;
    let tref = type_ref(t)?;
    o.push_str(indent);
    if t.class == 2 {
        o.push_str("row_major ");
    }
    let _ = write!(o, "{tref} {}", v.name);
    let po = packoffset_str(v.offset)?;
    let _ = write!(o, " : packoffset({po})");
    // size: omit when it matches the natural size of a non-array type.
    if natural_size(t) != Some(v.size) {
        let _ = write!(o, " size={}", v.size);
    }
    if v.flags == BIND_FLAG_USED {
        o.push_str(" used");
    } else if v.flags != 0 {
        let _ = write!(o, " vflags={:x}", v.flags);
    }
    if let Some(e) = &t.sm5_extra
        && *e != [0; 4]
    {
        let _ = write!(o, " sm5={:x},{:x},{:x},{:x}", e[0], e[1], e[2], e[3]);
    }
    if let Some(ts) = v.texture_start
        && (ts != SLOT_UNUSED || v.texture_size.unwrap_or(0) != 0)
    {
        let _ = write!(o, " tex={},{}", ts as i32, v.texture_size.unwrap_or(0));
    }
    if let Some(ss) = v.sampler_start
        && (ss != SLOT_UNUSED || v.sampler_size.unwrap_or(0) != 0)
    {
        let _ = write!(o, " samp={},{}", ss as i32, v.sampler_size.unwrap_or(0));
    }
    if !v.default_value.is_empty() {
        o.push_str(" default=");
        for byte in v.default_value.iter() {
            let _ = write!(o, "{byte:02x}");
        }
    }
    // Track whether this is an SM5 var (tex/samp Options present) for parse.
    let _ = sm5;
    o.push_str(";\n");
    Some(())
}

/// Serialize an RDEF to editable HLSL. Returns `None` for anything this codec
/// cannot model losslessly (caller falls back to the `key=value` form).
pub fn rdef_to_hlsl(rd: &ResourceDef<'_>) -> Option<String> {
    let sm5 = is_sm5(rd.target_version);
    let mut o = String::new();
    let _ = writeln!(o, "target {:08x}", rd.target_version);
    let _ = writeln!(o, "flags {:x}", rd.compile_flags);
    let _ = writeln!(o, "creator {}", rd.creator);
    if let Some(rd11) = &rd.rd11_extra {
        o.push_str("rd11");
        for x in rd11 {
            let _ = write!(o, " {x:x}");
        }
        o.push('\n');
    }
    o.push('\n');

    // Struct definitions: each distinct named struct type, once.
    let mut seen_structs: Vec<&str> = Vec::new();
    for cb in &rd.constant_buffers {
        for v in &cb.variables {
            let t = &v.var_type;
            if t.members.is_empty() {
                continue;
            }
            let name = t.name.as_ref();
            if name.is_empty() || seen_structs.contains(&name) {
                continue;
            }
            seen_structs.push(name);
            let _ = write!(o, "struct {name}");
            // A struct type descriptor carries non-obvious scalar fields
            // (fxc stores rows=1, cols=size/4). Preserve any that aren't the
            // struct defaults (class 5, the rest 0).
            if t.class != 5 {
                let _ = write!(o, " class={}", t.class);
            }
            if t.var_type != 0 {
                let _ = write!(o, " vtype={}", t.var_type);
            }
            if t.rows != 0 {
                let _ = write!(o, " rows={}", t.rows);
            }
            if t.columns != 0 {
                let _ = write!(o, " cols={}", t.columns);
            }
            if let Some(e) = &t.sm5_extra
                && *e != [0; 4]
            {
                let _ = write!(o, " sm5={:x},{:x},{:x},{:x}", e[0], e[1], e[2], e[3]);
            }
            o.push_str(" {\n");
            for m in &t.members {
                let mref = type_ref(&m.member_type)?;
                o.push_str("    ");
                if m.member_type.class == 2 {
                    o.push_str("row_major ");
                }
                let _ = write!(o, "{mref} {} +{}", m.name, m.offset);
                if let Some(e) = &m.member_type.sm5_extra
                    && *e != [0; 4]
                {
                    let _ = write!(o, " sm5={:x},{:x},{:x},{:x}", e[0], e[1], e[2], e[3]);
                }
                o.push_str(";\n");
            }
            o.push_str("}\n");
        }
    }
    if !seen_structs.is_empty() {
        o.push('\n');
    }

    // Resource bindings, in binding-array order.
    for b in &rd.bindings {
        emit_binding(&mut o, b, rd)?;
    }
    if !rd.bindings.is_empty() {
        o.push('\n');
    }

    // Constant-buffer / resource layouts, in cbuffer-array order.
    for cb in &rd.constant_buffers {
        let _ = write!(o, "cbuffer {}", cb.name);
        if cb.cb_type != 0 {
            let _ = write!(o, " kind={}", cb.cb_type);
        }
        if cb.flags != 0 {
            let _ = write!(o, " cbflags={:x}", cb.flags);
        }
        o.push_str(" {\n");
        for v in &cb.variables {
            emit_var(&mut o, v, "    ", sm5)?;
        }
        o.push_str("}\n");
    }
    Some(o)
}

// ---------------------------------------------------------------------------
// Parse
// ---------------------------------------------------------------------------

/// Collect `key=value` tokens (ignoring bare words) from an iterator.
fn kv<'a>(it: impl Iterator<Item = &'a str>) -> BTreeMap<&'a str, &'a str> {
    let mut m = BTreeMap::new();
    for tok in it {
        if let Some((k, v)) = tok.split_once('=') {
            m.insert(k, v);
        }
    }
    m
}

fn parse_sm5(s: &str) -> Option<[u32; 4]> {
    let mut it = s.split(',');
    let mut a = [0u32; 4];
    for slot in &mut a {
        *slot = u32::from_str_radix(it.next()?, 16).ok()?;
    }
    Some(a)
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

/// Resolve a type reference token (possibly `row_major`-qualified, with `[N]`)
/// against the known struct table. Returns the reconstructed `TypeDesc`.
fn resolve_type(
    tref: &str,
    row_major: bool,
    structs: &BTreeMap<String, TypeDesc<'static>>,
    sm5: bool,
) -> Option<TypeDesc<'static>> {
    // Array suffix.
    let (core, elements) = match tref.split_once('[') {
        Some((c, rest)) => (c, rest.strip_suffix(']')?.parse().ok()?),
        None => (tref, 0u16),
    };
    if let Some(st) = structs.get(core) {
        let mut td = st.clone();
        td.elements = elements;
        return Some(td);
    }
    let (class, base, rows, cols, _e) = parse_hlsl_type(tref)?;
    let class = if row_major && (class == 3) { 2 } else { class };
    // Canonical element name (no array suffix).
    let core_td = TypeDesc {
        class,
        var_type: base,
        rows,
        columns: cols,
        elements: 0,
        members: Vec::new(),
        sm5_extra: if sm5 { Some([0; 4]) } else { None },
        name: Cow::Borrowed(""),
    };
    let name = hlsl_type_name(&core_td).unwrap_or_default();
    Some(TypeDesc {
        class,
        var_type: base,
        rows,
        columns: cols,
        elements,
        members: Vec::new(),
        sm5_extra: if sm5 { Some([0; 4]) } else { None },
        name: Cow::Owned(name),
    })
}

fn parse_binding(line: &str) -> Option<ResourceBinding<'static>> {
    // `<typespec> <name> : register(<x><slot>)[count] [annotations];`
    let line = line.strip_suffix(';').unwrap_or(line);
    let mut it = line.split_whitespace();
    let spec = it.next()?;
    let name = it.next()?;
    // After the name, find `register(...)`.
    let rest: Vec<&str> = it.collect();
    let reg_tok = rest.iter().find(|t| t.starts_with("register("))?;
    let inner = reg_tok
        .strip_prefix("register(")?
        .split(')')
        .next()?;
    let regclass = inner.chars().next()?;
    // slot then optional `[count]` (count may be appended to the register token).
    let after = &reg_tok[reg_tok.find(')')? + 1..];
    let mut bind_count = 1u32;
    if let Some(c) = after.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        bind_count = c.parse().ok()?;
    }
    let slot: u32 = inner[1..].parse().ok()?;
    let m = kv(rest.iter().copied());

    // Decode the typespec into input_type / dimension / return_type.
    let (input_type, mut dimension, mut return_type) = decode_typespec(spec, regclass)?;
    if let Some(d) = m.get("dim") {
        dimension = d.parse().ok()?;
    }
    if let Some(r) = m.get("ret") {
        return_type = r.parse().ok()?;
    }
    let num_samples = match m.get("samples") {
        Some(s) => s.parse::<i64>().ok()? as u32, // accepts -1 sentinel
        None => 0,
    };
    let flags = match m.get("flags") {
        Some(s) => bind_flags_from(s)?,
        None => 0,
    };
    Some(ResourceBinding {
        name: Cow::Owned(String::from(name)),
        input_type,
        return_type,
        dimension,
        num_samples,
        bind_point: slot,
        bind_count,
        flags,
    })
}

/// Map an HLSL typespec + register class to `(input_type, dimension, return_type)`.
fn decode_typespec(spec: &str, regclass: char) -> Option<(u32, u32, u32)> {
    // Strip any `<...>` element/return wrapper.
    let (stem, inner) = match spec.split_once('<') {
        Some((s, rest)) => (s, Some(rest.strip_suffix('>')?)),
        None => (spec, None),
    };
    let ret = |inner: Option<&str>| inner.and_then(ret_from_hlsl).unwrap_or(0);
    Some(match stem {
        "cbuffer" => (0, 0, 0),
        "tbuffer" => (1, 0, 0),
        "SamplerState" => (3, 0, 0),
        "StructuredBuffer" => (5, 0, 0),
        "RWStructuredBuffer" => (if regclass == 'u' { 6 } else { 5 }, 0, 0),
        "ByteAddressBuffer" => (7, 0, 0),
        "RWByteAddressBuffer" => (8, 0, 0),
        "AppendStructuredBuffer" => (9, 0, 0),
        "ConsumeStructuredBuffer" => (10, 0, 0),
        other => {
            // Texture / RWTexture forms.
            let (prefix_rw, dim_name) = if let Some(d) = other.strip_prefix("RW") {
                (true, d)
            } else {
                (false, other)
            };
            let dim = dim_from_hlsl(dim_name)?;
            let input = if prefix_rw { 4 } else { 2 };
            (input, dim, ret(inner))
        }
    })
}

/// Parse a `key=value`-and-flags variable/member tail into the optional fields.
struct VarTail {
    size: Option<u32>,
    flags: u32,
    sm5: Option<[u32; 4]>,
    tex: Option<(u32, u32)>,
    samp: Option<(u32, u32)>,
    default: Vec<u8>,
}

fn parse_var_tail<'a>(toks: impl Iterator<Item = &'a str> + Clone) -> Option<VarTail> {
    let m = kv(toks.clone());
    let mut flags = 0u32;
    for t in toks {
        if t == "used" {
            flags = BIND_FLAG_USED;
        }
    }
    if let Some(v) = m.get("vflags") {
        flags = u32::from_str_radix(v, 16).ok()?;
    }
    let pair = |s: &str| -> Option<(u32, u32)> {
        let (a, b) = s.split_once(',')?;
        Some((a.parse::<i32>().ok()? as u32, b.parse().ok()?))
    };
    Some(VarTail {
        size: match m.get("size") {
            Some(s) => Some(s.parse().ok()?),
            None => None,
        },
        flags,
        sm5: match m.get("sm5") {
            Some(s) => Some(parse_sm5(s)?),
            None => None,
        },
        tex: match m.get("tex") {
            Some(s) => Some(pair(s)?),
            None => None,
        },
        samp: match m.get("samp") {
            Some(s) => Some(pair(s)?),
            None => None,
        },
        default: match m.get("default") {
            Some(h) => hex_bytes(h)?,
            None => Vec::new(),
        },
    })
}

/// Parse HLSL text produced by [`rdef_to_hlsl`] back into an owned RDEF.
pub fn rdef_from_hlsl(text: &str) -> Option<ResourceDef<'static>> {
    let mut rd = ResourceDef {
        constant_buffers: Vec::new(),
        bindings: Vec::new(),
        creator: Cow::Owned(String::new()),
        target_version: 0,
        compile_flags: 0,
        rd11_extra: None,
    };
    let mut structs: BTreeMap<String, TypeDesc<'static>> = BTreeMap::new();

    // State while inside a `struct { ... }` or `cbuffer { ... }` block.
    struct StructHdr {
        name: String,
        class: u16,
        var_type: u16,
        rows: u16,
        cols: u16,
        sm5: Option<[u32; 4]>,
    }
    enum Block {
        Struct(StructHdr, Vec<MemberDesc<'static>>),
        Cbuffer,
    }
    let mut block: Option<Block> = None;
    let mut sm5 = false;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        // End of a block.
        if line == "}" {
            match block.take()? {
                Block::Struct(hdr, members) => {
                    let td = TypeDesc {
                        class: hdr.class,
                        var_type: hdr.var_type,
                        rows: hdr.rows,
                        columns: hdr.cols,
                        elements: 0,
                        members,
                        sm5_extra: hdr.sm5.or(if sm5 { Some([0; 4]) } else { None }),
                        name: Cow::Owned(hdr.name.clone()),
                    };
                    structs.insert(hdr.name, td);
                }
                Block::Cbuffer => {}
            }
            continue;
        }

        // Inside a block: member or variable line (ends with `;`).
        if let Some(b) = &mut block {
            match b {
                Block::Struct(_, members) => {
                    // `<type> <name> +<offset> [sm5=..];`
                    let body = line.strip_suffix(';').unwrap_or(line);
                    let mut toks = body.split_whitespace();
                    let mut t0 = toks.next()?;
                    let row_major = t0 == "row_major";
                    if row_major {
                        t0 = toks.next()?;
                    }
                    let mname = toks.next()?;
                    let off_tok = toks.next()?;
                    let offset: u32 = off_tok.strip_prefix('+')?.parse().ok()?;
                    let m = kv(toks.clone());
                    let mut mt = resolve_type(t0, row_major, &structs, sm5)?;
                    if let Some(s) = m.get("sm5") {
                        mt.sm5_extra = Some(parse_sm5(s)?);
                    }
                    members.push(MemberDesc {
                        name: Cow::Owned(String::from(mname)),
                        member_type: mt,
                        offset,
                    });
                }
                Block::Cbuffer => {
                    let body = line.strip_suffix(';').unwrap_or(line);
                    let mut toks = body.split_whitespace();
                    let mut t0 = toks.next()?;
                    let row_major = t0 == "row_major";
                    if row_major {
                        t0 = toks.next()?;
                    }
                    let vname = toks.next()?;
                    // Expect `: packoffset(...)`.
                    let _colon = toks.next()?; // ":"
                    let po_tok = toks.next()?; // "packoffset(cN...)"
                    let po = po_tok
                        .strip_prefix("packoffset(")
                        .and_then(|s| s.split(')').next())?;
                    let offset = packoffset_parse(po)?;
                    let tail = parse_var_tail(toks)?;
                    let mut vt = resolve_type(t0, row_major, &structs, sm5)?;
                    if let Some(e) = tail.sm5 {
                        vt.sm5_extra = Some(e);
                    }
                    let size = match tail.size {
                        Some(s) => s,
                        None => natural_size(&vt)?,
                    };
                    let cb = rd.constant_buffers.last_mut()?;
                    cb.variables.push(CBufferVariable {
                        name: Cow::Owned(String::from(vname)),
                        offset,
                        size,
                        flags: tail.flags,
                        var_type: vt,
                        default_value: Cow::Owned(tail.default),
                        texture_start: tail.tex.map(|t| t.0).or(if sm5 {
                            Some(SLOT_UNUSED)
                        } else {
                            None
                        }),
                        texture_size: tail.tex.map(|t| t.1).or(if sm5 { Some(0) } else { None }),
                        sampler_start: tail.samp.map(|t| t.0).or(if sm5 {
                            Some(SLOT_UNUSED)
                        } else {
                            None
                        }),
                        sampler_size: tail.samp.map(|t| t.1).or(if sm5 { Some(0) } else { None }),
                    });
                }
            }
            continue;
        }

        // Top-level directives.
        let (head, rest) = line.split_once(' ').unwrap_or((line, ""));
        match head {
            "target" => {
                rd.target_version = u32::from_str_radix(rest.trim(), 16).ok()?;
                sm5 = is_sm5(rd.target_version);
            }
            "flags" => rd.compile_flags = u32::from_str_radix(rest.trim(), 16).ok()?,
            "creator" => rd.creator = Cow::Owned(String::from(rest)),
            "rd11" => {
                let mut a = [0u32; 8];
                let mut it = rest.split_whitespace();
                for slot in &mut a {
                    *slot = u32::from_str_radix(it.next()?, 16).ok()?;
                }
                rd.rd11_extra = Some(a);
            }
            "struct" => {
                // `struct Name [class=..] [vtype=..] [rows=..] [cols=..] [sm5=..] {`
                let inner = rest.trim_end_matches('{').trim();
                let mut it = inner.split_whitespace();
                let name = it.next()?;
                let m = kv(it);
                let getu = |k: &str, d: u16| -> Option<u16> {
                    match m.get(k) {
                        Some(s) => s.parse().ok(),
                        None => Some(d),
                    }
                };
                let hdr = StructHdr {
                    name: String::from(name),
                    class: getu("class", 5)?,
                    var_type: getu("vtype", 0)?,
                    rows: getu("rows", 0)?,
                    cols: getu("cols", 0)?,
                    sm5: match m.get("sm5") {
                        Some(s) => Some(parse_sm5(s)?),
                        None => None,
                    },
                };
                block = Some(Block::Struct(hdr, Vec::new()));
            }
            "cbuffer" => {
                // Could be a binding line (`cbuffer N : register(b0);`) or a
                // block opener (`cbuffer N [kind=..] [cbflags=..] {`).
                if line.ends_with('{') {
                    let inner = rest.trim_end_matches('{').trim();
                    let mut it = inner.split_whitespace();
                    let name = it.next()?;
                    let m = kv(it);
                    rd.constant_buffers.push(CBufferDef {
                        name: Cow::Owned(String::from(name)),
                        variables: Vec::new(),
                        size: 0, // patched below
                        flags: match m.get("cbflags") {
                            Some(s) => u32::from_str_radix(s, 16).ok()?,
                            None => 0,
                        },
                        cb_type: match m.get("kind") {
                            Some(s) => s.parse().ok()?,
                            None => 0,
                        },
                    });
                    block = Some(Block::Cbuffer);
                } else {
                    rd.bindings.push(parse_binding(line)?);
                }
            }
            // Any other head is a resource-binding declaration.
            _ => rd.bindings.push(parse_binding(line)?),
        }
    }

    // Constant-buffer `size` is the byte span of its variables. Regular
    // cbuffers (kind 0) round up to a 16-byte register; resource cbuffers
    // (structured-buffer `$Element`, kind != 0) use the exact struct size.
    for cb in &mut rd.constant_buffers {
        let mut end = 0u32;
        for v in &cb.variables {
            end = end.max(v.offset + v.size);
        }
        cb.size = if cb.cb_type == 0 {
            end.div_ceil(16) * 16
        } else {
            end
        };
    }
    Some(rd)
}
