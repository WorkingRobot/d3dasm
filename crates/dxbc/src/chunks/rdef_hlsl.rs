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
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use super::ChunkWriter;
use super::rdef::{
    BIND_FLAG_COMPARISON_SAMPLER, BIND_FLAG_TEX_COMP_0, BIND_FLAG_TEX_COMP_1, BIND_FLAG_USED,
    BIND_FLAG_USER_PACKED, CBufferDef, CBufferVariable, MemberDesc, ResourceBinding, ResourceDef,
    SLOT_UNUSED, TypeDesc, hlsl_type_name, parse_hlsl_type,
};
use crate::shex::{ComponentSelect, Operand, OperandIndex, Program, RegisterType};

// ---------------------------------------------------------------------------
// Constant-buffer usage analysis (derives the `used` flag from the program)
// ---------------------------------------------------------------------------

/// Sentinel `flags` meaning "no explicit tag — derive the used bit".
const DERIVE_USED: u32 = 0xFFFF_FFFF;

/// Which bytes of one constant buffer the shader program reads.
#[derive(Default)]
struct CbReads {
    /// Statically-addressed bytes.
    bytes: BTreeSet<u32>,
    /// Base byte offsets of dynamically-indexed arrays (`cb[base + r]`).
    dyn_base: BTreeSet<u32>,
    /// A fully dynamic index was seen; treat the whole buffer as read.
    dyn_all: bool,
}

fn collect_cb_reads(op: &Operand, t: &mut BTreeMap<u32, CbReads>) {
    for idx in op.indices.iter() {
        if let OperandIndex::Relative(o) | OperandIndex::RelativePlusImm(_, o) = idx {
            collect_cb_reads(o, t);
        }
    }
    if op.reg_type != RegisterType::ConstantBuffer {
        return;
    }
    let mut it = op.indices.iter();
    let (Some(i0), Some(i1)) = (it.next(), it.next()) else {
        return;
    };
    let OperandIndex::Imm32(reg) = i0 else { return };
    let comps: Vec<u32> = match &op.components {
        ComponentSelect::Swizzle(s) => {
            let mut v: Vec<u32> = s.iter().map(|&c| c as u32).collect();
            v.sort_unstable();
            v.dedup();
            v
        }
        ComponentSelect::Scalar(c) => alloc::vec![*c as u32],
        _ => alloc::vec![0, 1, 2, 3],
    };
    let e = t.entry(*reg).or_default();
    match i1 {
        OperandIndex::Imm32(v) => {
            let v = *v;
            for c in &comps {
                for b in 0..4 {
                    e.bytes.insert(v * 16 + c * 4 + b);
                }
            }
        }
        OperandIndex::RelativePlusImm(base, _) => {
            e.dyn_base.insert(*base * 16);
        }
        _ => e.dyn_all = true,
    }
}

/// Map constant-buffer register → the bytes the program reads from it.
fn cbuffer_reads(program: &Program) -> BTreeMap<u32, CbReads> {
    let mut t = BTreeMap::new();
    for ins in &program.instructions {
        for op in ins.operands() {
            collect_cb_reads(op, &mut t);
        }
    }
    t
}

/// Whether any byte of `[offset, offset+size)` is read.
fn var_used(r: &CbReads, offset: u32, size: u32) -> bool {
    r.dyn_all
        || r.dyn_base.iter().any(|&b| b >= offset && b < offset + size)
        || (offset..offset + size).any(|b| r.bytes.contains(&b))
}

/// Reads for the constant buffer named `cb_name`, if its register is known.
fn cb_reads_for<'a>(
    rd: &ResourceDef<'_>,
    reads: Option<&'a BTreeMap<u32, CbReads>>,
    cb_name: &str,
) -> Option<&'a CbReads> {
    let reg = rd
        .bindings
        .iter()
        .find(|b| b.name == cb_name && b.input_type == 0)?
        .bind_point;
    reads?.get(&reg)
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn is_sm5(target_version: u32) -> bool {
    ((target_version >> 8) & 0xFF) >= 5
}

fn base_size(var_type: u16) -> u32 {
    match var_type {
        39 => 8, // double
        _ => 4,  // bool/int/uint/float
    }
}

/// Byte size of a type: the struct span for struct types, else the natural
/// scalar/vector/matrix size. Used to derive structured-buffer strides.
fn type_size(td: &TypeDesc<'_>) -> u32 {
    if td.members.is_empty() {
        natural_size(td).unwrap_or(0)
    } else {
        td.members
            .iter()
            .filter_map(|m| natural_size(&m.member_type).map(|s| m.offset + s))
            .max()
            .unwrap_or(0)
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

/// Split a `type[N]` spelling into `(stem, Some(N-text))` (array on the type),
/// so the caller can move the suffix onto the variable name (proper HLSL).
fn split_array(tref: &str) -> (&str, Option<&str>) {
    match tref.split_once('[') {
        Some((s, rest)) => (s, rest.strip_suffix(']')),
        None => (tref, None),
    }
}

/// Split a `name[N]` token into `(name, elements)`.
fn split_name(tok: &str) -> Option<(&str, u16)> {
    match tok.split_once('[') {
        Some((n, rest)) => Some((n, rest.strip_suffix(']')?.parse().ok()?)),
        None => Some((tok, 0)),
    }
}

// ---------------------------------------------------------------------------
// Emit
// ---------------------------------------------------------------------------

/// HLSL element-type spelling for a structured-buffer binding (its `$Element`
/// variable's type), used as the `<T>` in `StructuredBuffer<T>`. This lets the
/// merged form reconstruct the resource's constant-buffer def from the binding.
fn elem_type_ref(b: &ResourceBinding<'_>, rd: &ResourceDef<'_>) -> Option<String> {
    let cb = rd.constant_buffers.iter().find(|c| c.name == b.name)?;
    let v = cb.variables.first()?;
    type_ref(&v.var_type)
}

/// `(typespec, register-class char)` for a binding; None if unmodelled.
fn binding_typespec(b: &ResourceBinding<'_>, rd: &ResourceDef<'_>) -> Option<(String, char)> {
    let elem = elem_type_ref(b, rd);
    let wrap = |kw: &str| match &elem {
        Some(e) => alloc::format!("{kw}<{e}>"),
        None => String::from(kw),
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
    // Annotations for fields the typespec does not capture. Structured buffers
    // always carry dimension=Buffer(1) and return=mixed(6), and textures default
    // to the -1 "not multisampled" sentinel — all derived on parse.
    let structured = matches!(b.input_type, 5 | 6 | 9 | 10 | 11);
    let dim_default = if structured { 1 } else { 0 };
    let ret_default = if structured { 6 } else { 0 };
    if !is_texture(b.input_type) && b.dimension != dim_default {
        let _ = write!(o, " dim={}", b.dimension);
    }
    if !is_texture(b.input_type) && b.return_type != ret_default {
        let _ = write!(o, " ret={}", b.return_type);
    }
    if structured {
        // stride == sizeof(element); derived from the def on parse.
    } else if is_texture(b.input_type) {
        if b.num_samples != 0xFFFF_FFFF {
            let _ = write!(o, " samples={}", b.num_samples);
        }
    } else if b.num_samples != 0 {
        let _ = write!(o, " samples={}", b.num_samples);
    }
    if b.flags != 0 {
        let _ = write!(o, " flags={}", bind_flags_str(b.flags));
    }
    o.push_str(";\n");
    Some(())
}

fn emit_var(
    o: &mut String,
    v: &CBufferVariable<'_>,
    indent: &str,
    sm5: bool,
    cb_reads: Option<&CbReads>,
) -> Option<()> {
    let t = &v.var_type;
    let tref = type_ref(t)?;
    let (stem, arr) = split_array(&tref);
    o.push_str(indent);
    if t.class == 2 {
        o.push_str("row_major ");
    }
    let _ = write!(o, "{stem} {}", v.name);
    if let Some(n) = arr {
        let _ = write!(o, "[{n}]");
    }
    let po = packoffset_str(v.offset)?;
    let _ = write!(o, " : packoffset({po})");
    // size: omit when it matches the natural size of a non-array type.
    if natural_size(t) != Some(v.size) {
        let _ = write!(o, " size={}", v.size);
    }
    // The `used` flag is derived from the program when available; tag the
    // variable only when the stored flags differ from that derivation.
    let baseline = match cb_reads {
        Some(r) if var_used(r, v.offset, v.size) => BIND_FLAG_USED,
        Some(_) => 0,
        None => 0,
    };
    if v.flags != baseline {
        if v.flags == BIND_FLAG_USED {
            o.push_str(" used");
        } else if v.flags == 0 {
            o.push_str(" unused");
        } else {
            let _ = write!(o, " vflags={:x}", v.flags);
        }
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
pub fn rdef_to_hlsl(rd: &ResourceDef<'_>, program: Option<&Program>) -> Option<String> {
    let sm5 = is_sm5(rd.target_version);
    let reads = program.map(cbuffer_reads);
    let reads = reads.as_ref();
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
            // A struct type descriptor stores rows=1 and cols=size/4; both are
            // derived on parse, so only deviations need annotating.
            let struct_size: u32 = t
                .members
                .iter()
                .filter_map(|m| natural_size(&m.member_type).map(|s| m.offset + s))
                .max()
                .unwrap_or(0);
            let _ = write!(o, "struct {name}");
            if t.class != 5 {
                let _ = write!(o, " class={}", t.class);
            }
            if t.var_type != 0 {
                let _ = write!(o, " vtype={}", t.var_type);
            }
            if t.rows != 1 {
                let _ = write!(o, " rows={}", t.rows);
            }
            if t.columns as u32 != struct_size / 4 {
                let _ = write!(o, " cols={}", t.columns);
            }
            if let Some(e) = &t.sm5_extra
                && *e != [0; 4]
            {
                let _ = write!(o, " sm5={:x},{:x},{:x},{:x}", e[0], e[1], e[2], e[3]);
            }
            o.push_str(" {\n");
            // Members are tightly packed; emit `+offset` only when one breaks
            // the running natural-size layout.
            let mut running = 0u32;
            for m in &t.members {
                let mref = type_ref(&m.member_type)?;
                let (stem, arr) = split_array(&mref);
                o.push_str("    ");
                if m.member_type.class == 2 {
                    o.push_str("row_major ");
                }
                let _ = write!(o, "{stem} {}", m.name);
                if let Some(n) = arr {
                    let _ = write!(o, "[{n}]");
                }
                if m.offset != running {
                    let _ = write!(o, " +{}", m.offset);
                }
                if let Some(e) = &m.member_type.sm5_extra
                    && *e != [0; 4]
                {
                    let _ = write!(o, " sm5={:x},{:x},{:x},{:x}", e[0], e[1], e[2], e[3]);
                }
                o.push_str(";\n");
                running = m.offset + natural_size(&m.member_type).unwrap_or(0);
            }
            o.push_str("};\n");
        }
    }
    if !seen_structs.is_empty() {
        o.push('\n');
    }

    // Body: prefer merged single-declaration cbuffers; fall back to the
    // two-section form when the array orderings don't allow a faithful merge.
    if let Some(body) = emit_merged_body(rd, sm5, reads) {
        let full = alloc::format!("{o}{body}");
        if let Some(rd2) = rdef_from_hlsl(&full, program)
            && rd2.to_writable().data == rd.to_writable().data
        {
            return Some(full);
        }
    }
    let body = emit_twosection_body(rd, sm5, reads)?;
    Some(alloc::format!("{o}{body}"))
}

fn emit_cbuffer_block(
    o: &mut String,
    cb: &CBufferDef<'_>,
    sm5: bool,
    register: Option<u32>,
    bind_flags: u32,
    cb_reads: Option<&CbReads>,
) -> Option<()> {
    let _ = write!(o, "cbuffer {}", cb.name);
    if let Some(slot) = register {
        let _ = write!(o, " : register(b{slot})");
    }
    if bind_flags != 0 {
        let _ = write!(o, " flags={}", bind_flags_str(bind_flags));
    }
    if cb.cb_type != 0 {
        let _ = write!(o, " kind={}", cb.cb_type);
    }
    if cb.flags != 0 {
        let _ = write!(o, " cbflags={:x}", cb.flags);
    }
    o.push_str(" {\n");
    for v in &cb.variables {
        emit_var(o, v, "    ", sm5, cb_reads)?;
    }
    o.push_str("};\n");
    Some(())
}

/// Two-section body: every binding as a declaration, then every cbuffer layout.
fn emit_twosection_body(
    rd: &ResourceDef<'_>,
    sm5: bool,
    reads: Option<&BTreeMap<u32, CbReads>>,
) -> Option<String> {
    let mut o = String::new();
    for b in &rd.bindings {
        emit_binding(&mut o, b, rd)?;
    }
    if !rd.bindings.is_empty() {
        o.push('\n');
    }
    for cb in &rd.constant_buffers {
        let cbr = cb_reads_for(rd, reads, cb.name.as_ref());
        emit_cbuffer_block(&mut o, cb, sm5, None, 0, cbr)?;
    }
    Some(o)
}

/// Merged body: cbuffer bindings become single `cbuffer N : register(bN) {..}`
/// blocks and resource cbuffer defs are reconstructed from their bindings.
/// Returns None when binding order can't reproduce the cbuffer-def order.
fn emit_merged_body(
    rd: &ResourceDef<'_>,
    sm5: bool,
    reads: Option<&BTreeMap<u32, CbReads>>,
) -> Option<String> {
    // Reconstruction places kind-0 defs (in cbuffer-binding order) first, then
    // resource defs (in structured-binding order); mergeable only when that
    // reproduces the actual cbuffer-def order. Regular cbuffers become
    // `cbuffer N : register(bN) {..}` blocks; resource defs are rebuilt from
    // their `StructuredBuffer<T>` declarations on parse (no kind=3 block).
    let mut expected: Vec<&str> = rd
        .bindings
        .iter()
        .filter(|b| b.input_type == 0)
        .map(|b| b.name.as_ref())
        .collect();
    expected.extend(
        rd.bindings
            .iter()
            .filter(|b| matches!(b.input_type, 5 | 6 | 9 | 10 | 11))
            .map(|b| b.name.as_ref()),
    );
    let dnames: Vec<&str> = rd
        .constant_buffers
        .iter()
        .map(|c| c.name.as_ref())
        .collect();
    // The reconstructed order must contain exactly the cbuffer-def names.
    let mut es = expected.clone();
    es.sort_unstable();
    let mut ds = dnames.clone();
    ds.sort_unstable();
    if es != ds {
        return None;
    }
    let mut o = String::new();
    // When fxc's def order differs from the reconstructed order, record it.
    if expected != dnames {
        o.push_str("cborder");
        for n in &dnames {
            let _ = write!(o, " {n}");
        }
        o.push('\n');
    }
    for b in &rd.bindings {
        if b.input_type == 0 {
            let cb = rd.constant_buffers.iter().find(|c| c.name == b.name)?;
            let cbr = reads.and_then(|m| m.get(&b.bind_point));
            emit_cbuffer_block(&mut o, cb, sm5, Some(b.bind_point), b.flags, cbr)?;
        } else {
            emit_binding(&mut o, b, rd)?;
        }
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
    let inner = reg_tok.strip_prefix("register(")?.split(')').next()?;
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
    let structured = matches!(input_type, 5 | 6 | 9 | 10 | 11);
    if structured {
        dimension = 1; // Buffer
        return_type = 6; // mixed
    }
    if let Some(d) = m.get("dim") {
        dimension = d.parse().ok()?;
    }
    if let Some(r) = m.get("ret") {
        return_type = r.parse().ok()?;
    }
    let num_samples = if structured {
        match m.get("stride") {
            Some(s) => s.parse().ok()?,
            None => 0,
        }
    } else if is_texture(input_type) {
        match m.get("samples") {
            Some(s) => s.parse().ok()?,
            None => 0xFFFF_FFFF, // not multisampled
        }
    } else {
        match m.get("samples") {
            Some(s) => s.parse::<i64>().ok()? as u32,
            None => 0,
        }
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
    /// `None` means no flag tag was present — derive the used bit on parse.
    flags: Option<u32>,
    sm5: Option<[u32; 4]>,
    tex: Option<(u32, u32)>,
    samp: Option<(u32, u32)>,
    default: Vec<u8>,
}

fn parse_var_tail<'a>(toks: impl Iterator<Item = &'a str> + Clone) -> Option<VarTail> {
    let m = kv(toks.clone());
    let mut flags: Option<u32> = None;
    for t in toks {
        if t == "used" {
            flags = Some(BIND_FLAG_USED);
        } else if t == "unused" {
            flags = Some(0);
        }
    }
    if let Some(v) = m.get("vflags") {
        flags = Some(u32::from_str_radix(v, 16).ok()?);
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
pub fn rdef_from_hlsl(text: &str, program: Option<&Program>) -> Option<ResourceDef<'static>> {
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
        rows: Option<u16>,
        cols: Option<u16>,
        sm5: Option<[u32; 4]>,
    }
    enum Block {
        Struct(StructHdr, Vec<MemberDesc<'static>>),
        Cbuffer,
    }
    let mut block: Option<Block> = None;
    let mut sm5 = false;
    // Resource (kind != 0) defs reconstructed in merged mode, appended after all
    // regular cbuffer defs to reproduce fxc's def-array order.
    let mut resource_defs: Vec<CBufferDef<'static>> = Vec::new();
    // Explicit cbuffer-def order (merged mode, when it differs from the natural
    // reconstructed order).
    let mut cb_order: Option<Vec<String>> = None;

    // Merged mode: every cbuffer layout block carries a register, so resource
    // (structured-buffer) defs aren't written out and must be reconstructed
    // from their `StructuredBuffer<T>` declarations. (Two-section mode has
    // register-less `cbuffer N {..}` blocks for them instead.)
    let merged_mode = !text.lines().any(|l| {
        let l = l.trim();
        l.starts_with("cbuffer ") && l.ends_with('{') && !l.contains(": register(")
    });

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        // End of a block.
        if line == "}" || line == "};" {
            match block.take()? {
                Block::Struct(hdr, members) => {
                    // Derive rows=1 and cols=size/4 unless overridden.
                    let struct_size: u32 = members
                        .iter()
                        .filter_map(|m| natural_size(&m.member_type).map(|s| m.offset + s))
                        .max()
                        .unwrap_or(0);
                    let td = TypeDesc {
                        class: hdr.class,
                        var_type: hdr.var_type,
                        rows: hdr.rows.unwrap_or(1),
                        columns: hdr.cols.unwrap_or((struct_size / 4) as u16),
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
                    // `<type> <name> [+<offset>] [sm5=..];` — offset defaults to
                    // the running tight-packed position.
                    let body = line.strip_suffix(';').unwrap_or(line);
                    let mut toks = body.split_whitespace();
                    let mut t0 = toks.next()?;
                    let row_major = t0 == "row_major";
                    if row_major {
                        t0 = toks.next()?;
                    }
                    let mname_tok = toks.next()?;
                    let (mname, arr) = split_name(mname_tok)?;
                    let rest: Vec<&str> = toks.collect();
                    let offset: u32 = match rest.iter().find_map(|t| t.strip_prefix('+')) {
                        Some(s) => s.parse().ok()?,
                        None => members
                            .last()
                            .map(|m| m.offset + natural_size(&m.member_type).unwrap_or(0))
                            .unwrap_or(0),
                    };
                    let m = kv(rest.iter().copied());
                    let mut mt = resolve_type(t0, row_major, &structs, sm5)?;
                    if arr > 0 {
                        mt.elements = arr;
                    }
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
                    let vname_tok = toks.next()?;
                    let (vname, arr) = split_name(vname_tok)?;
                    // Expect `: packoffset(...)`.
                    let _colon = toks.next()?; // ":"
                    let po_tok = toks.next()?; // "packoffset(cN...)"
                    let po = po_tok
                        .strip_prefix("packoffset(")
                        .and_then(|s| s.split(')').next())?;
                    let offset = packoffset_parse(po)?;
                    let tail = parse_var_tail(toks)?;
                    let mut vt = resolve_type(t0, row_major, &structs, sm5)?;
                    if arr > 0 {
                        vt.elements = arr;
                    }
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
                        flags: tail.flags.unwrap_or(DERIVE_USED),
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
            "cborder" => {
                cb_order = Some(rest.split_whitespace().map(String::from).collect());
            }
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
                let hdr = StructHdr {
                    name: String::from(name),
                    class: match m.get("class") {
                        Some(s) => s.parse().ok()?,
                        None => 5,
                    },
                    var_type: match m.get("vtype") {
                        Some(s) => s.parse().ok()?,
                        None => 0,
                    },
                    rows: match m.get("rows") {
                        Some(s) => Some(s.parse().ok()?),
                        None => None,
                    },
                    cols: match m.get("cols") {
                        Some(s) => Some(s.parse().ok()?),
                        None => None,
                    },
                    sm5: match m.get("sm5") {
                        Some(s) => Some(parse_sm5(s)?),
                        None => None,
                    },
                };
                block = Some(Block::Struct(hdr, Vec::new()));
            }
            "cbuffer" => {
                // Three shapes: a merged block `cbuffer N : register(bN) {` (a
                // binding *and* its layout), a two-section layout block
                // `cbuffer N [kind=..] [cbflags=..] {`, or a two-section binding
                // line `cbuffer N : register(bN);`.
                if line.ends_with('{') {
                    let inner = rest.trim_end_matches('{').trim();
                    let mut it = inner.split_whitespace();
                    let name = it.next()?;
                    let toks: Vec<&str> = it.collect();
                    let m = kv(toks.iter().copied());
                    if let Some(reg) = toks.iter().find_map(|t| {
                        t.strip_prefix("register(")
                            .and_then(|s| s.strip_suffix(')'))
                    }) {
                        // Merged: emit both the binding and the layout def.
                        let slot: u32 = reg.get(1..)?.parse().ok()?;
                        rd.bindings.push(ResourceBinding {
                            name: Cow::Owned(String::from(name)),
                            input_type: 0,
                            return_type: 0,
                            dimension: 0,
                            num_samples: 0,
                            bind_point: slot,
                            bind_count: 1,
                            flags: match m.get("flags") {
                                Some(s) => bind_flags_from(s)?,
                                None => 0,
                            },
                        });
                    }
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
            _ => {
                rd.bindings.push(parse_binding(line)?);
                // In merged mode, a structured-buffer binding implies a resource
                // def; reconstruct it from the declared element type `<T>`.
                let (input_type, bname) = {
                    let b = rd.bindings.last()?;
                    (b.input_type, b.name.clone())
                };
                if merged_mode && matches!(input_type, 5 | 6 | 9 | 10 | 11) {
                    let spec = line.split_whitespace().next()?;
                    let elem = spec
                        .split_once('<')
                        .and_then(|(_, r)| r.strip_suffix('>'))?;
                    let et = resolve_type(elem, false, &structs, sm5)?;
                    let size = type_size(&et);
                    resource_defs.push(CBufferDef {
                        name: bname,
                        variables: alloc::vec![CBufferVariable {
                            name: Cow::Owned(String::from("$Element")),
                            offset: 0,
                            size,
                            flags: BIND_FLAG_USED,
                            var_type: et,
                            default_value: Cow::Owned(Vec::new()),
                            texture_start: if sm5 { Some(SLOT_UNUSED) } else { None },
                            texture_size: if sm5 { Some(0) } else { None },
                            sampler_start: if sm5 { Some(SLOT_UNUSED) } else { None },
                            sampler_size: if sm5 { Some(0) } else { None },
                        }],
                        size,
                        flags: 0,
                        cb_type: 3,
                    });
                }
            }
        }
    }

    // Reconstructed resource defs follow all regular cbuffer defs.
    rd.constant_buffers.append(&mut resource_defs);
    // Apply an explicit def order when fxc's differed from the natural one.
    if let Some(order) = &cb_order {
        rd.constant_buffers.sort_by_key(|cb| {
            order
                .iter()
                .position(|n| n.as_str() == cb.name.as_ref())
                .unwrap_or(usize::MAX)
        });
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

    // A structured-buffer binding's stride (num_samples) is sizeof(element),
    // derived from its matching resource def's `$Element` type.
    let strides: Vec<(usize, u32)> = rd
        .bindings
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b.input_type, 5 | 6 | 9 | 10 | 11) && b.num_samples == 0)
        .filter_map(|(i, b)| {
            let cb = rd.constant_buffers.iter().find(|c| c.name == b.name)?;
            let v = cb.variables.first()?;
            Some((i, type_size(&v.var_type)))
        })
        .collect();
    for (i, s) in strides {
        rd.bindings[i].num_samples = s;
    }

    // Resolve cbuffer variables left as "derive" by computing the used bit from
    // the program (or 0 when no program is available).
    let reads = program.map(cbuffer_reads);
    let regmap: BTreeMap<String, u32> = rd
        .bindings
        .iter()
        .filter(|b| b.input_type == 0)
        .map(|b| (String::from(b.name.as_ref()), b.bind_point))
        .collect();
    for cb in &mut rd.constant_buffers {
        let cbr = regmap
            .get(cb.name.as_ref())
            .and_then(|r| reads.as_ref().and_then(|m| m.get(r)));
        for v in &mut cb.variables {
            if v.flags == DERIVE_USED {
                v.flags = match cbr {
                    Some(r) if var_used(r, v.offset, v.size) => BIND_FLAG_USED,
                    _ => 0,
                };
            }
        }
    }
    Some(rd)
}
