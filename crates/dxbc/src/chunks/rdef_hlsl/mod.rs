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

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use super::rdef::{ResourceBinding, ResourceDef, TypeDesc, hlsl_type_name, parse_hlsl_type};
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
    element_size(td)
}

/// Byte size of one element (ignoring any array dimension).
fn element_size(td: &TypeDesc<'_>) -> Option<u32> {
    if !td.members.is_empty() {
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

/// Total byte size of a cbuffer variable, applying HLSL array packing: each
/// element is padded to a 16-byte register, except the last keeps its size, so
/// `size = (n-1)·round16(elem) + elem`. Matches the stored size exactly.
fn derived_var_size(td: &TypeDesc<'_>) -> Option<u32> {
    let elem = element_size(td)?;
    Some(if td.elements == 0 {
        elem
    } else {
        (td.elements as u32 - 1) * elem.div_ceil(16) * 16 + elem
    })
}

/// Number of scalar components in a non-array scalar/vector/matrix type.
fn component_count(t: &TypeDesc<'_>) -> Option<usize> {
    if t.elements != 0 || !t.members.is_empty() {
        return None;
    }
    Some(match t.class {
        0 => 1,
        1 => t.columns as usize,
        2 | 3 => t.rows as usize * t.columns as usize,
        _ => return None,
    })
}

/// One scalar component's value → HLSL literal (None if it can't round-trip).
fn scalar_value_str(var_type: u16, w: u32) -> Option<String> {
    Some(match var_type {
        3 => {
            let f = f32::from_bits(w);
            let s = alloc::format!("{f}");
            if s.parse::<f32>().ok()?.to_bits() != w {
                return None;
            }
            s
        }
        2 => alloc::format!("{}", w as i32),
        19 => alloc::format!("{w}"),
        1 if w == 0 => String::from("false"),
        _ => return None,
    })
}

fn scalar_value_parse(var_type: u16, p: &str) -> Option<u32> {
    Some(match var_type {
        3 => p.parse::<f32>().ok()?.to_bits(),
        2 => p.parse::<i32>().ok()? as u32,
        19 => p.parse::<u32>().ok()?,
        1 if p == "false" => 0,
        1 if p == "true" => 1,
        _ => return None,
    })
}

fn read_u32(bytes: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes([
        *bytes.get(off)?,
        *bytes.get(off + 1)?,
        *bytes.get(off + 2)?,
        *bytes.get(off + 3)?,
    ]))
}

/// Render a default value as an HLSL initializer (`1.5`, `float4(1,2,3,4)`,
/// `false`, `{0,1,2}`). Returns `None` for types/values that can't round-trip
/// exactly, so the caller keeps the raw hex form. (No spaces — initializers must
/// stay a single whitespace token.)
fn render_default(t: &TypeDesc<'_>, bytes: &[u8]) -> Option<String> {
    if !t.members.is_empty() {
        return None;
    }
    if t.elements == 0 {
        let comps = component_count(t)?;
        let vals: Option<Vec<String>> = (0..comps)
            .map(|i| scalar_value_str(t.var_type, read_u32(bytes, i * 4)?))
            .collect();
        let vals = vals?;
        if comps == 1 {
            return Some(vals.into_iter().next().unwrap());
        }
        return Some(alloc::format!("{}({})", hlsl_type_name(t)?, vals.join(",")));
    }
    // Array. Only scalar-element arrays (the corpus only has `int[N]`); each
    // element is padded to a 16-byte register, so padding must be zero.
    if t.class != 0 {
        return None;
    }
    let n = t.elements as usize;
    if bytes.len() != (n - 1) * 16 + 4 {
        return None;
    }
    let mut parts = Vec::with_capacity(n);
    for (e, chunk) in bytes.chunks(16).enumerate() {
        // Trailing bytes of each element (beyond the scalar) must be padding.
        if e + 1 < n && chunk.get(4..).is_some_and(|p| p.iter().any(|&b| b != 0)) {
            return None;
        }
        parts.push(scalar_value_str(t.var_type, read_u32(chunk, 0)?)?);
    }
    Some(alloc::format!("{{{}}}", parts.join(",")))
}

/// Parse an HLSL default initializer back into the raw little-endian bytes.
fn parse_default(t: &TypeDesc<'_>, s: &str) -> Option<Vec<u8>> {
    if t.elements != 0 {
        // `{v0,v1,...}` scalar array.
        let inner = s.strip_prefix('{')?.strip_suffix('}')?;
        let parts: Vec<&str> = inner.split(',').collect();
        let n = t.elements as usize;
        if parts.len() != n {
            return None;
        }
        let mut out = alloc::vec![0u8; (n - 1) * 16 + 4];
        for (e, p) in parts.iter().enumerate() {
            let w = scalar_value_parse(t.var_type, p.trim())?;
            out[e * 16..e * 16 + 4].copy_from_slice(&w.to_le_bytes());
        }
        return Some(out);
    }
    let comps = component_count(t)?;
    let parts: Vec<&str> = if comps == 1 {
        alloc::vec![s]
    } else {
        s.split_once('(')?.1.strip_suffix(')')?.split(',').collect()
    };
    if parts.len() != comps {
        return None;
    }
    let mut out = Vec::with_capacity(comps * 4);
    for p in parts {
        out.extend_from_slice(&scalar_value_parse(t.var_type, p.trim())?.to_le_bytes());
    }
    Some(out)
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

/// `D3D_SHADER_INPUT_FLAGS` bits (the binding-flags field uses different bit
/// meanings than the variable-flags field).
const SIF_USER_PACKED: u32 = 0x1;
const SIF_COMPARISON_SAMPLER: u32 = 0x2;
const SIF_TEXTURE_COMPONENT_0: u32 = 0x4;
const SIF_TEXTURE_COMPONENT_1: u32 = 0x8;
const SIF_UNUSED: u32 = 0x10;
/// The two texture-component bits encode (component count − 1).
const SIF_TEX_COMPONENTS: u32 = SIF_TEXTURE_COMPONENT_0 | SIF_TEXTURE_COMPONENT_1;

/// Binding flags → `userPacked|comparisonSampler|...` (or hex for unknown bits).
/// `0` for an explicit no-flags override (when the derivation would set bits).
fn bind_flags_str(f: u32) -> String {
    if f == 0 {
        return String::from("0");
    }
    let mut parts: Vec<&str> = Vec::new();
    let mut known = 0u32;
    for (bit, name) in [
        (SIF_USER_PACKED, "userPacked"),
        (SIF_COMPARISON_SAMPLER, "comparisonSampler"),
        (SIF_TEXTURE_COMPONENT_0, "texComp0"),
        (SIF_TEXTURE_COMPONENT_1, "texComp1"),
        (SIF_UNUSED, "unused"),
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
            "userPacked" => SIF_USER_PACKED,
            "comparisonSampler" => SIF_COMPARISON_SAMPLER,
            "texComp0" => SIF_TEXTURE_COMPONENT_0,
            "texComp1" => SIF_TEXTURE_COMPONENT_1,
            "unused" => SIF_UNUSED,
            other => {
                let other = other.strip_prefix("0x").unwrap_or(other);
                u32::from_str_radix(other, 16).ok()?
            }
        };
    }
    Some(f)
}

/// Binding flags derivable from the declaration: textures/UAVs encode their
/// (4-component `float4`/etc.) return width in the texture-component bits.
fn derived_binding_flags(input_type: u32, return_type: u32) -> u32 {
    if (input_type == 2 || input_type == 4) && return_type != 0 {
        SIF_TEX_COMPONENTS // 4 components → (4-1) → both texComp bits
    } else {
        0
    }
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


mod decode;
mod encode;

pub use decode::rdef_from_hlsl;
pub use encode::rdef_to_hlsl;
