//! Parse editable HLSL back into an RDEF.

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::*;
use crate::chunks::rdef::{
    BIND_FLAG_USED, CBufferDef, CBufferVariable, MemberDesc, ResourceBinding, ResourceDef,
    SLOT_UNUSED, TypeDesc,
};
use crate::shex::Program;

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
        None => derived_binding_flags(input_type, return_type),
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
    /// HLSL initializer literal after `=`, resolved against the type later.
    default_init: Option<String>,
}

fn parse_var_tail<'a>(toks: impl Iterator<Item = &'a str> + Clone) -> Option<VarTail> {
    let toks: Vec<&str> = toks.collect();
    let m = kv(toks.iter().copied());
    let mut flags: Option<u32> = None;
    for t in &toks {
        if *t == "used" {
            flags = Some(BIND_FLAG_USED);
        } else if *t == "unused" {
            flags = Some(0);
        }
    }
    if let Some(v) = m.get("vflags") {
        flags = Some(u32::from_str_radix(v, 16).ok()?);
    }
    let default_init = toks
        .iter()
        .position(|t| *t == "=")
        .and_then(|i| toks.get(i + 1))
        .map(|s| String::from(*s));
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
        default_init,
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
                        None => derived_var_size(&vt)?,
                    };
                    let default = match &tail.default_init {
                        Some(init) => parse_default(&vt, init)?,
                        None => tail.default,
                    };
                    let cb = rd.constant_buffers.last_mut()?;
                    cb.variables.push(CBufferVariable {
                        name: Cow::Owned(String::from(vname)),
                        offset,
                        size,
                        flags: tail.flags.unwrap_or(DERIVE_USED),
                        var_type: vt,
                        default_value: Cow::Owned(default),
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
