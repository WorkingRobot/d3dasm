//! RDEF chunk parser — resource definitions.
//!
//! The RDEF chunk describes constant buffers, resource bindings, and the
//! compiler creator string. Layout (28-byte header):
//!   0x00: u32 — constant buffer count
//!   0x04: u32 — constant buffer offset
//!   0x08: u32 — bound resource count
//!   0x0C: u32 — bound resource offset
//!   0x10: u32 — target version (minor | major<<8 | type<<16)
//!   0x14: u32 — compile flags
//!   0x18: u32 — creator string offset

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

use nostdio::{ReadLe, Seek, SeekFrom, SliceCursor};

use super::ChunkWriter;
use crate::util::read_cstring;

/// Parsed RDEF chunk — constant buffers, resource bindings, and creator string.
#[derive(Debug)]
pub struct ResourceDef<'a> {
    /// Constant buffer definitions (layouts and variables).
    pub constant_buffers: Vec<CBufferDef<'a>>,
    /// Shader resource bindings (textures, samplers, UAVs, etc.).
    pub bindings: Vec<ResourceBinding<'a>>,
    /// Compiler identification string (e.g. `"Microsoft (R) HLSL Shader Compiler 10.1"`).
    pub creator: Cow<'a, str>,
    /// Target version (minor | major<<8 | type<<16).
    pub target_version: u32,
    /// Compile flags.
    pub compile_flags: u32,
    /// SM5 RD11 sub-header (32 bytes after the main header). None for SM4.
    pub rd11_extra: Option<[u32; 8]>,
}

/// A single constant buffer and its variable layout.
#[derive(Debug)]
pub struct CBufferDef<'a> {
    /// Constant buffer name (e.g. `"$Globals"`, `"cb0"`).
    pub name: Cow<'a, str>,
    /// Variables declared inside the buffer.
    pub variables: Vec<CBufferVariable<'a>>,
    /// Total buffer size in bytes.
    pub size: u32,
    /// Buffer flags.
    pub flags: u32,
    /// Buffer type (0=cbuffer, 1=tbuffer, etc.).
    pub cb_type: u32,
}

/// A variable inside a constant buffer.
#[derive(Debug)]
pub struct CBufferVariable<'a> {
    /// Variable name.
    pub name: Cow<'a, str>,
    /// Byte offset within the constant buffer.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
    /// Variable flags.
    pub flags: u32,
    /// Parsed type descriptor for this variable.
    pub var_type: TypeDesc<'a>,
    /// Default value bytes (empty if none).
    pub default_value: Cow<'a, [u8]>,
    /// SM5 extra: start texture slot (-1 if unused).
    pub texture_start: Option<u32>,
    /// SM5 extra: texture bind count.
    pub texture_size: Option<u32>,
    /// SM5 extra: start sampler slot (-1 if unused).
    pub sampler_start: Option<u32>,
    /// SM5 extra: sampler bind count.
    pub sampler_size: Option<u32>,
}

/// HLSL type descriptor (D3D11_SHADER_TYPE_DESC).
///
/// Binary layout (all u32):
///   0: `[class:16 | type:16]`
///   4: `[rows:16 | columns:16]`
///   8: `[elements:16 | members:16]`
///  12: member descriptor offset
/// SM5 adds 4 unknown u32s + 1 name offset u32.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeDesc<'a> {
    /// Variable class (scalar, vector, matrix, object, struct).
    pub class: u16,
    /// Variable type (void, bool, int, float, etc.).
    pub var_type: u16,
    /// Number of rows (1 for scalars/vectors, >1 for matrices).
    pub rows: u16,
    /// Number of columns.
    pub columns: u16,
    /// Array element count (0 if not an array).
    pub elements: u16,
    /// Struct member descriptors (empty if not a struct).
    pub members: Vec<MemberDesc<'a>>,
    /// SM5 extra: 4 unknown u32 values preserved for round-trip.
    pub sm5_extra: Option<[u32; 4]>,
    /// Type name (SM5 interface types only, empty otherwise).
    pub name: Cow<'a, str>,
}

/// A struct member descriptor inside a type.
///
/// Binary layout (12 bytes):
///   0: u32 name offset
///   4: u32 type offset (recursive)
///   8: u32 byte offset within parent
#[derive(Debug, Clone, PartialEq)]
pub struct MemberDesc<'a> {
    /// Member name.
    pub name: Cow<'a, str>,
    /// Parsed type for this member.
    pub member_type: TypeDesc<'a>,
    /// Byte offset within the parent structure.
    pub offset: u32,
}

/// Resource input type (D3D_SHADER_INPUT_TYPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ResourceInputType {
    /// Constant buffer.
    CBuffer = 0,
    /// Texture buffer.
    TBuffer = 1,
    /// Texture (SRV).
    Texture = 2,
    /// Sampler state.
    Sampler = 3,
    /// Read-write typed UAV.
    UavRwTyped = 4,
    /// Structured buffer (SRV).
    Structured = 5,
    /// Read-write structured buffer (UAV).
    UavRwStructured = 6,
    /// Byte-address buffer (SRV).
    ByteAddress = 7,
    /// Read-write byte-address buffer (UAV).
    UavRwByteAddress = 8,
    /// Append structured buffer (UAV).
    UavAppendStructured = 9,
    /// Consume structured buffer (UAV).
    UavConsumeStructured = 10,
    /// Read-write structured buffer with hidden counter (UAV).
    UavRwStructuredWithCounter = 11,
}

impl ResourceInputType {
    /// Converts a raw `D3D_SHADER_INPUT_TYPE` value to the corresponding variant.
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::CBuffer,
            1 => Self::TBuffer,
            2 => Self::Texture,
            3 => Self::Sampler,
            4 => Self::UavRwTyped,
            5 => Self::Structured,
            6 => Self::UavRwStructured,
            7 => Self::ByteAddress,
            8 => Self::UavRwByteAddress,
            9 => Self::UavAppendStructured,
            10 => Self::UavConsumeStructured,
            11 => Self::UavRwStructuredWithCounter,
            _ => return None,
        })
    }

    /// Returns the lowercase name used in disassembly output.
    pub fn name(self) -> &'static str {
        match self {
            Self::CBuffer => "cbuffer",
            Self::TBuffer => "tbuffer",
            Self::Texture => "texture",
            Self::Sampler => "sampler",
            Self::UavRwTyped => "uav_rwtyped",
            Self::Structured => "structured",
            Self::UavRwStructured => "uav_rwstructured",
            Self::ByteAddress => "byteaddress",
            Self::UavRwByteAddress => "uav_rwbyteaddress",
            Self::UavAppendStructured => "uav_append_structured",
            Self::UavConsumeStructured => "uav_consume_structured",
            Self::UavRwStructuredWithCounter => "uav_rwstructured_with_counter",
        }
    }
}

/// Resource dimension (D3D_SRV_DIMENSION).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ResourceDimension {
    /// Buffer resource.
    Buffer = 1,
    /// 1D texture.
    Texture1D = 2,
    /// 2D texture.
    Texture2D = 3,
    /// 2D multisampled texture.
    Texture2DMS = 4,
    /// 3D (volume) texture.
    Texture3D = 5,
    /// Cube-map texture.
    TextureCube = 6,
    /// 1D texture array.
    Texture1DArray = 7,
    /// 2D texture array.
    Texture2DArray = 8,
    /// 2D multisampled texture array.
    Texture2DMSArray = 9,
    /// Cube-map texture array.
    TextureCubeArray = 10,
}

impl ResourceDimension {
    /// Converts a raw `D3D_SRV_DIMENSION` value to the corresponding variant.
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            1 => Self::Buffer,
            2 => Self::Texture1D,
            3 => Self::Texture2D,
            4 => Self::Texture2DMS,
            5 => Self::Texture3D,
            6 => Self::TextureCube,
            7 => Self::Texture1DArray,
            8 => Self::Texture2DArray,
            9 => Self::Texture2DMSArray,
            10 => Self::TextureCubeArray,
            _ => return None,
        })
    }

    /// Returns the short dimension name used in disassembly output.
    pub fn name(self) -> &'static str {
        match self {
            Self::Buffer => "buf",
            Self::Texture1D => "1d",
            Self::Texture2D => "2d",
            Self::Texture2DMS => "2dMS",
            Self::Texture3D => "3d",
            Self::TextureCube => "cube",
            Self::Texture1DArray => "1darray",
            Self::Texture2DArray => "2darray",
            Self::Texture2DMSArray => "2dMSarray",
            Self::TextureCubeArray => "cubearray",
        }
    }
}

/// Binding was explicitly packed by the user.
pub const BIND_FLAG_USER_PACKED: u32 = 0x1;
/// Binding is actually used by the shader.
pub const BIND_FLAG_USED: u32 = 0x2;
/// Sampler is a comparison sampler.
pub const BIND_FLAG_COMPARISON_SAMPLER: u32 = 0x4;
/// Texture component flag bit 0.
pub const BIND_FLAG_TEX_COMP_0: u32 = 0x8;
/// Texture component flag bit 1.
pub const BIND_FLAG_TEX_COMP_1: u32 = 0x10;

/// Sentinel value for unused texture/sampler start slots.
pub const SLOT_UNUSED: u32 = 0xFFFFFFFF;

/// A shader resource binding (texture, sampler, cbuffer, UAV, etc.).
#[derive(Debug)]
pub struct ResourceBinding<'a> {
    /// Binding name.
    pub name: Cow<'a, str>,
    /// Resource type (0=cbuffer, 2=texture, 3=sampler, 4=uav_rwtyped, …).
    pub input_type: u32,
    /// Return type for typed resources.
    pub return_type: u32,
    /// Resource dimension (1=buffer, 2=1d, 3=2d, …).
    pub dimension: u32,
    /// Number of samples (for multisampled resources).
    pub num_samples: u32,
    /// Register slot (e.g. `t0`, `s1`, `b2`).
    pub bind_point: u32,
    /// Number of contiguous registers bound.
    pub bind_count: u32,
    /// Binding flags (userPacked, used, comparisonSampler, …).
    pub flags: u32,
}

impl ResourceBinding<'_> {
    fn type_name(&self) -> &'static str {
        match ResourceInputType::from_u32(self.input_type) {
            Some(t) => t.name(),
            None => "unknown",
        }
    }

    fn dim_name(&self) -> &'static str {
        match ResourceDimension::from_u32(self.dimension) {
            Some(d) => d.name(),
            None => "NA",
        }
    }

    fn write_flags(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.flags == 0 {
            return Ok(());
        }
        let parts: &[(&str, u32)] = &[
            ("userPacked", BIND_FLAG_USER_PACKED),
            ("used", BIND_FLAG_USED),
            ("comparisonSampler", BIND_FLAG_COMPARISON_SAMPLER),
            ("texComp0", BIND_FLAG_TEX_COMP_0),
            ("texComp1", BIND_FLAG_TEX_COMP_1),
        ];
        let mut first = true;
        let mut matched = false;
        for &(name, bit) in parts {
            if self.flags & bit != 0 {
                if !first {
                    f.write_str(";")?;
                }
                f.write_str(name)?;
                first = false;
                matched = true;
            }
        }
        if !matched {
            write!(f, "0x{:x}", self.flags)?;
        }
        Ok(())
    }

    /// Write the type, dimension, slot, bind count, and flags columns.
    pub fn fmt_columns(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<12} {:<8} {:<4} {:<5}",
            self.type_name(),
            self.dim_name(),
            self.bind_point,
            self.bind_count,
        )?;
        if self.flags != 0 {
            f.write_str(" ")?;
            self.write_flags(f)?;
        }
        Ok(())
    }
}

impl fmt::Display for ResourceBinding<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<20} ", self.name)?;
        self.fmt_columns(f)
    }
}

/// Parse an RDEF chunk.
///
/// RDEF header layout (28 bytes):
///   0: u32 constant_buffer_count
///   4: u32 constant_buffer_offset
///   8: u32 bound_resource_count
///  12: u32 bound_resource_offset
///  16: u32 target_version  (minor | major<<8 | type<<16)
///  20: u32 flags
///  24: u32 creator_offset
pub fn parse_rdef(data: &[u8]) -> Option<ResourceDef<'_>> {
    if data.len() < 28 {
        return None;
    }

    let mut c = SliceCursor::new(data);
    let cb_count = c.read_u32_le().ok()? as usize;
    let cb_offset = c.read_u32_le().ok()? as usize;
    let binding_count = c.read_u32_le().ok()? as usize;
    let binding_offset = c.read_u32_le().ok()? as usize;
    let target_version = c.read_u32_le().ok()?;
    let compile_flags = c.read_u32_le().ok()?;

    // SM5 uses 40-byte variable descriptors; SM4 uses 24-byte.
    let major_version = (target_version >> 8) & 0xFF;
    let is_sm5 = major_version >= 5;
    let var_stride: usize = if is_sm5 { 40 } else { 24 };

    // Creator string (at offset 24 in the RDEF header)
    let creator_off = c.read_u32_le().ok()? as usize;
    let creator: Cow<'_, str> = if creator_off < data.len() {
        Cow::Borrowed(read_cstring(data, creator_off))
    } else {
        Cow::Borrowed("")
    };

    // SM5 has an RD11 sub-header (8 u32s = 32 bytes) after the main header.
    let rd11_extra = if is_sm5 && data.len() >= 60 {
        Some([
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
        ])
    } else {
        None
    };

    // Parse resource bindings
    let mut bindings = Vec::with_capacity(binding_count);
    for i in 0..binding_count {
        let base = binding_offset + i * 32;
        if base + 32 > data.len() {
            break;
        }
        c.seek(SeekFrom::Start(base as u64)).ok()?;
        let name_off = c.read_u32_le().ok()? as usize;
        let input_type = c.read_u32_le().ok()?;
        let return_type = c.read_u32_le().ok()?;
        let dimension = c.read_u32_le().ok()?;
        let num_samples = c.read_u32_le().ok()?;
        let bind_point = c.read_u32_le().ok()?;
        let bind_count = c.read_u32_le().ok()?;
        let flags = c.read_u32_le().ok()?;
        bindings.push(ResourceBinding {
            name: Cow::Borrowed(read_cstring(data, name_off)),
            input_type,
            return_type,
            dimension,
            num_samples,
            bind_point,
            bind_count,
            flags,
        });
    }

    // Parse constant buffers
    let mut constant_buffers = Vec::with_capacity(cb_count);
    for i in 0..cb_count {
        let base = cb_offset + i * 24;
        if base + 24 > data.len() {
            break;
        }
        c.seek(SeekFrom::Start(base as u64)).ok()?;
        let name_off = c.read_u32_le().ok()? as usize;
        let var_count = c.read_u32_le().ok()? as usize;
        let var_offset = c.read_u32_le().ok()? as usize;
        let cb_size = c.read_u32_le().ok()?;
        let cb_flags = c.read_u32_le().ok()?;
        let cb_type = c.read_u32_le().ok()?;

        let mut variables = Vec::with_capacity(var_count);
        for j in 0..var_count {
            let vbase = var_offset + j * var_stride;
            if vbase + var_stride > data.len() {
                break;
            }
            c.seek(SeekFrom::Start(vbase as u64)).ok()?;
            let vname_off = c.read_u32_le().ok()? as usize;
            let v_offset = c.read_u32_le().ok()?;
            let v_size = c.read_u32_le().ok()?;
            let v_flags = c.read_u32_le().ok()?;
            let v_type_offset = c.read_u32_le().ok()? as usize;
            let v_default_value_offset = c.read_u32_le().ok()? as usize;
            let (tex_start, tex_size, samp_start, samp_size) = if is_sm5 {
                (
                    Some(c.read_u32_le().ok()?),
                    Some(c.read_u32_le().ok()?),
                    Some(c.read_u32_le().ok()?),
                    Some(c.read_u32_le().ok()?),
                )
            } else {
                (None, None, None, None)
            };

            let var_type = parse_type_desc(data, v_type_offset, is_sm5);
            let default_value: Cow<'_, [u8]> = if v_default_value_offset != 0 && v_size > 0 {
                let end = v_default_value_offset + v_size as usize;
                if end <= data.len() {
                    Cow::Borrowed(&data[v_default_value_offset..end])
                } else {
                    Cow::Borrowed(&[])
                }
            } else {
                Cow::Borrowed(&[])
            };

            variables.push(CBufferVariable {
                name: Cow::Borrowed(read_cstring(data, vname_off)),
                offset: v_offset,
                size: v_size,
                flags: v_flags,
                var_type,
                default_value,
                texture_start: tex_start,
                texture_size: tex_size,
                sampler_start: samp_start,
                sampler_size: samp_size,
            });
        }

        constant_buffers.push(CBufferDef {
            name: Cow::Borrowed(read_cstring(data, name_off)),
            variables,
            size: cb_size,
            flags: cb_flags,
            cb_type,
        });
    }

    Some(ResourceDef {
        constant_buffers,
        bindings,
        creator,
        target_version,
        compile_flags,
        rd11_extra,
    })
}

/// Parse a type descriptor at `offset` within the RDEF chunk data.
fn parse_type_desc<'a>(data: &'a [u8], offset: usize, is_sm5: bool) -> TypeDesc<'a> {
    let empty = TypeDesc {
        class: 0,
        var_type: 0,
        rows: 0,
        columns: 0,
        elements: 0,
        members: Vec::new(),
        sm5_extra: None,
        name: Cow::Borrowed(""),
    };
    if offset + 16 > data.len() {
        return empty;
    }
    let mut c = SliceCursor::new(data);
    if c.seek(SeekFrom::Start(offset as u64)).is_err() {
        return empty;
    }
    let class_type = match c.read_u32_le() {
        Ok(v) => v,
        Err(_) => return empty,
    };
    let rows_cols = match c.read_u32_le() {
        Ok(v) => v,
        Err(_) => return empty,
    };
    let elems_members = match c.read_u32_le() {
        Ok(v) => v,
        Err(_) => return empty,
    };
    let member_offset = match c.read_u32_le() {
        Ok(v) => v as usize,
        Err(_) => return empty,
    };

    let class = (class_type & 0xFFFF) as u16;
    let var_type = (class_type >> 16) as u16;
    let rows = (rows_cols & 0xFFFF) as u16;
    let columns = (rows_cols >> 16) as u16;
    let elements = (elems_members & 0xFFFF) as u16;
    let member_count = (elems_members >> 16) as u16;

    let sm5_extra = if is_sm5 {
        let a = c.read_u32_le().unwrap_or(0);
        let b = c.read_u32_le().unwrap_or(0);
        let d = c.read_u32_le().unwrap_or(0);
        let e = c.read_u32_le().unwrap_or(0);
        Some([a, b, d, e])
    } else {
        None
    };

    // Parse member descriptors (12 bytes each: name_off, type_off, byte_offset)
    let mut members = Vec::new();
    if member_count > 0 && member_offset + (member_count as usize) * 12 <= data.len() {
        members.reserve(member_count as usize);
        for k in 0..member_count as usize {
            let mbase = member_offset + k * 12;
            if c.seek(SeekFrom::Start(mbase as u64)).is_err() {
                break;
            }
            let mname_off = c.read_u32_le().unwrap_or(0) as usize;
            let mtype_off = c.read_u32_le().unwrap_or(0) as usize;
            let moffset = c.read_u32_le().unwrap_or(0);
            members.push(MemberDesc {
                name: Cow::Borrowed(read_cstring(data, mname_off)),
                member_type: parse_type_desc(data, mtype_off, is_sm5),
                offset: moffset,
            });
        }
    }

    // SM5: name offset comes after the 4 unknowns in the type descriptor body
    let name: Cow<'a, str> = if is_sm5 {
        // Name offset is at: base + 16 (fixed) + 16 (4 unknowns) = base + 32
        let name_pos = offset + 32;
        if name_pos + 4 <= data.len() {
            if c.seek(SeekFrom::Start(name_pos as u64)).is_ok() {
                let noff = c.read_u32_le().unwrap_or(0) as usize;
                if noff > 0 && noff < data.len() {
                    Cow::Borrowed(read_cstring(data, noff))
                } else {
                    Cow::Borrowed("")
                }
            } else {
                Cow::Borrowed("")
            }
        } else {
            Cow::Borrowed("")
        }
    } else {
        Cow::Borrowed("")
    };

    TypeDesc {
        class,
        var_type,
        rows,
        columns,
        elements,
        members,
        sm5_extra,
        name,
    }
}

impl ResourceDef<'_> {
    fn is_sm5(&self) -> bool {
        ((self.target_version >> 8) & 0xFF) >= 5
    }
}

// Low-level helpers for the byte-exact RDEF writer.

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn patch_u32(out: &mut [u8], pos: usize, v: u32) {
    out[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
}

/// Pad `out` to a 4-byte boundary with fxc's 0xAB fill byte.
fn pad4(out: &mut Vec<u8>) {
    while out.len() % 4 != 0 {
        out.push(0xAB);
    }
}

/// Intern a null-terminated string into a single content-deduped pool,
/// appending it at the current end of `out` on first use.
fn intern(out: &mut Vec<u8>, pool: &mut alloc::collections::BTreeMap<alloc::string::String, u32>, s: &str) -> u32 {
    if let Some(&off) = pool.get(s) {
        return off;
    }
    let off = out.len() as u32;
    out.extend_from_slice(s.as_bytes());
    out.push(0);
    pool.insert(alloc::string::String::from(s), off);
    off
}

/// Resolve a variable's type to a descriptor offset, reproducing fxc's sharing:
/// non-array types (`elements == 0`) are de-duplicated by value; array types
/// always get a fresh descriptor. Returns the assigned offset, placing the
/// type's name string + descriptor at the current cursor on first emission.
///
/// Struct members are not laid out here, so types with members won't reproduce
/// exactly — those RDEFs fall back to raw hex via the emit-time verification.
fn place_type<'a>(
    out: &mut Vec<u8>,
    pool: &mut alloc::collections::BTreeMap<alloc::string::String, u32>,
    placed: &mut Vec<(TypeDesc<'a>, u32)>,
    td: &TypeDesc<'a>,
    is_sm5: bool,
) -> u32 {
    if td.elements == 0 {
        if let Some((_, off)) = placed.iter().find(|(t, _)| t == td) {
            return *off;
        }
    }
    // Type name first (struct/object types reference an interned name string).
    let name_off = if td.name.is_empty() {
        0
    } else {
        intern(out, pool, &td.name)
    };
    // For struct types: lay out member names + member type descriptors, then
    // the member-descriptor array, before the struct descriptor itself.
    let member_off = if td.members.is_empty() {
        0u32
    } else {
        let mut child_offsets = Vec::with_capacity(td.members.len());
        for m in &td.members {
            intern(out, pool, &m.name);
            child_offsets.push(place_type(out, pool, placed, &m.member_type, is_sm5));
        }
        pad4(out);
        let moff = out.len() as u32;
        for (i, m) in td.members.iter().enumerate() {
            let mn = intern(out, pool, &m.name);
            push_u32(out, mn);
            push_u32(out, child_offsets[i]);
            push_u32(out, m.offset);
        }
        moff
    };
    pad4(out);
    let off = out.len() as u32;
    push_u32(out, (td.class as u32) | ((td.var_type as u32) << 16));
    push_u32(out, (td.rows as u32) | ((td.columns as u32) << 16));
    push_u32(out, (td.elements as u32) | ((td.members.len() as u32) << 16));
    push_u32(out, member_off);
    if is_sm5 {
        match &td.sm5_extra {
            Some(e) => {
                for x in e {
                    push_u32(out, *x);
                }
            }
            None => out.extend_from_slice(&[0u8; 16]),
        }
        push_u32(out, name_off);
    }
    placed.push((td.clone(), off));
    off
}

impl ChunkWriter for ResourceDef<'_> {
    fn fourcc(&self) -> [u8; 4] {
        *b"RDEF"
    }

    fn write_payload(&self) -> Vec<u8> {
        use alloc::collections::BTreeMap;
        use alloc::string::String;

        let is_sm5 = self.is_sm5();
        let rd11_size = if self.rd11_extra.is_some() { 32 } else { 0 };

        let mut out: Vec<u8> = Vec::new();
        // Binding/cbuffer names share one pool; variable + type names use a
        // separate pool (fxc does not dedup variable names against bindings).
        let mut bpool: BTreeMap<String, u32> = BTreeMap::new();
        let mut vpool: BTreeMap<String, u32> = BTreeMap::new();

        // Header (28 bytes). cb_offset and creator_offset are patched later.
        push_u32(&mut out, self.constant_buffers.len() as u32);
        let cb_off_pos = out.len();
        push_u32(&mut out, 0);
        push_u32(&mut out, self.bindings.len() as u32);
        push_u32(&mut out, (28 + rd11_size) as u32); // binding section always present
        push_u32(&mut out, self.target_version);
        push_u32(&mut out, self.compile_flags);
        let creator_pos = out.len();
        push_u32(&mut out, 0);

        if let Some(rd11) = &self.rd11_extra {
            for v in rd11 {
                push_u32(&mut out, *v);
            }
        }

        // Resource bindings, then their name strings (cbuffer names share these).
        let mut binding_name_pos = Vec::with_capacity(self.bindings.len());
        for b in &self.bindings {
            binding_name_pos.push(out.len());
            push_u32(&mut out, 0);
            push_u32(&mut out, b.input_type);
            push_u32(&mut out, b.return_type);
            push_u32(&mut out, b.dimension);
            push_u32(&mut out, b.num_samples);
            push_u32(&mut out, b.bind_point);
            push_u32(&mut out, b.bind_count);
            push_u32(&mut out, b.flags);
        }
        for (i, b) in self.bindings.iter().enumerate() {
            let off = intern(&mut out, &mut bpool, &b.name);
            patch_u32(&mut out, binding_name_pos[i], off);
        }

        // Constant-buffer descriptors (cb_offset is 0 when there are none).
        // Only align when a cbuffer section follows.
        if !self.constant_buffers.is_empty() {
            pad4(&mut out);
        }
        let cb_section = out.len() as u32;
        patch_u32(
            &mut out,
            cb_off_pos,
            if self.constant_buffers.is_empty() { 0 } else { cb_section },
        );
        let mut cb_name_pos = Vec::with_capacity(self.constant_buffers.len());
        let mut cb_var_pos = Vec::with_capacity(self.constant_buffers.len());
        for cb in &self.constant_buffers {
            cb_name_pos.push(out.len());
            push_u32(&mut out, 0);
            push_u32(&mut out, cb.variables.len() as u32);
            cb_var_pos.push(out.len());
            push_u32(&mut out, 0);
            push_u32(&mut out, cb.size);
            push_u32(&mut out, cb.flags);
            push_u32(&mut out, cb.cb_type);
        }
        for (i, cb) in self.constant_buffers.iter().enumerate() {
            let off = intern(&mut out, &mut bpool, &cb.name);
            patch_u32(&mut out, cb_name_pos[i], off);
        }

        // Per cbuffer: variable descriptors, then per-variable name/type/default.
        let mut placed_types: Vec<(TypeDesc<'_>, u32)> = Vec::new();
        for (ci, cb) in self.constant_buffers.iter().enumerate() {
            pad4(&mut out);
            let var_section = out.len() as u32;
            patch_u32(&mut out, cb_var_pos[ci], var_section);

            let mut name_pos = Vec::with_capacity(cb.variables.len());
            let mut type_pos = Vec::with_capacity(cb.variables.len());
            let mut def_pos = Vec::with_capacity(cb.variables.len());
            for v in &cb.variables {
                name_pos.push(out.len());
                push_u32(&mut out, 0);
                push_u32(&mut out, v.offset);
                push_u32(&mut out, v.size);
                push_u32(&mut out, v.flags);
                type_pos.push(out.len());
                push_u32(&mut out, 0);
                def_pos.push(out.len());
                push_u32(&mut out, 0);
                if is_sm5 {
                    push_u32(&mut out, v.texture_start.unwrap_or(SLOT_UNUSED));
                    push_u32(&mut out, v.texture_size.unwrap_or(0));
                    push_u32(&mut out, v.sampler_start.unwrap_or(SLOT_UNUSED));
                    push_u32(&mut out, v.sampler_size.unwrap_or(0));
                }
            }
            for (vi, v) in cb.variables.iter().enumerate() {
                let noff = intern(&mut out, &mut vpool, &v.name);
                patch_u32(&mut out, name_pos[vi], noff);
                let toff = place_type(&mut out, &mut vpool, &mut placed_types, &v.var_type, is_sm5);
                patch_u32(&mut out, type_pos[vi], toff);
                if v.default_value.is_empty() {
                    patch_u32(&mut out, def_pos[vi], 0);
                } else {
                    let doff = out.len() as u32;
                    out.extend_from_slice(&v.default_value);
                    patch_u32(&mut out, def_pos[vi], doff);
                }
            }
        }

        // Creator string, last.
        let coff = intern(&mut out, &mut vpool, &self.creator);
        patch_u32(&mut out, creator_pos, coff);

        // Pad the whole chunk to a 4-byte boundary.
        pad4(&mut out);
        out
    }
}
