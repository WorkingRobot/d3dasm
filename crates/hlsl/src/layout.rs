//! Constant buffer layout — the fields a buffer holds, read from the RDEF chunk.
//!
//! Whatever named the buffer usually says nothing about its inside, while the reflection describes it
//! in full. This lives here rather than beside a caller because it is what a
//! [`Buffer`](crate::Buffer) is declared from, and every consumer has to agree on it: a second copy
//! that sizes a field differently produces a different declaration, and the two only disagree on
//! buffers nobody was looking at.

use dxbc::chunks::rdef;

/// A field of a constant buffer, as the shader declared it.
#[derive(Clone)]
pub struct Member {
    /// Field name (e.g. `"m_TransformMatrix"`).
    pub name: String,
    /// Bytes from the start of one element of the buffer.
    pub offset: u32,
    /// Bytes it takes, including whatever padding follows it before the next field.
    pub size: u32,
    /// How its type reads in HLSL.
    pub kind: String,
}

/// How a type reads in HLSL, which is what the reflection records it as.
pub fn type_name(desc: &rdef::TypeDesc<'_>) -> String {
    let base = match desc.name.is_empty() {
        true => format!("type {}", desc.var_type),
        false => desc.name.to_string(),
    };
    match desc.elements {
        0 => base,
        count => format!("{base}[{count}]"),
    }
}

/// The fields of a constant buffer as the shader declared them.
///
/// A buffer normally holds one struct instance named after itself, whose members are the real
/// fields. One that instead holds a single bare array named after itself says nothing the register
/// grid does not already show, so that case comes back empty.
pub fn members(buffer: &rdef::CBufferDef<'_>) -> Vec<Member> {
    if let [only] = buffer.variables.as_slice() {
        if !only.var_type.members.is_empty() {
            let members = &only.var_type.members;
            // The fields describe one element, so the last of them ends where the element does, not
            // where the buffer does. Running it to the end of the buffer instead hands the last field
            // the whole array, which hides that the buffer holds more than one element and leaves it
            // declared as registers.
            let element = buffer.size / u32::from(only.var_type.elements).max(1);
            return members
                .iter()
                .enumerate()
                .map(|(index, member)| Member {
                    name: member.name.to_string(),
                    offset: member.offset,
                    // Where a field ends is where the next one starts. Working it out from the type
                    // instead misses the padding a matrix row or an array element carries, which
                    // leaves the registers in between looking unaccounted for.
                    size: members
                        .get(index + 1)
                        .map_or(element, |next| next.offset)
                        .saturating_sub(member.offset),
                    kind: type_name(&member.member_type),
                })
                .collect();
        }
        // A buffer holding one bare array named after itself says nothing the register grid does not
        // already show, so it is left to be indexed. A matrix is different: the reflection records
        // its shape, and that shape is the difference between reading a transform and reading four
        // rows that happen to sit together.
        if only.name == buffer.name && only.var_type.rows < 2 {
            return Vec::new();
        }
    }
    buffer
        .variables
        .iter()
        .map(|variable| Member {
            name: variable.name.to_string(),
            offset: variable.offset,
            size: variable.size,
            kind: type_name(&variable.var_type),
        })
        .collect()
}
