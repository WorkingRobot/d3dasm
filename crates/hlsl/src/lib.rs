//! `hlsl` — a decompiler: a shader read back as HLSL rather than as the instruction stream it
//! compiles to.
//!
//! SM4/SM5 bytecode keeps its control flow structured — `if` and `endif` are matched markers, there
//! are no jumps and no subroutines — so recovering blocks is a walk over those markers rather than
//! the graph analysis decompiling usually needs. What the bytecode does not keep is expressions:
//! every instruction writes a register, so a line of source comes back as a dozen of them. The rest
//! of this crate puts those back together.
//!
//! # Quick start
//!
//! ```rust,ignore
//! // `names` is empty here, so a constant reads as `cb0[3].xyz`. See `Names`.
//! let names = hlsl::Names::default();
//! for line in hlsl::decompile(&program, &names).lines {
//!     println!("{line}");
//! }
//! ```
//!
//! # What it invents
//!
//! The output is meant to be source rather than a sketch of it, and compiles. What the bytecode
//! cannot say, this invents rather than leaves out — an entry point and signature, a struct for each
//! constant buffer, and a written-out function wherever the machine has an operation the language
//! does not. It targets a modern compiler, so a per-component choice is `select` rather than the
//! conditional operator.
//!
//! Names are the one thing it will not invent. A register is called something only because the
//! caller said so, via [`Names`]; [`layout`] reads a constant buffer's fields out of the RDEF chunk
//! to help fill it. With none supplied, every register reads as the raw slot.
//!
//! # What it does not promise
//!
//! It compiling is not it being right. Nothing checks that it computes what the original did, and
//! the resources it declares are shaped to suit the reading rather than to match the bindings the
//! caller makes.

use dxbc::shex::{Instruction, InstructionKind, Opcode, Program, ReturnType};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Constant buffer field layout, read from the RDEF chunk.
pub mod layout;

mod body;
mod expr;
mod idiom;
mod matrix;

/// The names to give the registers a shader binds. Slots are per-shader, so this is built against
/// the shader being read rather than against whatever collection it came out of.
///
/// Every map is optional: a slot with no entry reads as the register itself.
#[derive(Default)]
pub struct Names {
    /// Constant buffers, by `b` slot.
    pub constants: HashMap<u16, Buffer>,
    /// Texture and buffer resources, by `t` slot.
    pub textures: HashMap<u16, String>,
    /// Samplers, by `s` slot.
    pub samplers: HashMap<u16, String>,
    /// The shader's own input signature, by register.
    pub inputs: HashMap<u32, Semantic>,
    /// The shader's own output signature, by register.
    pub outputs: HashMap<u32, Semantic>,
}

/// A constant buffer and what sits in it.
pub struct Buffer {
    /// Buffer name (e.g. `"g_CameraParameter"`).
    pub name: String,
    /// The fields and the pads covering what they leave between them, in declaration order. HLSL
    /// packs a member after the one before it, so a field only lands on the register the bytecode
    /// reads it from when everything ahead of it takes exactly the room it claims.
    members: Vec<Member>,
}

/// One line of a buffer declaration, and the room it takes.
struct Member {
    /// Member name, or the generated name of a pad.
    name: String,
    /// The type as declared, which is not always how the field was described.
    kind: String,
    /// Elements, where the field is declared as the registers it spans rather than as its own type.
    elements: u32,
    /// The register its first component sits in, counted from the start of the buffer.
    register: u32,
    /// The first component of `register` it sits in.
    first: u8,
    /// Registers it spans.
    registers: u32,
    /// Components of each register it fills, which a matrix narrower than four leaves short.
    width: u8,
}

/// A field of a constant buffer, as the caller wants it declared.
pub struct Field {
    /// Field name (e.g. `"m_TransformMatrix"`).
    pub name: String,
    /// How the field is declared, so the preamble reads as the HLSL that would bind it.
    pub kind: String,
    /// Its register, counted from the start of the buffer.
    pub register: u32,
    /// Registers it spans, which is what says whether indexing into it is meaningful.
    pub registers: u32,
    /// Which components of its first register it covers. Whole registers for a buffer the reflection
    /// described; a real mask where the caller knows several fields share one register.
    pub mask: u8,
}

/// One entry of a shader's input or output signature. The name is both what the body calls it and
/// what the pipeline binds it to, which HLSL is happy to have be the same word.
pub struct Semantic {
    /// Semantic name, with its index where the signature has more than one (e.g. `"TEXCOORD2"`).
    pub name: String,
    /// Its declared type, sized to the components the signature marks.
    pub kind: String,
}

impl Semantic {
    /// A signature entry as the shader declared it. The index is only spelled out where there is
    /// more than one of the semantic, since a system value with a nought after it is not always
    /// accepted.
    pub fn new(name: &str, index: u32, component_type: u32, mask: u8) -> Self {
        let base = match component_type {
            1 => "uint",
            2 => "int",
            _ => "float",
        };
        Self {
            name: match index {
                0 => name.to_owned(),
                index => format!("{name}{index}"),
            },
            kind: match 8 - mask.max(1).leading_zeros() {
                1 => base.to_owned(),
                lanes => format!("{base}{lanes}"),
            },
        }
    }
}

impl Buffer {
    /// A buffer and its fields, tiled so that implicit packing places each field where the bytecode
    /// reads it.
    pub fn new(name: String, mut fields: Vec<Field>) -> Self {
        fields.sort_by_key(|field| (field.register, field.mask.trailing_zeros()));
        let mut members: Vec<Member> = fields.iter().map(Member::new).collect();
        members.extend(gaps(&members));
        members.sort_by_key(|member| (member.register, member.first));
        Self { name, members }
    }

    /// Whether the buffer is one member named after itself, which a hand-written header declares
    /// straight into the `cbuffer` with no struct around it. Spelling the name twice to reach in
    /// says nothing, so it is left off — and a gap beside such a member takes the buffer's name so
    /// that two buffers cannot both declare a `pad3` at the outer scope.
    fn alone(&self) -> bool {
        matches!(self.members.as_slice(), [only] if only.name == self.name)
    }

    /// What a member of this buffer is written under.
    fn inside(&self) -> String {
        match self.alone() {
            true => String::new(),
            false => format!("{}.", self.name),
        }
    }

    /// What a register no member covers is written under.
    fn gap(&self, register: u32) -> String {
        match self.alone() {
            true => format!("{}_{}", self.name, padding(register)),
            false => format!("{}.{}", self.name, padding(register)),
        }
    }

    /// The rows of a member declared as a matrix. An array of matrices is declared as the registers
    /// it spans, so its name is not a matrix to multiply by and this passes it over.
    pub(crate) fn rows(&self, member: &str) -> Option<u32> {
        let held = self.members.iter().find(|held| held.name == member)?;
        let (rows, _) = held.kind.strip_prefix("float")?.split_once('x')?;
        rows.parse().ok()
    }

    /// The register past the last one any member reaches.
    pub(crate) fn span(&self) -> u32 {
        span(&self.members)
    }
}

fn span(members: &[Member]) -> u32 {
    members
        .iter()
        .map(|member| member.register + member.registers)
        .max()
        .unwrap_or(0)
}

impl Member {
    fn new(field: &Field) -> Self {
        // A matrix answers to being indexed by register, since a row is a register, and an array of
        // them answers to being indexed twice: by element, and then by row within it. An array of
        // anything else counts elements rather than registers, so it keeps the registers it is,
        // which is what the bytecode addresses.
        //
        // Rows narrower than a register are the other way to lose a place: they leave a gap in every
        // one of them, and there is nowhere to declare a pad in the middle of a matrix.
        let (base, count) = shape(&field.kind);
        let own = base.ends_with("x4") || (count == 0 && field.registers == 1);
        let (kind, elements) = match own {
            true => (base, count),
            false => ("float4".to_owned(), field.registers),
        };
        let (registers, width) = footprint(&kind, elements);
        Self {
            name: field.name.clone(),
            kind,
            elements,
            register: field.register,
            first: field.mask.trailing_zeros().min(3) as u8,
            registers,
            width,
        }
    }

    /// Rows in each element. One where the member is not a matrix, since a register is then the
    /// whole of an element.
    fn rows(&self) -> u32 {
        let Some((left, _)) = self.kind.split_once('x') else {
            return 1;
        };
        left.trim_start_matches(|held: char| !held.is_ascii_digit())
            .parse()
            .unwrap_or(1)
    }

    fn covers(&self, register: u32, comp: u8) -> bool {
        if register < self.register || register >= self.register + self.registers {
            return false;
        }
        let first = match register == self.register {
            true => self.first,
            false => 0,
        };
        comp >= first && comp < first + self.width
    }

    /// Where the member sits in one element of the buffer, for a buffer whose elements are picked at
    /// run time and so cannot be reached into by name.
    fn placed(&self) -> String {
        let lanes: String = (self.first..self.first + self.width)
            .map(expr::lane)
            .collect();
        let at = match self.registers > 1 {
            true => format!(
                "[{}..{}]",
                self.register,
                self.register + self.registers - 1
            ),
            false => format!("[{}].{lanes}", self.register),
        };
        format!("//   {at:<10} {} {}{}", self.kind, self.name, self.suffix())
    }

    fn suffix(&self) -> String {
        match self.elements {
            0 => String::new(),
            count => format!("[{count}]"),
        }
    }

    fn declaration(&self) -> String {
        // Only what is still declared a matrix, and only so that indexing one gives the register the
        // body asked for rather than a column of it.
        let order = match self.kind.contains('x') {
            true => "row_major ",
            false => "",
        };
        format!("    {order}{} {}{};", self.kind, self.name, self.suffix())
    }
}

/// A declared type split into its shape and how many of them: `float3x4[2]` is three rows of four,
/// twice over.
fn shape(kind: &str) -> (String, u32) {
    match kind.split_once('[') {
        Some((base, rest)) => (
            base.to_owned(),
            rest.trim_end_matches(']').parse().unwrap_or(0),
        ),
        None => (kind.to_owned(), 0),
    }
}

/// How much of a buffer a declared type takes: registers, and components of each. An array spends a
/// whole register on every element, and a matrix one on every row, so an array of matrices spends
/// one on each row of each element.
fn footprint(kind: &str, elements: u32) -> (u32, u8) {
    let digits: Vec<u32> = kind.chars().filter_map(|c| c.to_digit(10)).collect();
    let (rows, width) = match digits.as_slice() {
        [rows, columns] => (*rows, (*columns).clamp(1, 4) as u8),
        [lanes] => (1, (*lanes).clamp(1, 4) as u8),
        _ => (1, 1),
    };
    (rows * elements.max(1), width)
}

/// The pads filling what the fields leave between them, without which the member after a gap slides
/// down into it and takes every member after it along.
fn gaps(members: &[Member]) -> Vec<Member> {
    let mut used = vec![0u8; span(members) as usize];
    for member in members {
        for register in member.register..member.register + member.registers {
            let first = match register == member.register {
                true => member.first,
                false => 0,
            };
            used[register as usize] |= (((1u16 << member.width) - 1) << first) as u8;
        }
    }

    let mut out = Vec::new();
    for (register, taken) in used.into_iter().enumerate() {
        let register = register as u32;
        let mut comp = 0u8;
        while comp < 4 {
            let width = (comp..4).take_while(|at| taken & (1 << at) == 0).count() as u8;
            if width == 0 {
                comp += 1;
                continue;
            }
            out.push(Member {
                name: match width {
                    4 => padding(register),
                    _ => format!("{}_{comp}", padding(register)),
                },
                kind: match width {
                    1 => "float".to_owned(),
                    lanes => format!("float{lanes}"),
                },
                elements: 0,
                register,
                first: comp,
                registers: 1,
                width,
            });
            comp += width;
        }
    }
    out
}

impl Field {
    /// A field the bytecode's reflection described, placed by where its bytes start.
    pub fn described(name: String, kind: String, offset: u32, size: u32) -> Self {
        Self {
            name,
            kind,
            register: offset / 16,
            registers: size.div_ceil(16).max(1),
            // Fields narrower than a register share it with their neighbours, so each says which
            // components are its own.
            mask: match size >= 16 {
                true => 0xF,
                false => {
                    let first = (offset % 16) / 4;
                    let lanes = (size / 4).clamp(1, 4);
                    (((1u32 << lanes) - 1) << first) as u8
                }
            },
        }
    }

    /// A material parameter, which shares its register with up to three others.
    pub fn packed(name: String, size: u16, register: u32, mask: u8) -> Self {
        Self {
            name,
            kind: match size / 4 {
                1 => "float".to_owned(),
                lanes => format!("float{lanes}"),
            },
            register,
            registers: 1,
            mask,
        }
    }
}

/// How a declared type is read, so an integer input is not taken for a float.
pub(crate) fn domain(kind: &str) -> expr::Domain {
    match kind {
        _ if kind.starts_with("uint") => expr::Domain::Uint,
        _ if kind.starts_with("int") => expr::Domain::Int,
        _ => expr::Domain::Float,
    }
}

/// The name a register with no field of its own is given, so that every register a shader reads has
/// something to read from.
fn padding(register: u32) -> String {
    format!("pad{register}")
}

impl Names {
    /// Where each component of `cb{slot}[{register}]` sits: the name holding it, and which of that
    /// name's components it is.
    ///
    /// Components are resolved one at a time because a read can straddle two fields packed into one
    /// register, and there is no single name for that.
    fn constant(&self, slot: u16, register: u32, comps: &[u8]) -> Option<Vec<(String, u8)>> {
        self.reaching(slot, None, register, comps)
    }

    /// A register of one element of a buffer whose element is picked at run time, where `element` is
    /// the text of the index. The fields describe a single element, so the register is the one within
    /// it and the subscript goes in front of them.
    pub(crate) fn element(
        &self,
        slot: u16,
        element: &str,
        register: u32,
        comps: &[u8],
    ) -> Option<Vec<(String, u8)>> {
        self.reaching(slot, Some(element), register, comps)
    }

    fn reaching(
        &self,
        slot: u16,
        element: Option<&str>,
        register: u32,
        comps: &[u8],
    ) -> Option<Vec<(String, u8)>> {
        let buffer = self.constants.get(&slot)?;
        // A buffer nothing names the inside of is declared as the array of registers it is, and an
        // array is indexed rather than reached into.
        if buffer.members.is_empty() {
            return None;
        }
        // One element of the buffer, where which element is only known at run time.
        let inside = match element {
            None => buffer.inside(),
            Some(held) => format!("{}[{held}].", buffer.name),
        };
        Some(
            comps
                .iter()
                .map(|comp| {
                    let held = buffer
                        .members
                        .iter()
                        .find(|member| member.covers(register, comp % 4));
                    let Some(member) = held else {
                        return (buffer.gap(register), *comp);
                    };
                    // A member spanning registers is indexed by which of them is being read, the
                    // first one included; an array of matrices twice over, since a register is one
                    // row of one element. A row of a matrix is bracketed, because taking components
                    // straight off one is not a subscript the language allows.
                    let span = register - member.register;
                    let rows = member.rows();
                    let base = match (member.registers > 1, member.elements == 0, rows > 1) {
                        (true, true, _) => {
                            format!("({inside}{}[{span}])", member.name)
                        }
                        (true, false, true) => {
                            format!(
                                "({inside}{}[{}][{}])",
                                member.name,
                                span / rows,
                                span % rows
                            )
                        }
                        (true, false, false) => {
                            format!("{inside}{}[{span}]", member.name)
                        }
                        (false, ..) => format!("{inside}{}", member.name),
                    };
                    // Within a member, the components are counted from where it starts.
                    let first = match member.register == register {
                        true => member.first,
                        false => 0,
                    };
                    (base, comp.saturating_sub(first))
                })
                .collect(),
        )
    }

    /// A texture's name, kept apart from any other bound at a different slot under the same one.
    fn texture(&self, slot: u16) -> String {
        let Some(name) = self.textures.get(&slot) else {
            return format!("t{slot}");
        };
        match self
            .textures
            .iter()
            .any(|(other, alias)| *other < slot && alias == name)
        {
            true => format!("{name}_t{slot}"),
            false => name.clone(),
        }
    }

    /// A sampler's name. A caller may well give a sampler the same name as the texture it goes with, and
    /// two things in one scope cannot share one, so the sampler gives way.
    fn sampler(&self, slot: u16) -> String {
        let Some(name) = self.samplers.get(&slot) else {
            return format!("s{slot}");
        };
        let taken = self.textures.values().any(|texture| texture == name)
            || self
                .samplers
                .iter()
                .any(|(other, alias)| *other < slot && alias == name);
        match taken {
            true => format!("{name}_s{slot}"),
            false => name.clone(),
        }
    }
}

/// A statement, which is either an instruction or a block of them.
enum Stmt {
    Op(usize),
    /// `at` is the `if` itself, which carries the condition and the sense to test it in.
    If {
        at: usize,
        then: Vec<Stmt>,
        els: Vec<Stmt>,
    },
    Loop(Vec<Stmt>),
}

/// Deeper than anything a compiler emits, real shaders reaching about five. The cap only stops a
/// malformed stream of unclosed blocks from recursing away the stack, which on wasm is fatal.
const MAX_DEPTH: usize = 64;

/// Statements up to the marker closing the enclosing block, leaving the cursor on it.
fn block(instructions: &[Instruction], at: &mut usize, depth: usize) -> Vec<Stmt> {
    let mut stmts = Vec::new();
    let opening = |at: &usize| instructions.get(*at).map(|ins| ins.opcode);
    while let Some(opcode) = opening(at) {
        match opcode {
            Opcode::Else | Opcode::EndIf | Opcode::EndLoop => break,
            Opcode::If | Opcode::Loop if depth >= MAX_DEPTH => {
                stmts.push(Stmt::Op(*at));
                *at += 1;
            }
            Opcode::If => {
                let head = *at;
                *at += 1;
                let then = block(instructions, at, depth + 1);
                let mut els = Vec::new();
                if opening(at) == Some(Opcode::Else) {
                    *at += 1;
                    els = block(instructions, at, depth + 1);
                }
                if opening(at) == Some(Opcode::EndIf) {
                    *at += 1;
                }
                stmts.push(Stmt::If {
                    at: head,
                    then,
                    els,
                });
            }
            Opcode::Loop => {
                *at += 1;
                let body = block(instructions, at, depth + 1);
                if opening(at) == Some(Opcode::EndLoop) {
                    *at += 1;
                }
                stmts.push(Stmt::Loop(body));
            }
            _ => {
                stmts.push(Stmt::Op(*at));
                *at += 1;
            }
        }
    }
    stmts
}

/// The instruction stream as nested blocks.
///
/// Every instruction lands in the tree exactly once, including a marker that closes a block nothing
/// opened: it stays a plain statement rather than being dropped, so the reading never silently
/// loses code a stream this walk did not expect.
fn structure(instructions: &[Instruction]) -> Vec<Stmt> {
    let mut at = 0;
    let mut stmts = Vec::new();
    while at < instructions.len() {
        stmts.append(&mut block(instructions, &mut at, 0));
        if at < instructions.len() {
            stmts.push(Stmt::Op(at));
            at += 1;
        }
    }
    stmts
}

/// The HLSL type a resource returns, as its declaration writes it.
fn returns(types: &[ReturnType; 4]) -> &'static str {
    match types[0] {
        ReturnType::Sint => "int4",
        ReturnType::Uint => "uint4",
        ReturnType::Double => "double4",
        _ => "float4",
    }
}

/// The HLSL object type a `dcl_resource` dimension names.
fn object(dimension: &str) -> &'static str {
    match dimension {
        "buffer" => "Buffer",
        "texture1d" => "Texture1D",
        "texture1darray" => "Texture1DArray",
        "texture2darray" => "Texture2DArray",
        "texture2dms" => "Texture2DMS",
        "texture2dmsarray" => "Texture2DMSArray",
        "texture3d" => "Texture3D",
        "texturecube" => "TextureCube",
        "texturecubearray" => "TextureCubeArray",
        _ => "Texture2D",
    }
}

/// The immediate index of an operand, which is what a declaration binds itself to.
fn slot(instruction: &Instruction) -> Option<u32> {
    match instruction.operands().first()?.indices.first()? {
        dxbc::shex::OperandIndex::Imm32(value) => Some(*value),
        _ => None,
    }
}

/// The registers a constant buffer declares, which is its second index.
fn extent(instruction: &Instruction) -> Option<u32> {
    match instruction.operands().first()?.indices.get(1)? {
        dxbc::shex::OperandIndex::Imm32(value) => Some(*value),
        _ => None,
    }
}

/// Constant buffer slots a shader indexes with a computed register.
///
/// Those cannot be a struct of named fields, because nothing names a field a shader picks at run
/// time, so they are declared as the array of registers they really are.
fn computed(program: &Program) -> HashSet<u16> {
    let mut slots = HashSet::new();
    for operand in program.instructions.iter().flat_map(Instruction::operands) {
        if operand.reg_type != dxbc::shex::RegisterType::ConstantBuffer {
            continue;
        }
        if !matches!(
            operand.indices.get(1),
            Some(dxbc::shex::OperandIndex::Imm32(_)) | None
        ) && let Some(dxbc::shex::OperandIndex::Imm32(slot)) = operand.indices.first()
        {
            slots.insert(*slot as u16);
        }
    }
    slots
}

/// The resources a shader binds, as the HLSL that would declare them.
fn preamble(
    program: &Program,
    names: &Names,
    computed: &HashSet<u16>,
    arrays: &HashSet<u16>,
    helpers: &BTreeSet<String>,
    out: &mut Vec<String>,
) {
    for instruction in &program.instructions {
        let Some(at) = slot(instruction) else {
            continue;
        };
        match &instruction.kind {
            InstructionKind::DclConstantBuffer { .. } => {
                let registers = extent(instruction).unwrap_or(1).max(1);
                buffer(names, computed, arrays, at, registers, out);
            }
            InstructionKind::DclResource {
                dimension,
                return_type,
                ..
            } => out.push(format!(
                "{}<{}> {} : register(t{at});",
                object(dimension),
                returns(return_type),
                names.texture(at as u16)
            )),
            // A structured or raw load reads dwords at a byte offset, which is what this buffer
            // does and what a typed one could not.
            InstructionKind::DclResourceStructured { stride, .. } => out.push(format!(
                "ByteAddressBuffer {} : register(t{at});   // {stride}-byte stride",
                names.texture(at as u16)
            )),
            InstructionKind::DclResourceRaw { .. } => out.push(format!(
                "ByteAddressBuffer {} : register(t{at});",
                names.texture(at as u16)
            )),
            InstructionKind::DclSampler { mode, .. } => {
                let kind = match *mode {
                    "comparison" => "SamplerComparisonState",
                    _ => "SamplerState",
                };
                out.push(format!(
                    "{kind} {} : register(s{at});",
                    names.sampler(at as u16)
                ));
            }
            _ => {}
        }
    }

    if let Some(values) = program.instructions.iter().find_map(|ins| match &ins.kind {
        InstructionKind::CustomData { values, .. } if !values.is_empty() => Some(values),
        _ => None,
    }) {
        out.push(String::new());
        out.push(format!("static const float4 icb[{}] =", values.len()));
        out.push("{".to_owned());
        for value in values {
            let lanes: Vec<String> = value.iter().map(|lane| format!("{lane:?}")).collect();
            out.push(format!("    float4({}),", lanes.join(", ")));
        }
        out.push("};".to_owned());
    }

    if !helpers.is_empty() {
        out.push(String::new());
        for helper in helpers {
            out.push(helper.clone());
        }
    }
}

/// One constant buffer, as the struct the shader declared it with.
///
/// Registers no field accounts for get one of their own, so that every register the body reads has
/// somewhere to read it from even where the reflection left a gap.
fn buffer(
    names: &Names,
    computed: &HashSet<u16>,
    arrays: &HashSet<u16>,
    at: u32,
    registers: u32,
    out: &mut Vec<String>,
) {
    let held = names.constants.get(&(at as u16));
    let name = held.map_or_else(|| format!("cb{at}"), |buffer| buffer.name.clone());

    // A buffer picked apart by element is declared as the array of structs it is, even though the
    // element is only known at run time.
    let named = held
        .filter(|buffer| !buffer.members.is_empty())
        .filter(|_| !computed.contains(&(at as u16)) || arrays.contains(&(at as u16)));
    let Some(buffer) = named else {
        // A buffer whose element is picked at run time is an array of registers, since a computed
        // register cannot reach a field by name. Where the fields are known they still say what each
        // register of an element holds, which is the only way back from an index like `i * 6 + 5`.
        if let Some(held) = held.filter(|held| !held.members.is_empty()) {
            let stride = held.span();
            if stride > 0 && registers > stride && registers.is_multiple_of(stride) {
                out.push(format!(
                    "// {} of {stride} registers each, indexed as `element * {stride} + register`:",
                    match registers / stride {
                        1 => "1 element".to_owned(),
                        count => format!("{count} elements"),
                    }
                ));
                out.extend(held.members.iter().map(Member::placed));
            }
        }
        out.push(format!("cbuffer cb{at} : register(b{at})"));
        out.push("{".to_owned());
        // One register is not an array of anything.
        out.push(match registers {
            1 => format!("    float4 {name};"),
            held => format!("    float4 {name}[{held}];"),
        });
        out.push("};".to_owned());
        return;
    };

    let mut members: Vec<(u32, u8, String)> = buffer
        .members
        .iter()
        .map(|member| (member.register, member.first, member.declaration()))
        .collect();
    // An array of structs describes one element, so what it has to cover is an element rather than
    // the whole buffer, and there is one of them per element.
    let elements = match arrays.contains(&(at as u16)) {
        true => registers / buffer.span().max(1),
        false => 1,
    };
    // The body can read past everything the fields describe, and a register it reads still has to
    // have something declared at it.
    for register in buffer.span()..registers / elements {
        members.push((
            register,
            0,
            format!(
                "    float4 {};",
                buffer.gap(register).rsplit('.').next().unwrap_or_default()
            ),
        ));
    }
    members.sort_by_key(|(register, first, _)| (*register, *first));

    // One member named after the buffer needs no struct around it, which is how a hand-written
    // header has these. Anything with more than that keeps the struct, so a field name cannot
    // collide with another buffer's at the outer scope.
    out.push(match buffer.alone() {
        true => format!("cbuffer {name} : register(b{at})"),
        false => format!("struct {name}_t"),
    });
    out.push("{".to_owned());
    out.extend(members.into_iter().map(|(_, _, text)| text));
    out.push("};".to_owned());
    if buffer.alone() {
        return;
    }
    out.push(format!("cbuffer cb{at} : register(b{at})"));
    out.push("{".to_owned());
    let held = match elements {
        1 => String::new(),
        count => format!("[{count}]"),
    };
    out.push(format!("    {name}_t {name}{held};"));
    out.push("};".to_owned());
}

/// A shader as HLSL, and where its declarations stop.
pub struct Decompiled {
    /// The source, one entry per line, with no trailing newlines.
    pub lines: Vec<String>,
    /// The first line of the entry point. Everything before it declares what the shader binds, which
    /// is most of the text and none of the shading.
    pub body: usize,
}

/// How much of the shader to put back together.
///
/// The difference is worth having beyond curiosity: a plain reading is a transliteration nothing has
/// rearranged, so comparing the two says whether the rearranging changed what the shader computes.
/// It is also what to fall back to when a folded reading is under suspicion.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reading {
    /// Every value moved to its reader where it can be, registers renamed where one holds a second
    /// value, and runs of lines put back together into the operations they came from.
    Folded,
    /// Folded, but recognising nothing: no operation recovered from the run of instructions that
    /// implemented it, and no transform rebuilt from a sum of rows. Values still move to their
    /// readers and registers are still named apart, so the reading is followable, but every
    /// expression is the arithmetic the machine actually does.
    Exact,
    /// One statement per instruction, every value written down under the register it went to.
    Plain,
}

/// Decompile a program to HLSL, folded as far as it will go. [`read`] with [`Reading::Folded`].
pub fn decompile(program: &Program, names: &Names) -> Decompiled {
    read(program, names, Reading::Folded)
}

/// Decompile a program to HLSL at the given [`Reading`].
pub fn read(program: &Program, names: &Names, reading: Reading) -> Decompiled {
    let tree = structure(&program.instructions);
    let computed = computed(program);
    let mut builder = body::Builder::new(program, names, &computed, reading);
    builder.run(&tree, 1);

    let mut out = vec![format!(
        "// {}_{}_{}",
        program.shader_type, program.major_version, program.minor_version
    )];
    if let Some(flags) = program.instructions.iter().find_map(|ins| match &ins.kind {
        InstructionKind::DclGlobalFlags { flags } if !flags.is_empty() => Some(flags),
        _ => None,
    }) {
        out.push(format!("// {}", flags.join(", ")));
    }
    out.push(String::new());
    preamble(
        program,
        names,
        &computed,
        &builder.arrays,
        &builder.helpers,
        &mut out,
    );
    out.push(String::new());

    // Declaration order is what assigns an output its register, so the order has to be the one the
    // signature gives rather than any reading of the names.
    let mut written: Vec<(&u32, &Semantic)> = names.outputs.iter().collect();
    written.sort_by_key(|(register, _)| **register);
    if !written.is_empty() {
        out.push("struct Output".to_owned());
        out.push("{".to_owned());
        for (_, entry) in &written {
            out.push(format!(
                "    {} {} : {};",
                entry.kind, entry.name, entry.name
            ));
        }
        out.push("};".to_owned());
        out.push(String::new());
    }

    let mut taken: Vec<(&u32, &Semantic)> = names.inputs.iter().collect();
    taken.sort_by_key(|(register, _)| **register);
    let parameters: Vec<String> = taken
        .iter()
        .map(|(_, entry)| format!("{} {} : {}", entry.kind, entry.name, entry.name))
        .collect();
    let returns = match written.is_empty() {
        true => "void",
        false => "Output",
    };
    let body = out.len();
    match parameters.is_empty() {
        true => out.push(format!("{returns} main()")),
        false => {
            out.push(format!("{returns} main("));
            for (at, parameter) in parameters.iter().enumerate() {
                let comma = match at + 1 == parameters.len() {
                    true => "",
                    false => ",",
                };
                out.push(format!("    {parameter}{comma}"));
            }
            out.push(")".to_owned());
        }
    }
    out.push("{".to_owned());
    if !written.is_empty() {
        out.push("    Output output = (Output)0;".to_owned());
    }

    // The register a value lands in is whichever one the compiler had free, so the numbers run
    // sparse and high — a merged pass reaches r1713 for five hundred registers. Nothing depends on
    // them but the reading, so they are renumbered in the order they first appear.
    renumber(&mut builder.out, &mut builder.renamed);
    // A register that only ever carries bits between an `asfloat` and an `asuint` is a float in
    // name alone.
    let carried = uncast(&mut builder.out);

    let temps = program
        .instructions
        .iter()
        .find_map(|ins| match ins.kind {
            InstructionKind::DclTemps { count } => Some(count),
            _ => None,
        })
        .unwrap_or(0);
    if temps > 0 {
        // The shader declares as many temps as the compiler allocated, but most of them end up
        // folded into what reads them and never appear. Declaring those too costs nothing to the
        // compiler and everything to whoever is reading: a merged pass allocates a temp per value
        // and would open with a line naming eighteen hundred registers, five hundred of which the
        // body mentions.
        let registers: Vec<String> = (0..temps)
            .map(|index| format!("r{index}"))
            .filter(|name| builder.out.iter().any(|line| mentions(line, name)))
            .collect();
        let widths = widths(&builder.out);
        // A register that is only ever one component does not need naming one.
        narrow(&mut builder.out, &widths);
        // A register the body only ever reads one component of does not need four, and saying so is
        // the difference between a wall of `float4` and a declaration that describes the shader.
        let mut banded: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for name in registers {
            // Carrying bits changes what a register is declared as, never how wide it is.
            let base = carried.get(&name).copied().unwrap_or("float");
            let kind = match widths.get(&name).copied().unwrap_or(4) {
                1 => base.to_owned(),
                held => format!("{base}{held}"),
            };
            banded.entry(kind).or_default().push(name);
        }
        banded
            .entry("float4".to_owned())
            .or_default()
            .append(&mut builder.renamed);
        for (kind, held) in &banded {
            // Five hundred of them on one line is not a declaration anybody reads; wrapped, it is a
            // block that can be skipped over.
            for (row, names) in held.chunks(WRAP).enumerate() {
                let opening = match row {
                    0 => format!("    {kind} "),
                    _ => "        ".to_owned(),
                };
                let closing = match (row + 1) * WRAP < held.len() {
                    true => ",",
                    false => ";",
                };
                out.push(format!("{opening}{}{closing}", names.join(", ")));
            }
        }
    }
    for instruction in &program.instructions {
        let InstructionKind::DclIndexableTemp {
            reg,
            size,
            components,
        } = instruction.kind
        else {
            continue;
        };
        out.push(format!("    float{components} x{reg}[{size}];"));
    }
    out.push(String::new());
    out.append(&mut builder.out);
    out.push("}".to_owned());
    Decompiled { lines: out, body }
}

/// Registers that only carry bits, rewritten to the type those bits are, and named with it.
///
/// The machine has one register file and it is untyped, so an integer travelling in one is written
/// through `asfloat` and read back through `asuint`. Where *every* write is that cast and *every*
/// read undoes it with the same one, the casts say nothing the declaration cannot, and both come off.
///
/// Bit-preserving in both directions, which is what makes it safe: the declared type changes to the
/// one the bits already were.
fn uncast(body: &mut [String]) -> HashMap<String, &'static str> {
    let mut kinds: HashMap<String, &'static str> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    for line in body.iter() {
        if let Some((held, _)) = line.trim_start().split_once(" = ")
            && held.starts_with('r')
            && held[1..].chars().all(|held| held.is_ascii_digit())
        {
            names.push(held.to_owned());
        }
    }
    names.sort_unstable();
    names.dedup();

    for name in names {
        let write = format!("{name} = asfloat(");
        let (mut writes, mut kind) = (0usize, None);
        let mut usable = true;
        for line in body.iter() {
            let held = line.trim_start();
            if held.starts_with(&write) && held.ends_with(");") {
                writes += 1;
                continue;
            }
            // Every other mention has to be one of the two casts undoing it.
            let mut from = 0;
            while let Some(at) = line[from..].find(&name) {
                let at = from + at;
                let before = line[..at].chars().next_back();
                let after = line[at + name.len()..].chars().next();
                from = at + name.len();
                if before.is_some_and(|held| held.is_ascii_alphanumeric() || held == '_')
                    || after.is_some_and(|held| held.is_ascii_alphanumeric() || held == '_')
                {
                    continue;
                }
                let wrapped = ["asuint(", "asint("]
                    .into_iter()
                    .find(|cast| line[..at].ends_with(cast) && after == Some(')'));
                match wrapped {
                    Some(cast) if kind.is_none_or(|held| held == cast) => kind = Some(cast),
                    _ => {
                        usable = false;
                        break;
                    }
                }
            }
            if !usable {
                break;
            }
        }
        let Some(cast) = kind.filter(|_| usable && writes > 0) else {
            continue;
        };
        kinds.insert(
            name.clone(),
            match cast {
                "asuint(" => "uint",
                _ => "int",
            },
        );
        for line in body.iter_mut() {
            let held = line.trim_start();
            if held.starts_with(&write) && held.ends_with(");") {
                let inner = &held[write.len()..held.len() - 2];
                *line = format!("    {name} = {inner};");
                continue;
            }
            *line = line.replace(&format!("{cast}{name})"), &name);
        }
    }
    kinds
}

/// Registers to a line where they are declared together.
const WRAP: usize = 12;

/// `.x` taken off the registers that hold nothing else, since a `float` has no other component.
///
/// Only a bare `.x` goes: `r1.xx` is a broadcast into two components and still means something on a
/// scalar.
fn narrow(body: &mut [String], widths: &HashMap<String, usize>) {
    for line in body.iter_mut() {
        for (name, _) in widths.iter().filter(|(_, wide)| **wide == 1) {
            let mut from = 0;
            let wanted = format!("{name}.x");
            while let Some(at) = line[from..].find(&wanted) {
                let at = from + at;
                let before = line[..at].chars().next_back();
                let after = line[at + wanted.len()..].chars().next();
                let bare = !before.is_some_and(|held| held.is_ascii_alphanumeric() || held == '_')
                    && !after.is_some_and(|held| matches!(held, 'x' | 'y' | 'z' | 'w'));
                match bare {
                    true => {
                        line.replace_range(at..at + wanted.len(), name);
                        from = at + name.len();
                    }
                    false => from = at + wanted.len(),
                }
            }
        }
    }
}

/// How wide each register has to be declared: one past the highest component the body names.
///
/// Components are not remapped, so a register read at `.z` stays three wide even where nothing
/// touches `.y`; the alternative is rewriting every swizzle that mentions it. A register used with
/// no swizzle at all is whatever it was, which is four.
fn widths(body: &[String]) -> HashMap<String, usize> {
    let mut widths: HashMap<String, usize> = HashMap::new();
    for line in body {
        let held: Vec<char> = line.chars().collect();
        let mut at = 0;
        while at < held.len() {
            let starts = held[at] == 'r'
                && at + 1 < held.len()
                && held[at + 1].is_ascii_digit()
                && (at == 0 || !(held[at - 1].is_ascii_alphanumeric() || held[at - 1] == '_'));
            if !starts {
                at += 1;
                continue;
            }
            let mut end = at + 1;
            while end < held.len() && held[end].is_ascii_digit() {
                end += 1;
            }
            if held
                .get(end)
                .is_some_and(|held| held.is_alphabetic() || *held == '_')
            {
                at = end;
                continue;
            }
            let name: String = held[at..end].iter().collect();
            let mut wide = 4;
            if held.get(end) == Some(&'.') {
                let lanes: String = held[end + 1..]
                    .iter()
                    .take_while(|held| matches!(held, 'x' | 'y' | 'z' | 'w'))
                    .collect();
                if !lanes.is_empty() {
                    wide = lanes
                        .chars()
                        .map(|held| "xyzw".find(held).unwrap_or(3) + 1)
                        .max()
                        .unwrap_or(4);
                }
            }
            let into = widths.entry(name).or_insert(1);
            *into = (*into).max(wide);
            at = end;
        }
    }
    widths
}

/// Registers renumbered densely, in the order the body first mentions them.
///
/// A name is `r` and digits, and the typed copies of one lane carry it as a prefix (`r184_x`), so
/// both move together.
fn renumber(body: &mut [String], renamed: &mut [String]) {
    let mut dense: HashMap<String, String> = HashMap::new();
    let mut rewrite = |line: &mut String| {
        let mut out = String::with_capacity(line.len());
        let held: Vec<char> = line.chars().collect();
        let mut at = 0;
        while at < held.len() {
            let starts = held[at] == 'r'
                && at + 1 < held.len()
                && held[at + 1].is_ascii_digit()
                && (at == 0 || !(held[at - 1].is_ascii_alphanumeric() || held[at - 1] == '_'));
            if !starts {
                out.push(held[at]);
                at += 1;
                continue;
            }
            let mut end = at + 1;
            while end < held.len() && held[end].is_ascii_digit() {
                end += 1;
            }
            // `r1a` is somebody else's name; only digits then a break is a register.
            match held.get(end).is_some_and(|held| held.is_ascii_alphabetic()) {
                true => {
                    out.extend(&held[at..end]);
                }
                false => {
                    let name: String = held[at..end].iter().collect();
                    let next = dense.len();
                    out.push_str(
                        dense
                            .entry(name)
                            .or_insert_with(|| format!("r{next}"))
                            .as_str(),
                    );
                }
            }
            at = end;
        }
        *line = out;
    };
    for line in body.iter_mut() {
        rewrite(line);
    }
    for name in renamed.iter_mut() {
        rewrite(name);
    }
}

/// Whether a line names this register, as opposed to one whose name it merely begins.
fn mentions(line: &str, name: &str) -> bool {
    let ident = |held: char| held.is_ascii_alphanumeric() || held == '_';
    let mut from = 0;
    while let Some(at) = line[from..].find(name) {
        let at = from + at;
        let before = line[..at].chars().next_back();
        let after = line[at + name.len()..].chars().next();
        if !before.is_some_and(ident) && !after.is_some_and(ident) {
            return true;
        }
        from = at + name.len();
    }
    false
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn a_register_is_not_named_by_one_it_prefixes() {
        assert!(mentions("    r1.x = r20.y;", "r1"));
        assert!(!mentions("    r20.x = r201.y;", "r2"));
        assert!(!mentions("    uint r18_x = 0;", "r18"));
        assert!(mentions("    r7 = float4(0.0, 0.0, 0.0, 0.0);", "r7"));
    }

    fn field(name: &str, register: u32, registers: u32, mask: u8) -> Field {
        Field {
            name: name.to_owned(),
            // Sized from the mask the way both of the real builders size it, since a type wider than
            // the components it claims is what the tiling exists to keep out.
            kind: match (registers > 1, mask.count_ones()) {
                (true, _) | (_, 4) => "float4".to_owned(),
                (_, 1) => "float".to_owned(),
                (_, lanes) => format!("float{lanes}"),
            },
            register,
            registers,
            mask,
        }
    }

    fn named(fields: Vec<Field>) -> Names {
        let mut names = Names::default();
        names
            .constants
            .insert(0, Buffer::new("g_Buffer".to_owned(), fields));
        names
    }

    /// The declaration falls back to an array of registers when nothing names the inside, so the
    /// reading has to index rather than reach in, or it names a member that was never declared.
    #[test]
    fn a_buffer_without_fields_is_indexed() {
        assert!(named(Vec::new()).constant(0, 3, &[0]).is_none());
    }

    #[test]
    fn a_field_wider_than_a_register_is_indexed_from_its_first() {
        let names = named(vec![field("m_View", 0, 4, 0xF)]);
        let parts = names.constant(0, 2, &[0, 1]).unwrap();
        assert_eq!(parts[0].0, "g_Buffer.m_View[2]");
        assert_eq!(parts[1], ("g_Buffer.m_View[2]".to_owned(), 1));
    }

    /// Four parameters can share one register, so which field answers depends on the component.
    #[test]
    fn packed_fields_are_told_apart_by_component() {
        let names = named(vec![
            field("m_Near", 0, 1, 0b0011),
            field("m_Far", 0, 1, 0b1100),
        ]);
        let parts = names.constant(0, 0, &[0, 3]).unwrap();
        assert_eq!(parts[0], ("g_Buffer.m_Near".to_owned(), 0));
        assert_eq!(parts[1], ("g_Buffer.m_Far".to_owned(), 1));
    }

    #[test]
    fn a_register_no_field_covers_falls_to_its_own_name() {
        let names = named(vec![field("m_Near", 0, 1, 0xF)]);
        assert_eq!(
            names.constant(0, 5, &[2]).unwrap()[0],
            ("g_Buffer.pad5".to_owned(), 2)
        );
    }

    /// Packing puts a member after the one before it, so a field keeps the register the bytecode
    /// reads it from only when what it is declared after fills everything ahead of it.
    #[test]
    fn what_the_fields_leave_between_them_is_declared_too() {
        let buffer = Buffer::new(
            "g_Buffer".to_owned(),
            vec![field("m_Near", 0, 1, 0b0001), field("m_Far", 2, 1, 0b0011)],
        );
        let declared: Vec<String> = buffer
            .members
            .iter()
            .map(|member| format!("{} {}", member.kind, member.name))
            .collect();
        assert_eq!(
            declared,
            [
                "float m_Near",
                "float3 pad0_1",
                "float4 pad1",
                "float2 m_Far",
                "float2 pad2_2"
            ]
        );
    }

    /// A field the reflection sizes by where the next one starts claims the padding after it, which
    /// its own type does not fill.
    #[test]
    fn a_field_narrower_than_the_room_it_claims_is_padded_to_it() {
        let mut colour = field("m_Colour", 0, 1, 0xF);
        colour.kind = "float3".to_owned();
        let names = named(vec![colour]);
        assert_eq!(
            names.constant(0, 0, &[3]).unwrap()[0],
            ("g_Buffer.pad0_3".to_owned(), 0)
        );
    }

    /// A caller may give a sampler the same name as its texture, and two things in one scope cannot
    /// share one.
    #[test]
    fn a_sampler_gives_way_to_the_texture_it_shares_a_name_with() {
        let mut names = Names::default();
        names.textures.insert(6, "g_SamplerNormal".to_owned());
        names.samplers.insert(2, "g_SamplerNormal".to_owned());
        assert_eq!(names.texture(6), "g_SamplerNormal");
        assert_eq!(names.sampler(2), "g_SamplerNormal_s2");
    }

    fn program(opcodes: &[Opcode]) -> Program {
        Program {
            shader_type: "ps",
            major_version: 5,
            minor_version: 0,
            fourcc: *b"SHEX",
            warnings: Vec::new(),
            instructions: opcodes
                .iter()
                .map(|opcode| Instruction {
                    opcode: *opcode,
                    saturate: false,
                    test_nonzero: true,
                    precise_mask: 0,
                    resinfo_return_type: None,
                    sync_flags: 0,
                    tex_offsets: None,
                    resource_dim: None,
                    resource_return_type: None,
                    kind: InstructionKind::Generic {
                        operands: Default::default(),
                    },
                })
                .collect(),
        }
    }

    /// Every instruction has to land in the tree exactly once, which is what makes the reading a
    /// rearrangement of the shader rather than a summary of it.
    fn count(tree: &[Stmt]) -> usize {
        tree.iter()
            .map(|stmt| match stmt {
                Stmt::Op(_) => 1,
                Stmt::If { then, els, .. } => 1 + count(then) + count(els),
                Stmt::Loop(inner) => 1 + count(inner),
            })
            .sum()
    }

    #[test]
    fn nests_matched_blocks() {
        let opcodes = [
            Opcode::Mov,
            Opcode::If,
            Opcode::Mul,
            Opcode::Else,
            Opcode::Add,
            Opcode::EndIf,
            Opcode::Ret,
        ];
        let tree = structure(&program(&opcodes).instructions);
        assert_eq!(tree.len(), 3);
        let Stmt::If { then, els, .. } = &tree[1] else {
            panic!("the if did not nest");
        };
        assert_eq!(then.len(), 1);
        assert_eq!(els.len(), 1);
    }

    #[test]
    fn a_block_nothing_closes_keeps_its_body() {
        let opcodes = [Opcode::If, Opcode::Mul, Opcode::Ret];
        let tree = structure(&program(&opcodes).instructions);
        assert_eq!(count(&tree), opcodes.len());
        assert!(matches!(tree.first(), Some(Stmt::If { .. })));
    }

    #[test]
    fn a_marker_nothing_opened_is_kept() {
        let opcodes = [Opcode::Mul, Opcode::EndIf, Opcode::Ret];
        let tree = structure(&program(&opcodes).instructions);
        assert_eq!(count(&tree), 3);
    }

    #[test]
    fn nesting_past_the_cap_stays_flat() {
        let mut opcodes = vec![Opcode::If; MAX_DEPTH + 4];
        opcodes.push(Opcode::Ret);
        let instructions = program(&opcodes).instructions;
        let tree = structure(&instructions);
        assert_eq!(count(&tree), instructions.len());
    }

    /// A buffer holding one member named after itself is declared straight into the `cbuffer`, the
    /// way a hand-written header has it, so nothing reaches through a name spelled twice.
    #[test]
    fn a_buffer_that_is_one_matrix_needs_no_struct() {
        let mut only = field("g_WorldMatrix", 0, 3, 0xF);
        only.kind = "float3x4".to_owned();
        let mut names = Names::default();
        names
            .constants
            .insert(0, Buffer::new("g_WorldMatrix".to_owned(), vec![only]));
        let mut out = Vec::new();
        buffer(&names, &HashSet::new(), &HashSet::new(), 0, 3, &mut out);
        assert_eq!(out[0], "cbuffer g_WorldMatrix : register(b0)");
        assert_eq!(out[2], "    row_major float3x4 g_WorldMatrix;");
        // And a read of one of its rows names the matrix once.
        assert_eq!(
            names.constant(0, 1, &[0]).unwrap()[0],
            ("(g_WorldMatrix[1])".to_owned(), 0)
        );
    }

    /// More than one member keeps the struct, so a field cannot collide with another buffer's.
    #[test]
    fn a_buffer_with_more_than_one_member_keeps_its_struct() {
        let names = named(vec![field("m_Near", 0, 1, 0xF), field("m_Far", 1, 1, 0xF)]);
        let mut out = Vec::new();
        buffer(&names, &HashSet::new(), &HashSet::new(), 0, 2, &mut out);
        assert_eq!(out[0], "struct g_Buffer_t");
        assert_eq!(
            names.constant(0, 0, &[0]).unwrap()[0],
            ("g_Buffer.m_Near".to_owned(), 0)
        );
    }

    /// An array of matrices is indexed twice: a register is one row of one element, and the whole of
    /// an element is the matrix a transform multiplies by.
    #[test]
    fn an_array_of_matrices_is_indexed_by_element_and_row() {
        let mut only = field("g_WorldViewMatrix", 0, 6, 0xF);
        only.kind = "float3x4[2]".to_owned();
        let mut names = Names::default();
        names
            .constants
            .insert(0, Buffer::new("g_WorldViewMatrix".to_owned(), vec![only]));
        let mut out = Vec::new();
        buffer(&names, &HashSet::new(), &HashSet::new(), 0, 6, &mut out);
        assert_eq!(out[2], "    row_major float3x4 g_WorldViewMatrix[2];");
        // Register 0 is the first row of the first, register 4 the second row of the second.
        assert_eq!(
            names.constant(0, 0, &[0]).unwrap()[0],
            ("(g_WorldViewMatrix[0][0])".to_owned(), 0)
        );
        assert_eq!(
            names.constant(0, 4, &[2]).unwrap()[0],
            ("(g_WorldViewMatrix[1][1])".to_owned(), 2)
        );
        // And the array spends a register on every row of every element, so nothing is padded past.
        assert_eq!(out.len(), 4);
    }

    /// An array of anything else counts elements rather than registers, so it keeps the registers it
    /// really is and stays indexed by them.
    #[test]
    fn an_array_of_vectors_stays_indexed_by_register() {
        let mut only = field("g_Data", 0, 3, 0xF);
        only.kind = "float4[3]".to_owned();
        let names = named(vec![only]);
        assert_eq!(
            names.constant(0, 2, &[1]).unwrap()[0],
            ("g_Buffer.g_Data[2]".to_owned(), 1)
        );
    }

    /// A buffer whose element is picked at run time is still reachable by name, since the fields
    /// describe one element and the subscript goes in front of them.
    #[test]
    fn an_element_picked_at_run_time_still_reaches_a_field() {
        let mut matrix = field("m_TransformMatrix", 0, 3, 0xF);
        matrix.kind = "float3x4".to_owned();
        let names = named(vec![matrix, field("m_Colour", 5, 1, 0b0111)]);
        // Within the element the register is the one the field sits at, whatever element it is.
        assert_eq!(
            names.element(0, "held", 5, &[0]).unwrap()[0],
            ("g_Buffer[held].m_Colour".to_owned(), 0)
        );
        assert_eq!(
            names.element(0, "held", 1, &[0]).unwrap()[0],
            ("(g_Buffer[held].m_TransformMatrix[1])".to_owned(), 0)
        );
        // A buffer read as one struct spells no subscript at all.
        assert_eq!(
            names.constant(0, 5, &[0]).unwrap()[0],
            ("g_Buffer.m_Colour".to_owned(), 0)
        );
    }

    /// One element is not an array, so a buffer that holds a single struct keeps its plain instance.
    #[test]
    fn a_buffer_of_one_element_is_not_an_array() {
        let names = named(vec![field("m_Near", 0, 1, 0xF), field("m_Far", 1, 1, 0xF)]);
        let mut out = Vec::new();
        let mut arrays = HashSet::new();
        arrays.insert(0);
        buffer(&names, &HashSet::new(), &arrays, 0, 2, &mut out);
        assert!(out.iter().any(|line| line == "    g_Buffer_t g_Buffer;"));
    }
}
