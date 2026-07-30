//! Expression recovery — the register traffic of a shader folded back into statements.
//!
//! Every instruction writes a register, so a line of the original source comes back as a run of
//! them. This walks the instruction stream keeping, for each component of each register, the value
//! that last landed there, and folds that value into whatever reads it next. A value read once, in
//! the same block, with nothing disturbing what it was computed from, disappears into its reader; a
//! value read twice, read from another block, or read after something it depends on was overwritten
//! is written down under the register's own name.
//!
//! The decision needs to know how a value gets used, which is only clear later, so the walk runs
//! more than once over the same bookkeeping, and no run can decide something an earlier one did not
//! see. It runs more than twice because the decisions feed each other: what a value depends on is
//! what its text names, and its text names a register wherever the value behind it stayed in one,
//! so which values move decides which are free to move. That is gone round until a reading comes
//! back describing itself. What the walks see also settles where a register stops carrying one
//! value and starts carrying the next, so each of those can take a name rather than sharing one.

use std::collections::{HashMap, HashSet};

use dxbc::shex::{
    ComponentSelect, Instruction, InstructionKind, Opcode, Operand, OperandIndex, Program,
    RegisterType,
};

use super::Names;
use super::expr::{Domain, Expr, call, coerce, letters};

/// One component of one register, keyed by the raw register file so the tuple can be hashed.
type Slot = (u32, u32, u8);

/// Which value owns each component of a register, and which of its lanes.
type Cells = [Option<(usize, u8)>; 4];

/// A read of a register component, and how many values had been made when it happened.
type Touch = ((u32, u32), u8, usize);

/// How many times to go round looking for a reading that describes itself. Two is the common answer
/// and three is rare; past that it is oscillating rather than settling.
const ROUNDS: usize = 4;

/// A value an instruction produced.
struct Def {
    /// Where it went, spelled the way a reader of the output would write it.
    base: String,
    /// The register it landed in, which several values in a row may share.
    key: (u32, u32),
    /// The destination components it covers, ascending.
    lanes: Vec<u8>,
    expr: Expr,
    domain: Domain,
    /// Register components the expression reads, through anything folded into it. A write to one of
    /// these before the value is used means it can no longer be moved to the reader.
    reads: Vec<Slot>,
    /// The block it was produced in. Folding across one would move work into or out of a branch.
    block: usize,
    /// Set where the destination is one the output has to name: a shader output, or an array the
    /// walk does not track.
    fixed: bool,
    /// Set where the value went to a name of its own rather than into a register, which is how a
    /// value that is not a float avoids being reinterpreted on the way in and out again.
    local: bool,
    /// Set where the value is produced inside a loop. The register is what carries a value from one
    /// turn to the next, so such a value cannot be given a name of its own.
    looped: bool,
}

#[derive(Default, Clone)]
struct Usage {
    reads: usize,
    /// A read that took this value together with another one, which neither can be folded into.
    mixed: bool,
    /// A read from a block other than the one that produced it.
    distant: bool,
    /// Something the expression depends on was overwritten before it was read.
    stale: bool,
    /// The last value to read this one, which is how far it has to last.
    last: usize,
    /// How many of the reads took this as a register index into a buffer of structs, and the size of
    /// an element where every one of them agreed on it.
    indexed: usize,
    stride: Option<u32>,
}

/// A value read out of the machine, and what reading it depended on.
struct Sourced {
    expr: Expr,
    domain: Domain,
    reads: Vec<Slot>,
    /// The value read, where the read came down to a single one. What a value is used *for* is only
    /// known where it is read, and one used only to index a buffer of structs can be the element
    /// rather than the register.
    owner: Option<usize>,
}

/// The walk over a shader's instructions, and everything it remembers while folding them.
pub struct Builder<'a> {
    instructions: &'a [Instruction],
    names: &'a Names,
    cells: HashMap<(u32, u32), Cells>,
    defs: Vec<Def>,
    usage: Vec<Usage>,
    /// Values reading each register component that nothing has read yet.
    waiting: HashMap<Slot, Vec<usize>>,
    /// Every register component read, and when, so that the point a register stops carrying one
    /// value and starts carrying the next can be found.
    touches: Vec<Touch>,
    /// Empty on the counting run, which writes every value down.
    folded: Vec<bool>,
    /// Which values get a name of their own instead of a register.
    localised: Vec<bool>,
    /// Which values start their register over under a name of its own.
    splits: Vec<bool>,
    reading: super::Reading,
    /// Set while nothing is known about what moves, so every value is taken to depend on everything
    /// behind it.
    assume: bool,
    /// What each register is called from here on, which changes where one value in it is done with.
    versions: HashMap<(u32, u32), usize>,
    /// The names those changes took, in the order the body introduces them.
    pub renamed: Vec<String>,
    /// How many loops the walk is inside.
    loops: usize,
    /// Names already handed out, since two values in one register cannot share one.
    issued: HashMap<String, usize>,
    /// How many coordinate components each texture slot takes.
    dimensions: HashMap<u32, usize>,
    /// What each texture slot is, for the helper that reports its size.
    kinds: HashMap<u32, &'static str>,
    /// The element size of each buffer slot, which turns an element index into a byte address.
    strides: HashMap<u32, u32>,
    /// Registers each constant buffer declares, which says whether it holds one struct or many.
    spans: HashMap<u16, u32>,
    /// Whether every run-time read of a buffer looked like an element of it and a register within
    /// that element. One read that did not is enough to leave the whole buffer as registers.
    reachable: HashMap<u16, bool>,
    /// The values used to pick an element of each buffer, so that one used for anything else can stop
    /// the whole buffer being read that way.
    indexers: HashMap<u16, Vec<usize>>,
    /// The element size of each value that holds an element rather than the register it starts at,
    /// nought for every other value. Decided once, so it does not depend on the walk in hand.
    elemental: Vec<u32>,
    /// Buffers being read as arrays of structs rather than as arrays of registers.
    pub arrays: HashSet<u16>,
    /// Constant buffers declared as an array of registers rather than as named fields.
    computed: &'a HashSet<u16>,
    /// Whether the shader hands a value back, which decides what a return says.
    returns: bool,
    /// Functions the body needs that HLSL does not provide.
    pub helpers: std::collections::BTreeSet<String>,
    /// What each emitted line assigns, where it assigns anything, so that a run of them can be
    /// recognised as one operation afterwards.
    emitted: Vec<Option<super::matrix::Emitted>>,
    block: usize,
    blocks: usize,
    /// The statements written so far, one per line and already indented.
    pub out: Vec<String>,
}

/// A condition and the constant it picks, where an `and` is a mask against one.
///
/// The constant is only worth reading as a float where it plainly is one. An integer kept in a
/// register is a handful of bits, which as a float is a denormal nobody wrote on purpose, and
/// saying `1e-44` in place of `31` helps no one.
fn chosen(sources: &[(Expr, Domain)]) -> Option<(Expr, Expr)> {
    let [left, right] = sources else {
        return None;
    };
    // A float nobody wrote on purpose: an integer kept in a register is a handful of bits, which
    // read as a float is a denormal, and saying `1e-44` in place of `31` helps no one.
    let plain = |bits: &u32| {
        let value = f32::from_bits(*bits);
        value == 0.0 || (value.is_finite() && value.abs() >= f32::MIN_POSITIVE)
    };
    for ((cond, domain), (held, _)) in [(left, right), (right, left)] {
        let Expr::Literal { bits, .. } = held else {
            continue;
        };
        if *domain != Domain::Bool || !bits.iter().all(plain) {
            continue;
        }
        let held = Expr::Literal {
            bits: bits.clone(),
            domain: Domain::Float,
        };
        return Some((cond.clone(), held));
    }
    None
}

/// Whether everything a value reads is still what it was by the time the last reader has had it.
///
/// A value with a single reader only has to last until that one, which the staleness sweep already
/// watches. One written out again at each of its readers has to last until the last of them, and a
/// write anywhere in between ends it.
fn survives(defs: &[Def], def: &Def, from: usize, to: usize) -> bool {
    let arrays = RegisterType::IndexableTemp.to_u32();
    let over = to.max(from + 1).min(defs.len());
    !defs[from + 1..over]
        .iter()
        .any(|held| match held.key.0 == arrays {
            // Which element of an array a read took is not known, so a write to any of it ends them all.
            true => def.reads.iter().any(|(kind, ..)| *kind == arrays),
            false => held
                .lanes
                .iter()
                .any(|lane| def.reads.contains(&(held.key.0, held.key.1, lane % 4))),
        })
}

/// A product with the given constant factor taken off, which is what the value was before the
/// machine turned an element into the register it starts at.
fn without_stride(expr: Expr, stride: u32) -> Expr {
    let Expr::Binary {
        op: "*",
        left,
        right,
    } = &expr
    else {
        return expr;
    };
    let held = |side: &Expr| matches!(side, Expr::Literal { bits, .. } if bits.iter().all(|bit| *bit == stride));
    match (held(left), held(right)) {
        (true, false) => (**right).clone(),
        (false, true) => (**left).clone(),
        _ => expr,
    }
}

/// Components sharing a name come back as one read; the rest are put side by side.
fn gathered(parts: Vec<(String, u8)>) -> Expr {
    match parts.split_first() {
        Some((first, rest)) if rest.iter().all(|(base, _)| *base == first.0) => Expr::Read {
            base: first.0.clone(),
            swizzle: parts.iter().map(|(_, comp)| *comp).collect(),
        },
        _ => Expr::Vector(
            parts
                .into_iter()
                .map(|(base, comp)| Expr::Read {
                    base,
                    swizzle: vec![comp],
                })
                .collect(),
        ),
    }
}

/// The components a set of lanes covers, as bits.
fn covering(lanes: &[u8]) -> u8 {
    lanes.iter().fold(0, |bits, lane| bits | 1 << (lane % 4))
}

/// Where a register stops carrying one value and starts carrying the next.
///
/// The compiler hands a register to one value after another, and reading it back gives no sign of
/// that: every one of them is spelled the same, so following any of them means checking each line
/// in between for a write that ended it. A value can take a name of its own where the one before it
/// is finished with, which is where it is written over in the same straight run of instructions,
/// with nothing reading the register in between, and nothing reading a component afterwards that
/// the run does not write. Anywhere else a branch could arrive still holding the value before, and
/// the register's own name is what carries it.
fn splits(defs: &[Def], touches: &[Touch], folded: &[bool], localised: &[bool]) -> Vec<bool> {
    let mut written: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    let mut read: HashMap<(u32, u32), Vec<(usize, u8)>> = HashMap::new();
    for (id, def) in defs.iter().enumerate() {
        written.entry(def.key).or_default().push(id);
    }
    for (key, lane, at) in touches {
        read.entry(*key).or_default().push((*at, *lane));
    }
    // The components read from a point on, gathered from the back so each answer is one lookup.
    let after: HashMap<(u32, u32), Vec<u8>> = read
        .iter_mut()
        .map(|(key, times)| {
            times.sort_unstable();
            let mut bits = vec![0u8; times.len() + 1];
            for (at, (_, lane)) in times.iter().enumerate().rev() {
                bits[at] = bits[at + 1] | 1 << lane;
            }
            (*key, bits)
        })
        .collect();
    // Where the reads of a register after a point begin, which answers both what is still to be
    // read and whether anything is read in a span at all.
    let onwards = |key: &(u32, u32), from: usize| -> usize {
        read.get(key)
            .map_or(0, |times| times.partition_point(|(at, _)| *at <= from))
    };
    let reach = |key: &(u32, u32), from: usize| -> u8 {
        after.get(key).map_or(0, |bits| bits[onwards(key, from)])
    };
    let between = |key: &(u32, u32), from: usize, to: usize| -> bool {
        read.get(key)
            .and_then(|times| times.get(onwards(key, from)))
            .is_some_and(|(at, _)| *at <= to)
    };

    let mut splits = vec![false; defs.len()];
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for (id, def) in defs.iter().enumerate() {
        // Nothing to split away from until the register has held something before, and nothing to
        // name where the value never reaches the register in the first place.
        if seen.insert(def.key) || def.block != 0 || def.fixed || folded[id] || localised[id] {
            continue;
        }
        let mut covered = 0;
        for over in written[&def.key].iter().copied().skip_while(|at| *at < id) {
            if defs[over].block != 0 {
                continue;
            }
            // A read taken between the value that starts the run and the end of it is a read of the
            // value before, which the new name no longer holds.
            if over > id && between(&def.key, id, over) {
                break;
            }
            covered |= covering(&defs[over].lanes);
            if reach(&def.key, over) & !covered == 0 {
                splits[id] = true;
                break;
            }
            if covered == 0xF {
                break;
            }
        }
    }
    splits
}

/// A cast to as many lanes as are being converted.
fn sized(base: &str, width: usize) -> String {
    match width {
        0 | 1 => base.to_owned(),
        lanes => format!("{base}{lanes}"),
    }
}

/// Destination operands an instruction leads with.
fn destinations(opcode: Opcode) -> usize {
    match opcode {
        Opcode::Sincos | Opcode::IMul | Opcode::UMul | Opcode::UDiv | Opcode::Swapc => 2,
        Opcode::Ret
        | Opcode::Retc
        | Opcode::Discard
        | Opcode::Break
        | Opcode::Breakc
        | Opcode::Continue
        | Opcode::Continuec
        | Opcode::Nop
        | Opcode::If
        | Opcode::Else
        | Opcode::EndIf
        | Opcode::Loop
        | Opcode::EndLoop
        | Opcode::Emit
        | Opcode::Cut
        | Opcode::Sync
        | Opcode::StoreRaw
        | Opcode::StoreStructured
        | Opcode::StoreUavTyped => 0,
        _ => 1,
    }
}

/// Whether the sources line up component for component with the destination. What does not — a dot
/// product, a texture fetch — builds its whole value first and is narrowed afterwards.
fn elementwise(opcode: Opcode) -> bool {
    !matches!(
        opcode,
        Opcode::Dp2
            | Opcode::Dp3
            | Opcode::Dp4
            | Opcode::Sample
            | Opcode::SampleB
            | Opcode::SampleL
            | Opcode::SampleD
            | Opcode::SampleC
            | Opcode::SampleCLz
            | Opcode::Gather4
            | Opcode::Ld
            | Opcode::LdMs
            | Opcode::LdStructured
            | Opcode::LdRaw
            | Opcode::Resinfo
            | Opcode::BufInfo
            | Opcode::SampleInfo
            | Opcode::SamplePos
            | Opcode::Lod
    )
}

/// The components an operand supplies, given the destination lanes wanted.
fn components(operand: &Operand, lanes: &[u8]) -> Vec<u8> {
    match &operand.components {
        ComponentSelect::Swizzle(swizzle) => lanes
            .iter()
            .map(|lane| swizzle[usize::from(*lane) % swizzle.len()])
            .collect(),
        ComponentSelect::Mask(mask) => (0..4).filter(|bit| mask & (1 << bit) != 0).collect(),
        ComponentSelect::Scalar(component) => vec![*component],
        ComponentSelect::OneComponent => vec![0],
        ComponentSelect::ZeroComponent => Vec::new(),
    }
}

/// The destination components an instruction writes, ascending.
fn written(operand: &Operand) -> Vec<u8> {
    match &operand.components {
        ComponentSelect::Mask(mask) => (0..4).filter(|bit| mask & (1 << bit) != 0).collect(),
        ComponentSelect::Swizzle(swizzle) => swizzle.to_vec(),
        ComponentSelect::Scalar(component) => vec![*component],
        _ => vec![0],
    }
}

fn immediate(operand: &Operand) -> Option<u32> {
    match operand.indices.first()? {
        OperandIndex::Imm32(value) => Some(*value),
        _ => None,
    }
}

/// The bit operations the machine has and the language does not, written out so the reading is a
/// whole shader rather than one that leans on names nothing defines.
fn bitfield_extract(width: usize) -> String {
    let kind = sized("uint", width);
    format!(
        "{kind} bitfield_extract({kind} value, {kind} offset, {kind} width)\n{{\n    \
         return select(width == 0, ({kind})0, (value << (32 - width - offset)) >> (32 - width));\n}}"
    )
}

fn bitfield_insert(width: usize) -> String {
    let kind = sized("uint", width);
    format!(
        "{kind} bitfield_insert({kind} base, {kind} insert, {kind} offset, {kind} width)\n{{\n    \
         {kind} mask = ((({kind})1 << width) - 1) << offset;\n    \
         return ((insert << offset) & mask) | (base & ~mask);\n}}"
    )
}

/// The top half of a multiply, which the machine produces alongside the bottom and HLSL cannot
/// express directly. Split into halves so the intermediate stays inside thirty-two bits.
fn mul_high(width: usize) -> String {
    let kind = sized("uint", width);
    format!(
        "{kind} mul_high({kind} a, {kind} b)\n{{\n    \
         {kind} low = (a & 0xffffu) * (b & 0xffffu);\n    \
         {kind} mid = (a >> 16) * (b & 0xffffu) + (low >> 16);\n    \
         {kind} rest = (a & 0xffffu) * (b >> 16) + (mid & 0xffffu);\n    \
         return (a >> 16) * (b >> 16) + (mid >> 16) + (rest >> 16);\n}}"
    )
}

/// What a resource reports about itself. One is written per texture that is asked, because a
/// function taking a texture as a parameter is not something every compiler accepts.
fn dimensions_helper(name: &str, dimension: &str) -> String {
    let (outs, value) = match dimension {
        "texture1d" => ("width, levels", "float4(width, 0.0, 0.0, levels)"),
        "texture1darray" => (
            "width, elements, levels",
            "float4(width, elements, 0.0, levels)",
        ),
        "texture3d" => (
            "width, height, depth, levels",
            "float4(width, height, depth, levels)",
        ),
        "texture2darray" | "texturecubearray" => (
            "width, height, elements, levels",
            "float4(width, height, elements, levels)",
        ),
        _ => (
            "width, height, levels",
            "float4(width, height, 0.0, levels)",
        ),
    };
    format!(
        "float4 dimensions_{name}(float mip)\n{{\n    uint {outs};\n    \
         {name}.GetDimensions((uint)mip, {outs});\n    return {value};\n}}"
    )
}

impl<'a> Builder<'a> {
    /// A builder over a program, primed with what its declarations say about the resources it
    /// binds: each texture's dimension and return type, and each buffer's stride and span.
    pub fn new(
        program: &'a Program,
        names: &'a Names,
        computed: &'a HashSet<u16>,
        reading: super::Reading,
    ) -> Self {
        let mut dimensions = HashMap::new();
        let mut kinds = HashMap::new();
        let mut strides = HashMap::new();
        let mut spans = HashMap::new();
        for instruction in &program.instructions {
            let Some(slot) = instruction.operands().first().and_then(immediate) else {
                continue;
            };
            match &instruction.kind {
                InstructionKind::DclResource { dimension, .. } => {
                    let count = match *dimension {
                        "texture1d" | "buffer" => 1,
                        "texture1darray" | "texture2d" | "texture2dms" => 2,
                        "texturecubearray" => 4,
                        _ => 3,
                    };
                    dimensions.insert(slot, count);
                    kinds.insert(slot, *dimension);
                }
                InstructionKind::DclResourceStructured { stride, .. } => {
                    strides.insert(slot, *stride);
                }
                InstructionKind::DclConstantBuffer { .. } => {
                    if let Some(OperandIndex::Imm32(span)) = instruction
                        .operands()
                        .first()
                        .and_then(|held| held.indices.get(1))
                    {
                        spans.insert(slot as u16, *span);
                    }
                }
                _ => {}
            }
        }
        Self {
            instructions: &program.instructions,
            names,
            cells: HashMap::new(),
            defs: Vec::new(),
            usage: Vec::new(),
            waiting: HashMap::new(),
            touches: Vec::new(),
            folded: Vec::new(),
            localised: Vec::new(),
            splits: Vec::new(),
            reading,
            assume: true,
            versions: HashMap::new(),
            renamed: Vec::new(),
            loops: 0,
            issued: HashMap::new(),
            dimensions,
            kinds,
            strides,
            spans,
            reachable: HashMap::new(),
            indexers: HashMap::new(),
            elemental: Vec::new(),
            arrays: HashSet::new(),
            computed,
            returns: !names.outputs.is_empty(),
            helpers: std::collections::BTreeSet::new(),
            emitted: Vec::new(),
            block: 0,
            blocks: 0,
            out: Vec::new(),
        }
    }

    /// Which values can be folded into their reader, after a run that only counted.
    fn decide(&self) -> (Vec<bool>, Vec<bool>) {
        // A choice between two constants is what the machine leaves where the shader asked for a
        // conditional, and parking it in a register hides that: `r3.w * r2.w + 1.0` says nothing,
        // while the same line with the choice in it is a ternary anything can recognise. It costs
        // nothing to write out again at each reader, so it goes to all of them rather than one —
        // which is why it has to have survived to the last of them rather than merely to the first.
        let repeatable = |id: usize, def: &Def, usage: &Usage| {
            matches!(&def.expr, Expr::Select { then, els, .. }
                if matches!(**then, Expr::Literal { .. }) && matches!(**els, Expr::Literal { .. }))
                && survives(&self.defs, def, id, usage.last)
        };
        let folded: Vec<bool> = self
            .defs
            .iter()
            .zip(&self.usage)
            .enumerate()
            .map(|(id, (def, usage))| {
                !def.fixed
                    && !usage.mixed
                    && !usage.distant
                    && !usage.stale
                    // Nothing reading it does not mean nothing needs it: a branch the walk did not
                    // follow may be what leaves the register holding this, so a value read nowhere
                    // still has to be written down.
                    && usage.reads > 0
                    && (usage.reads == 1 || repeatable(id, def, usage))
            })
            .collect();
        // A register component two blocks both write is one the branch not taken leaves behind, and
        // only the register itself carries that. Anything else names a value one of them never
        // reached.
        let mut writers: HashMap<Slot, usize> = HashMap::new();
        for def in &self.defs {
            for lane in &def.lanes {
                let slot = (def.key.0, def.key.1, lane % 4);
                let block = writers.entry(slot).or_insert(def.block);
                if *block != def.block {
                    *block = usize::MAX;
                }
            }
        }
        let shared = |def: &Def| {
            def.lanes
                .iter()
                .any(|lane| writers.get(&(def.key.0, def.key.1, lane % 4)) == Some(&usize::MAX))
        };
        // A value that is not a float costs a reinterpretation going into a register and another
        // coming out, which is noise rather than arithmetic. Given a name of its own it needs
        // neither. Only where every read takes it whole, and none from another block: a name
        // declared inside one is not in scope outside it.
        let localised = self
            .defs
            .iter()
            .zip(&self.usage)
            .zip(&folded)
            .map(|((def, usage), folded)| {
                !folded
                    && !def.fixed
                    && !def.looped
                    && matches!(def.domain, Domain::Int | Domain::Uint | Domain::Bool)
                    && !usage.mixed
                    && !usage.distant
                    && !shared(def)
            })
            .collect();
        (folded, localised)
    }

    fn reset(&mut self, folded: Vec<bool>, localised: Vec<bool>, assume: bool) {
        self.issued.clear();
        self.loops = 0;
        self.assume = assume;
        self.localised = localised;
        self.cells.clear();
        self.defs.clear();
        self.usage.clear();
        self.waiting.clear();
        self.reachable.clear();
        self.indexers.clear();
        self.touches.clear();
        self.versions.clear();
        self.renamed.clear();
        self.out.clear();
        self.emitted.clear();
        self.helpers.clear();
        self.block = 0;
        self.blocks = 0;
        self.folded = folded;
    }

    /// The body of a shader, walked until the reading it produces is the one its own decisions
    /// describe.
    pub fn run(&mut self, tree: &[super::Stmt], depth: usize) {
        if self.reading == super::Reading::Plain {
            self.reset(Vec::new(), Vec::new(), true);
            self.body(tree, depth);
            return;
        }
        // The counting walk takes every value to depend on everything behind it, which is the most a
        // later write can disturb and so the one decision that is safe without knowing the others.
        self.reset(Vec::new(), Vec::new(), true);
        self.body(tree, depth);
        let counted = self.usage.clone();
        let (careful, guarded) = self.decide();
        self.reading_arrays();

        // What a value depends on is what its text names, and a value written down leaves behind a
        // name settled where it was made: a register, or one of its own that is only ever assigned
        // once. Nothing later can disturb what went into it. A value that moves instead brings what
        // it names along, so which values move decides which are free to, and that only settles by
        // going round.
        //
        // Both decisions have to come back unchanged, not just the folding. A value's domain
        // follows what it was made from, and a written-down integer reads back as a float, so
        // folding moves domains and domains decide which values take a name of their own. Accepting
        // a reading whose names were chosen under a domain it no longer has would declare one as the
        // wrong type, which converts where it meant to reinterpret and still compiles.
        let (mut folded, mut localised) = (careful.clone(), guarded.clone());
        let mut settled = false;
        for _ in 0..ROUNDS {
            // Where a register starts over rests on which components are written and read. Those are
            // the same whatever moves, so this is worked out from the walk before and stands.
            self.splits = splits(&self.defs, &self.touches, &folded, &localised);
            self.reset(folded.clone(), localised.clone(), false);
            self.body(tree, depth);
            let (again, held) = self.decide();
            settled = again == folded && held == localised;
            if settled {
                break;
            }
            (folded, localised) = (again, held);
        }
        if !settled {
            // Nothing starts over, since the only split points worked out are the ones belonging to
            // the reading just rejected. A register under its own name always reads correctly; it
            // only reads worse.
            self.splits.clear();
            self.reset(careful, guarded, true);
            self.body(tree, depth);
        }

        // How often a value is used comes out of the walk's own bookkeeping rather than out of any
        // of these decisions, so the walks agree on it whatever they decided. This has never had
        // anything to catch; it would take an instruction understood one way on one walk and
        // another way on the next, which is the one thing that would drop a value silently.
        if self.usage.len() != counted.len()
            || self
                .usage
                .iter()
                .zip(&counted)
                .any(|(second, first)| second.reads != first.reads)
        {
            self.splits.clear();
            self.reset(Vec::new(), Vec::new(), true);
            self.body(tree, depth);
        }
        // Recovering a transform sums the rows in the order the matrix has them rather than the order
        // the shader wrote them, so an exact reading leaves the lines as they are. Putting component
        // writes back together changes no arithmetic, so that stands either way.
        let assigned = match self.reading {
            super::Reading::Exact => self.emitted.clone(),
            _ => super::matrix::fold(self.names, &mut self.out, &self.emitted),
        };
        super::matrix::coalesce(&mut self.out, &assigned);
    }

    fn line(&mut self, depth: usize, text: String) {
        self.out.push(format!("{}{text}", "    ".repeat(depth)));
        self.emitted.push(None);
    }

    fn body(&mut self, tree: &[super::Stmt], depth: usize) {
        for stmt in tree {
            match stmt {
                super::Stmt::Op(at) => self.instruction(*at, depth),
                super::Stmt::If { at, then, els } => {
                    let test = self.condition(*at);
                    self.line(depth, format!("if ({test})"));
                    self.line(depth, "{".to_owned());
                    self.nested(then, depth + 1);
                    if !els.is_empty() {
                        self.line(depth, "}".to_owned());
                        self.line(depth, "else".to_owned());
                        self.line(depth, "{".to_owned());
                        self.nested(els, depth + 1);
                    }
                    self.line(depth, "}".to_owned());
                }
                super::Stmt::Loop(inner) => {
                    self.line(depth, "while (true)".to_owned());
                    self.line(depth, "{".to_owned());
                    self.loops += 1;
                    self.nested(inner, depth + 1);
                    self.loops -= 1;
                    self.line(depth, "}".to_owned());
                }
            }
        }
    }

    fn nested(&mut self, tree: &[super::Stmt], depth: usize) {
        self.blocks += 1;
        let outer = std::mem::replace(&mut self.block, self.blocks);
        self.body(tree, depth);
        self.block = outer;
    }

    /// The test an `if`, `discard` or conditional break is taken on.
    fn condition(&mut self, at: usize) -> String {
        let Some(instruction) = self.instructions.get(at) else {
            return "true".to_owned();
        };
        let Some(operand) = instruction.operands().first().cloned() else {
            return "true".to_owned();
        };
        let sourced = self.read(&operand, &[0]);
        let test = coerce(sourced.expr, sourced.domain, Domain::Bool);
        match instruction.test_nonzero {
            true => test.text(),
            false => Expr::Unary {
                op: "!",
                value: Box::new(test),
            }
            .text(),
        }
    }

    fn instruction(&mut self, at: usize, depth: usize) {
        let Some(instruction) = self.instructions.get(at) else {
            return;
        };
        if !matches!(instruction.kind, InstructionKind::Generic { .. }) {
            return;
        }
        let opcode = instruction.opcode;
        match opcode {
            Opcode::Ret => {
                let text = match self.returns {
                    true => "return output;",
                    false => "return;",
                };
                self.line(depth, text.to_owned());
            }
            Opcode::Nop => {}
            Opcode::Break => self.line(depth, "break;".to_owned()),
            Opcode::Continue => self.line(depth, "continue;".to_owned()),
            Opcode::Discard | Opcode::Breakc | Opcode::Continuec | Opcode::Retc => {
                let test = self.condition(at);
                let action = match opcode {
                    Opcode::Discard => "discard".to_owned(),
                    Opcode::Breakc => "break".to_owned(),
                    Opcode::Continuec => "continue".to_owned(),
                    _ => match self.returns {
                        true => "return output".to_owned(),
                        false => "return".to_owned(),
                    },
                };
                self.line(depth, format!("if ({test}) {action};"));
            }
            _ => self.assignment(at, depth),
        }
    }

    /// The instruction as the disassembler writes it, for anything not understood well enough to say
    /// more than the machine did. Keeping the line is what stops a reading from quietly losing an
    /// instruction.
    fn verbatim(&mut self, at: usize, depth: usize) {
        let text = format!(
            "// {}",
            dxbc::shex::format_instruction(&self.instructions[at])
        );
        self.line(depth, text);
    }

    fn assignment(&mut self, at: usize, depth: usize) {
        let instruction = &self.instructions[at];
        let opcode = instruction.opcode;
        let saturate = instruction.saturate;
        let operands = instruction.operands().to_vec();
        let count = destinations(opcode);
        if count == 0 || operands.len() <= count {
            self.verbatim(at, depth);
            return;
        }

        let lanes = written(&operands[0]);
        let source_lanes: Vec<u8> = match elementwise(opcode) {
            true => lanes.clone(),
            false => (0..4).collect(),
        };
        let mut reads = Vec::new();
        let mut sources = Vec::new();
        for operand in &operands[count..] {
            let sourced = self.read(operand, &source_lanes);
            reads.extend(sourced.reads.iter().copied());
            sources.push((sourced.expr, sourced.domain));
        }

        let values = self.evaluate(at, &sources, &operands, lanes.len());
        if values.is_empty() {
            self.verbatim(at, depth);
            return;
        }
        let written_to: Vec<(&Operand, Expr, Domain)> = values
            .into_iter()
            .enumerate()
            .filter_map(|(index, value)| {
                let (expr, domain) = value?;
                let dest = operands.get(index)?;
                // A value built whole rather than component by component still has to be narrowed
                // to the components the instruction actually writes.
                let expr = match elementwise(opcode) {
                    true => expr,
                    false => expr.select(&written(dest)),
                };
                Some((dest, expr, domain))
            })
            .collect();
        // An instruction with two destinations writes them out one line at a time, and both lines
        // carry whatever was folded into the instruction. So a value it read has to survive the
        // earlier line's write to still be there for the later one, and that write has already
        // happened by the time the later line is read.
        let arriving = self.defs.len();
        for (dest, ..) in written_to.iter().rev().skip(1) {
            let index = immediate(dest).unwrap_or(0);
            for lane in written(dest) {
                let slot = (dest.reg_type.to_u32(), index, lane);
                for waiting in self.waiting.get(&slot).into_iter().flatten() {
                    if self.usage[*waiting].last == arriving {
                        self.usage[*waiting].stale = true;
                    }
                }
            }
        }
        for (dest, expr, domain) in written_to {
            self.define(dest, expr, domain, reads.clone(), saturate, depth);
        }
    }

    /// What the instruction computes, one entry per destination.
    ///
    /// Sources arrive already narrowed to the destination's components, so everything elementwise is
    /// just its operator. The rest build a whole value and let the destination mask take from it.
    fn evaluate(
        &mut self,
        at: usize,
        sources: &[(Expr, Domain)],
        rest: &[Operand],
        width: usize,
    ) -> Vec<Option<(Expr, Domain)>> {
        let instruction = &self.instructions[at];
        let opcode = instruction.opcode;
        let rest = &rest[destinations(opcode).min(rest.len())..];
        let get = |at: usize, want: Domain| match sources.get(at) {
            Some((expr, domain)) => coerce(expr.clone(), *domain, want),
            None => Expr::Literal {
                bits: vec![0],
                domain: want,
            },
        };
        let float = |at: usize| get(at, Domain::Float);
        let int = |at: usize| get(at, Domain::Int);
        let uint = |at: usize| get(at, Domain::Uint);
        let unary = |op, value| Expr::Unary {
            op,
            value: Box::new(value),
        };
        let binary = |op, left, right| Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        };

        let single = |expr, domain| vec![Some((expr, domain))];
        use Opcode as O;
        match opcode {
            O::Mov => match sources.first() {
                Some((expr, domain)) => single(expr.clone(), *domain),
                None => Vec::new(),
            },
            O::Add => single(binary("+", float(0), float(1)), Domain::Float),
            O::Mul => single(binary("*", float(0), float(1)), Domain::Float),
            O::Div => single(binary("/", float(0), float(1)), Domain::Float),
            O::Mad => single(
                binary("+", binary("*", float(0), float(1)), float(2)),
                Domain::Float,
            ),
            O::Min => single(call("min", vec![float(0), float(1)], width), Domain::Float),
            O::Max => single(call("max", vec![float(0), float(1)], width), Domain::Float),
            O::Frc => single(call("frac", vec![float(0)], width), Domain::Float),
            O::Exp => single(call("exp2", vec![float(0)], width), Domain::Float),
            O::Log => single(call("log2", vec![float(0)], width), Domain::Float),
            O::Rcp => single(call("rcp", vec![float(0)], width), Domain::Float),
            O::Rsq => single(call("rsqrt", vec![float(0)], width), Domain::Float),
            O::Sqrt => single(call("sqrt", vec![float(0)], width), Domain::Float),
            O::Round_ne => single(call("round", vec![float(0)], width), Domain::Float),
            O::Round_ni => single(call("floor", vec![float(0)], width), Domain::Float),
            O::Round_pi => single(call("ceil", vec![float(0)], width), Domain::Float),
            O::Round_z => single(call("trunc", vec![float(0)], width), Domain::Float),
            O::Deriv_rtx => single(call("ddx", vec![float(0)], width), Domain::Float),
            O::Deriv_rty => single(call("ddy", vec![float(0)], width), Domain::Float),
            O::Deriv_rtx_coarse => single(call("ddx_coarse", vec![float(0)], width), Domain::Float),
            O::Deriv_rty_coarse => single(call("ddy_coarse", vec![float(0)], width), Domain::Float),
            O::Deriv_rtx_fine => single(call("ddx_fine", vec![float(0)], width), Domain::Float),
            O::Deriv_rty_fine => single(call("ddy_fine", vec![float(0)], width), Domain::Float),
            O::Eq => single(binary("==", float(0), float(1)), Domain::Bool),
            O::Ne => single(binary("!=", float(0), float(1)), Domain::Bool),
            O::Lt => single(binary("<", float(0), float(1)), Domain::Bool),
            O::Ge => single(binary(">=", float(0), float(1)), Domain::Bool),
            O::IEq => single(binary("==", int(0), int(1)), Domain::Bool),
            O::INe => single(binary("!=", int(0), int(1)), Domain::Bool),
            O::ILt => single(binary("<", int(0), int(1)), Domain::Bool),
            O::IGe => single(binary(">=", int(0), int(1)), Domain::Bool),
            O::ULt => single(binary("<", uint(0), uint(1)), Domain::Bool),
            O::UGe => single(binary(">=", uint(0), uint(1)), Domain::Bool),
            // Comparisons leave a mask, and the shader combines masks with the bitwise operators.
            // Where both sides really are conditions, the logical ones read better and mean the same.
            O::And | O::Or => {
                let both = sources.iter().all(|(_, domain)| *domain == Domain::Bool);
                let (logical, bitwise) = match opcode {
                    O::And => ("&&", "&"),
                    _ => ("||", "|"),
                };
                // A mask anded with a constant is that constant or nothing, which is the choice the
                // shader was written with. Saying it in the domain the constant plainly belongs to
                // is what stops the reading going out through `asfloat` and back again; the bits are
                // the same either way, since every cast between domains reinterprets rather than
                // converts.
                if let (O::And, Some((cond, held))) = (opcode, chosen(sources)) {
                    // Anded with one, the mask is the condition itself and nothing more. Handing it
                    // back as a condition is what lets it be written down as a `bool` where it
                    // cannot move, and read back as a choice by whatever wanted a number.
                    let Expr::Literal { bits, .. } = &held else {
                        unreachable!("chosen only answers with a constant")
                    };
                    if bits.iter().all(|held| f32::from_bits(*held) == 1.0) {
                        return single(cond, Domain::Bool);
                    }
                    return single(
                        Expr::Select {
                            cond: Box::new(cond),
                            then: Box::new(held.clone()),
                            els: Box::new(Expr::Literal {
                                bits: vec![0; held.width()],
                                domain: Domain::Float,
                            }),
                        },
                        Domain::Float,
                    );
                }
                match both {
                    true => single(
                        binary(logical, get(0, Domain::Bool), get(1, Domain::Bool)),
                        Domain::Bool,
                    ),
                    false => single(binary(bitwise, uint(0), uint(1)), Domain::Uint),
                }
            }
            O::Xor => single(binary("^", uint(0), uint(1)), Domain::Uint),
            O::Not => single(unary("~", uint(0)), Domain::Uint),
            O::INeg => single(unary("-", int(0)), Domain::Int),
            O::Iadd => single(binary("+", int(0), int(1)), Domain::Int),
            O::IMax => single(call("max", vec![int(0), int(1)], width), Domain::Int),
            O::IMin => single(call("min", vec![int(0), int(1)], width), Domain::Int),
            O::UMax => single(call("max", vec![uint(0), uint(1)], width), Domain::Uint),
            O::UMin => single(call("min", vec![uint(0), uint(1)], width), Domain::Uint),
            O::IMad => single(
                binary("+", binary("*", int(0), int(1)), int(2)),
                Domain::Int,
            ),
            O::UMad => single(
                binary("+", binary("*", uint(0), uint(1)), uint(2)),
                Domain::Uint,
            ),
            O::Ishl => single(binary("<<", int(0), uint(1)), Domain::Int),
            O::Ishr => single(binary(">>", int(0), uint(1)), Domain::Int),
            O::Ushr => single(binary(">>", uint(0), uint(1)), Domain::Uint),
            O::Ftoi => single(
                call(&sized("int", width), vec![float(0)], width),
                Domain::Int,
            ),
            O::Ftou => single(
                call(&sized("uint", width), vec![float(0)], width),
                Domain::Uint,
            ),
            O::Itof => single(
                call(&sized("float", width), vec![int(0)], width),
                Domain::Float,
            ),
            O::Utof => single(
                call(&sized("float", width), vec![uint(0)], width),
                Domain::Float,
            ),
            O::Movc => {
                let domain = sources.get(1).map_or(Domain::Float, |(_, domain)| *domain);
                single(
                    Expr::Select {
                        cond: Box::new(get(0, Domain::Bool)),
                        then: Box::new(get(1, domain)),
                        els: Box::new(get(2, domain)),
                    },
                    domain,
                )
            }
            O::Dp2 | O::Dp3 | O::Dp4 => {
                let take = match opcode {
                    O::Dp2 => 2,
                    O::Dp3 => 3,
                    _ => 4,
                };
                let lanes: Vec<u8> = (0..take).collect();
                single(
                    call(
                        "dot",
                        vec![float(0).select(&lanes), float(1).select(&lanes)],
                        1,
                    ),
                    Domain::Float,
                )
            }
            // Both destinations come from the one angle, and either may be discarded.
            O::Sincos => vec![
                Some((call("sin", vec![float(0)], width), Domain::Float)),
                Some((call("cos", vec![float(0)], width), Domain::Float)),
            ],
            O::IMul | O::UMul => {
                let domain = match opcode {
                    O::IMul => Domain::Int,
                    _ => Domain::Uint,
                };
                // The high half is only ever discarded in practice, so it costs nothing
                // to say plainly that it is not reconstructed.
                let high = match self.instructions[at]
                    .operands()
                    .first()
                    .map(|dest| dest.reg_type)
                {
                    Some(RegisterType::Null) | None => None,
                    _ => {
                        self.helpers.insert(mul_high(width));
                        Some((
                            call("mul_high", vec![get(0, domain), get(1, domain)], width),
                            domain,
                        ))
                    }
                };
                vec![
                    high,
                    Some((binary("*", get(0, domain), get(1, domain)), domain)),
                ]
            }
            O::UDiv => vec![
                Some((binary("/", uint(0), uint(1)), Domain::Uint)),
                Some((binary("%", uint(0), uint(1)), Domain::Uint)),
            ],
            // The condition picks which way round the two sources come out.
            O::Swapc => {
                let cond = get(0, Domain::Bool);
                let pick = |then: usize, els: usize| {
                    Some((
                        Expr::Select {
                            cond: Box::new(cond.clone()),
                            then: Box::new(get(then, Domain::Float)),
                            els: Box::new(get(els, Domain::Float)),
                        },
                        Domain::Float,
                    ))
                };
                vec![pick(2, 1), pick(1, 2)]
            }
            O::Ubfe | O::Ibfe => {
                let domain = match opcode {
                    O::Ubfe => Domain::Uint,
                    _ => Domain::Int,
                };
                self.helpers.insert(bitfield_extract(width));
                single(
                    call(
                        "bitfield_extract",
                        vec![get(2, domain), uint(1), uint(0)],
                        width,
                    ),
                    domain,
                )
            }
            O::Bfi => {
                self.helpers.insert(bitfield_insert(width));
                single(
                    call(
                        "bitfield_insert",
                        vec![uint(3), uint(2), uint(1), uint(0)],
                        width,
                    ),
                    Domain::Uint,
                )
            }
            O::Sample
            | O::SampleB
            | O::SampleL
            | O::SampleD
            | O::SampleC
            | O::SampleCLz
            | O::Gather4
            | O::Ld
            | O::LdMs
            | O::LdStructured
            | O::LdRaw
            | O::Resinfo => self.fetch(at, sources, rest),
            _ => Vec::new(),
        }
    }

    /// A texture or buffer read, which names its resource and sampler rather than their slots.
    fn fetch(
        &mut self,
        at: usize,
        sources: &[(Expr, Domain)],
        rest: &[Operand],
    ) -> Vec<Option<(Expr, Domain)>> {
        let instruction = &self.instructions[at];
        let opcode = instruction.opcode;
        // The resource is the operand after the coordinate everywhere but a structured load, which
        // takes an element index and a byte offset first.
        let resource_at = match opcode {
            Opcode::LdStructured => 2,
            _ => 1,
        };
        let Some(resource) = rest.get(resource_at) else {
            return Vec::new();
        };
        let slot = immediate(resource).unwrap_or(0);
        let name = self.names.texture(slot as u16);
        let sampler = rest
            .get(resource_at + 1)
            .filter(|operand| operand.reg_type == RegisterType::Sampler)
            .map(|operand| self.names.sampler(immediate(operand).unwrap_or(0) as u16));
        let dimension = self.dimensions.get(&slot).copied().unwrap_or(2);
        let scalar = |source: Option<&(Expr, Domain)>, want| match source {
            Some((expr, domain)) => coerce(expr.clone(), *domain, want).select(&[0]),
            None => Expr::Literal {
                bits: vec![0],
                domain: want,
            },
        };
        // Every sampling method leads with the sampler and the coordinate, and differs only in what
        // follows them.
        let sampled = |method: &'static str| {
            let lanes: Vec<u8> = (0..dimension as u8).collect();
            let mut args: Vec<Expr> = sampler
                .iter()
                .map(|name| Expr::Read {
                    base: name.clone(),
                    swizzle: Vec::new(),
                })
                .collect();
            args.push(match sources.first() {
                Some((expr, domain)) => coerce(expr.clone(), *domain, Domain::Float).select(&lanes),
                None => Expr::Literal {
                    bits: vec![0],
                    domain: Domain::Float,
                },
            });
            (method, args)
        };

        let (method, mut args) = match opcode {
            Opcode::Sample => sampled("Sample"),
            Opcode::SampleB => sampled("SampleBias"),
            Opcode::SampleL => sampled("SampleLevel"),
            Opcode::SampleD => sampled("SampleGrad"),
            Opcode::SampleC => sampled("SampleCmp"),
            Opcode::SampleCLz => sampled("SampleCmpLevelZero"),
            Opcode::Gather4 => {
                // Which channel comes back is the sampler operand's own component select.
                let channel = rest
                    .get(resource_at + 1)
                    .map(|operand| match &operand.components {
                        ComponentSelect::Scalar(component) => *component,
                        ComponentSelect::Swizzle(swizzle) => swizzle[0],
                        _ => 0,
                    })
                    .unwrap_or(0);
                sampled(match channel {
                    1 => "GatherGreen",
                    2 => "GatherBlue",
                    3 => "GatherAlpha",
                    _ => "GatherRed",
                })
            }
            // A load takes integer coordinates with the mip level in the last component.
            Opcode::Ld | Opcode::LdMs => (
                "Load",
                vec![match sources.first() {
                    Some((expr, domain)) => coerce(expr.clone(), *domain, Domain::Int)
                        .select(&(0..=dimension as u8).collect::<Vec<_>>()),
                    None => Expr::Literal {
                        bits: vec![0],
                        domain: Domain::Int,
                    },
                }],
            ),
            Opcode::LdRaw => ("Load4", vec![scalar(sources.first(), Domain::Uint)]),
            Opcode::LdStructured => {
                let stride = self.strides.get(&slot).copied().unwrap_or(16);
                let index = scalar(sources.first(), Domain::Uint);
                let address = Expr::Binary {
                    op: "*",
                    left: Box::new(index),
                    right: Box::new(Expr::Literal {
                        bits: vec![stride],
                        domain: Domain::Uint,
                    }),
                };
                let offset = scalar(sources.get(1), Domain::Uint);
                // The offset within an element is nearly always the start of it.
                let address = match &offset {
                    Expr::Literal { bits, .. } if bits.iter().all(|bit| *bit == 0) => address,
                    offset => Expr::Binary {
                        op: "+",
                        left: Box::new(address),
                        right: Box::new(offset.clone()),
                    },
                };
                ("Load4", vec![address])
            }
            _ => {
                let kind = self.kinds.get(&slot).copied().unwrap_or("texture2d");
                self.helpers.insert(dimensions_helper(&name, kind));
                let mip = scalar(sources.first(), Domain::Float);
                return vec![Some((
                    call(&format!("dimensions_{name}"), vec![mip], 4),
                    Domain::Float,
                ))];
            }
        };
        if opcode == Opcode::SampleD {
            args.extend(sources.iter().skip(3).take(2).map(|(expr, domain)| {
                coerce(expr.clone(), *domain, Domain::Float).select(&[0, 1])
            }));
        }
        // A bias, a level and a comparison value all sit in the same place after the sampler.
        if matches!(
            opcode,
            Opcode::SampleB | Opcode::SampleL | Opcode::SampleC | Opcode::SampleCLz
        ) && let Some((expr, domain)) = sources.get(3)
        {
            args.push(coerce(expr.clone(), *domain, Domain::Float).select(&[0]));
        }

        // The resource operand's own swizzle permutes what comes back before the destination mask
        // takes from it.
        let domain = match method {
            "Load4" => Domain::Uint,
            _ => Domain::Float,
        };
        let fetched = call(&format!("{name}.{method}"), args, 4);
        let permuted = match &resource.components {
            ComponentSelect::Swizzle(swizzle) => fetched.select(swizzle),
            _ => fetched,
        };
        vec![Some((permuted, domain))]
    }

    /// Record a value against its destination, writing it down unless it can be folded into whatever
    /// reads it.
    fn define(
        &mut self,
        dest: &Operand,
        expr: Expr,
        domain: Domain,
        reads: Vec<Slot>,
        saturate: bool,
        depth: usize,
    ) {
        if dest.reg_type == RegisterType::Null {
            return;
        }
        let lanes = written(dest);
        let expr = match saturate {
            true => {
                let width = expr.width();
                call("saturate", vec![expr], width)
            }
            false => expr,
        };
        let exact = self.reading == super::Reading::Exact;
        let expr = super::idiom::simplify(expr, exact);
        // A value used only to pick an element of a buffer of structs holds the element, not the
        // register the element starts at, so the multiply that turned one into the other goes.
        let expr = match self.elemental.get(self.defs.len()).copied().unwrap_or(0) {
            0 => expr,
            step => without_stride(expr, step),
        };
        let index = immediate(dest).unwrap_or(0);
        let id = self.defs.len();
        let folded = self.folded.get(id).copied().unwrap_or(false);
        // A transform arrives one row at a time, each row folding into the next, so recognising it
        // has to wait until the sum is whole. A value going to its reader is not yet.
        let expr = match folded || exact {
            true => expr,
            false => super::matrix::transform(self.names, expr),
        };
        let key = (dest.reg_type.to_u32(), index);
        // The register starts over here, so it goes by a new name from this line on. Sources were
        // read before this, and still spell the value that was in it.
        if self.splits.get(id).copied().unwrap_or(false) {
            *self.versions.entry(key).or_default() += 1;
            let name = self.spelling(index);
            self.renamed.push(name);
        }
        let base = self.register(dest, &lanes).expr;
        let base = match base {
            Expr::Read { base, .. } => base,
            other => other.text(),
        };

        let (expr, domain) = match dest.reg_type {
            RegisterType::Output => match self.names.outputs.get(&index) {
                Some(entry) => {
                    let want = super::domain(&entry.kind);
                    (coerce(expr, domain, want), want)
                }
                None => (expr, domain),
            },
            _ => (expr, domain),
        };
        // Anything the walk cannot follow component by component has to be written down: a shader
        // output because the pipeline reads it, an indexable array because a later read may not name
        // a component this ever saw.
        let fixed = !matches!(dest.reg_type, RegisterType::Temp);
        // A value that has earned a name of its own takes one after the register it would have gone
        // to, kept apart from any earlier value of the same one.
        let local = self.localised.get(id).copied().unwrap_or(false);
        let name = match local {
            false => base.clone(),
            true => {
                let stem = format!("{base}_{}", letters(&lanes));
                let seen = self.issued.entry(stem.clone()).or_default();
                *seen += 1;
                match *seen {
                    1 => stem,
                    n => format!("{stem}_{n}"),
                }
            }
        };
        self.defs.push(Def {
            base: name,
            key,
            lanes: lanes.clone(),
            expr,
            domain,
            reads,
            block: self.block,
            fixed,
            local,
            looped: self.loops > 0,
        });
        self.usage.push(Usage::default());

        // A write ends any value that was computed from what it lands on. This runs before the new
        // value registers what it reads, because an instruction reading the register it writes —
        // `lt r0.x, r0.x, l(0)` — would otherwise end itself.
        let mut pending: Vec<usize> = lanes
            .iter()
            .filter_map(|lane| self.waiting.get(&(key.0, key.1, *lane)))
            .flatten()
            .copied()
            .collect();
        // Which element of an array a write lands on is not known, so it ends every value that read
        // from one.
        if dest.reg_type == RegisterType::IndexableTemp {
            let held = self.waiting.iter().filter(|(slot, _)| slot.0 == key.0);
            pending.extend(held.flat_map(|(_, defs)| defs));
        }
        for waiting in pending {
            if self.usage[waiting].reads == 0 {
                self.usage[waiting].stale = true;
            }
        }
        for slot in &self.defs[id].reads {
            self.waiting.entry(*slot).or_default().push(id);
        }

        if matches!(dest.reg_type, RegisterType::Temp | RegisterType::Output) {
            let cells = self.cells.entry(key).or_default();
            for (position, lane) in lanes.iter().enumerate() {
                if let Some(cell) = cells.get_mut(usize::from(*lane) % 4) {
                    *cell = Some((id, position as u8));
                }
            }
        }

        if folded {
            return;
        }
        let expr = self.defs[id].expr.clone();
        let domain = self.defs[id].domain;
        let target = match lanes.as_slice() {
            [0, 1, 2, 3] => base.clone(),
            _ => format!("{base}.{}", letters(&lanes)),
        };
        // A register is declared as floats, so an integer value that has to go into one goes in as
        // the bits it is rather than being converted to the nearest float and back. A value with a
        // name of its own is simply declared as what it is.
        let (text, stored) = match (local, domain) {
            (true, _) => {
                let kind = match domain {
                    Domain::Int => "int",
                    Domain::Bool => "bool",
                    _ => "uint",
                };
                let text = format!(
                    "{} {} = {};",
                    sized(kind, lanes.len()),
                    self.defs[id].base,
                    expr.text()
                );
                (text, expr)
            }
            (false, Domain::Int | Domain::Uint) => {
                let width = expr.width();
                let stored = super::idiom::simplify(call("asfloat", vec![expr], width), exact);
                (format!("{target} = {};", stored.text()), stored)
            }
            _ => (format!("{target} = {};", expr.text()), expr),
        };
        self.line(depth, text);
        // Beside the text, what it assigns as it assigns it, the cast into the register included:
        // several of these together may be one operation, or one value. A value with a name of its
        // own never reached the register it was named after, so it says nothing about what is there.
        if !local {
            *self.emitted.last_mut().expect("just pushed") = Some(super::matrix::Emitted {
                depth,
                base,
                lanes,
                expr: stored,
            });
        }
    }

    /// A source operand, folding in whatever value already sits in it.
    fn read(&mut self, operand: &Operand, lanes: &[u8]) -> Sourced {
        // A constant carries values rather than a swizzle, so its lanes are the destination's.
        let comps = match operand.reg_type {
            RegisterType::Immediate32 | RegisterType::Immediate64 => lanes.to_vec(),
            _ => components(operand, lanes),
        };
        if !matches!(operand.reg_type, RegisterType::Temp | RegisterType::Output) {
            return self.register(operand, &comps);
        }
        let index = immediate(operand).unwrap_or(0);
        let key = (operand.reg_type.to_u32(), index);
        let at = self.defs.len();
        self.touches
            .extend(comps.iter().map(|lane| (key, lane % 4, at)));
        let cells = self.cells.get(&key).copied().unwrap_or_default();
        let owners: Vec<Option<(usize, u8)>> = comps
            .iter()
            .map(|lane| cells[usize::from(*lane) % 4])
            .collect();

        // One value behind every component read, or the read is of several and none of them can move
        // into it.
        let single = owners.first().copied().flatten().filter(|(id, _)| {
            owners
                .iter()
                .all(|owner| matches!(owner, Some((other, _)) if other == id))
        });
        let Some((id, _)) = single else {
            for owner in owners.into_iter().flatten() {
                self.usage[owner.0].mixed = true;
            }
            // A read the walk cannot pin to one value still came out of a register, and a later
            // write to that register has to end anything folded from it. Leaving this off lets a
            // value be moved past the write that invalidates it.
            let mut sourced = self.register(operand, &comps);
            sourced
                .reads
                .extend(comps.iter().map(|lane| (key.0, key.1, lane % 4)));
            sourced.reads.sort_unstable();
            sourced.reads.dedup();
            return sourced;
        };

        self.usage[id].reads += 1;
        self.usage[id].last = at;
        if self.defs[id].block != self.block {
            self.usage[id].distant = true;
        }
        let positions: Vec<u8> = owners
            .iter()
            .filter_map(|owner| owner.map(|(_, at)| at))
            .collect();
        // A value that moves to its reader brings what it reads along, since that is what the text
        // ends up naming. One written down instead leaves behind a name settled where it was made,
        // and the reader depends on that name alone: what went into it happened already and cannot
        // be disturbed now. The name, not the register — a value given one of its own is assigned
        // once and never again, so a read of one depends on nothing at all: the register it was
        // named after never held it, and whatever lands there later is a different value entirely.
        let folded = self.folded.get(id).copied().unwrap_or(false);
        let mut reads = match self.assume || folded {
            true => self.defs[id].reads.clone(),
            false => Vec::new(),
        };
        if self.assume || !self.localised.get(id).copied().unwrap_or(false) {
            reads.extend(comps.iter().map(|lane| (key.0, key.1, lane % 4)));
        }
        // Reads accumulate through everything folded in, so a long chain would carry the same few
        // registers hundreds of times over. There are only ever a register file's worth of them.
        reads.sort_unstable();
        reads.dedup();

        let expr = match (folded, self.defs[id].local) {
            (true, _) => self.defs[id].expr.clone().select(&positions),
            // A name of its own holds only the value's own lanes, so a read of it counts from those
            // rather than from the register the value never went into. A name holding one thing
            // needs no component naming it, and wider ones drop an identity selection when they
            // come to be written.
            (false, true) => Expr::Read {
                base: self.defs[id].base.clone(),
                swizzle: match self.defs[id].lanes.len() == 1 {
                    true => Vec::new(),
                    false => positions.clone(),
                },
            },
            (false, false) => Expr::Read {
                base: self.defs[id].base.clone(),
                swizzle: comps.clone(),
            },
        };
        let expr = self.modified(operand, expr);
        // What was written down went in as bits, so reading it back is a reinterpretation. A
        // condition is the exception: it stores as one or zero and reads as itself.
        let domain = match (folded || self.defs[id].local, self.defs[id].domain) {
            (false, Domain::Int | Domain::Uint) => Domain::Float,
            (_, domain) => domain,
        };
        Sourced {
            expr,
            domain,
            reads,
            owner: Some(id),
        }
    }

    /// An operand as the register it names, for everything the walk does not hold a value for.
    fn register(&mut self, operand: &Operand, comps: &[u8]) -> Sourced {
        let mut reads = Vec::new();
        if operand.reg_type == RegisterType::Immediate32 {
            let bits: Vec<u32> = match operand.immediate_values.len() {
                0 => vec![0],
                1 => operand.immediate_values.to_vec(),
                _ => comps
                    .iter()
                    .filter_map(|lane| operand.immediate_values.get(usize::from(*lane)).copied())
                    .collect(),
            };
            let bits = match bits.is_empty() {
                true => operand.immediate_values.to_vec(),
                false => bits,
            };
            // Constants start out as floats because that is what most of them are; anything reading
            // one as bits reinterprets it rather than casting.
            return Sourced {
                expr: self.modified(
                    operand,
                    Expr::Literal {
                        bits,
                        domain: Domain::Float,
                    },
                ),
                domain: Domain::Float,
                reads,
                owner: None,
            };
        }

        let index = immediate(operand).unwrap_or(0);
        let base = match operand.reg_type {
            RegisterType::Temp => self.spelling(index),
            RegisterType::Input => {
                if let Some(entry) = self.names.inputs.get(&index) {
                    let expr = Expr::Read {
                        base: entry.name.clone(),
                        swizzle: comps.to_vec(),
                    };
                    return Sourced {
                        expr: self.modified(operand, expr),
                        domain: super::domain(&entry.kind),
                        reads,
                        owner: None,
                    };
                }
                format!("v{index}")
            }
            // Outputs are fields of the one value the shader hands back.
            RegisterType::Output => self.names.outputs.get(&index).map_or_else(
                || format!("output.o{index}"),
                |entry| format!("output.{}", entry.name),
            ),
            // Depth has no register of its own, so the signature files it under none.
            RegisterType::OutputDepth => self.names.outputs.get(&u32::MAX).map_or_else(
                || "output.SV_Depth".to_owned(),
                |entry| format!("output.{}", entry.name),
            ),
            RegisterType::Resource => self.names.texture(index as u16),
            RegisterType::Sampler => self.names.sampler(index as u16),
            RegisterType::ConstantBuffer => {
                let slot = index as u16;
                // A buffer of structs is indexed by element and then by register within it, so a
                // run-time index that is a whole number of elements plus a register inside one can
                // still reach a field by name. Anything else about the buffer leaves it as registers.
                let step = self.element(slot);
                let inside = match (step, operand.indices.get(1)) {
                    (Some(step), Some(OperandIndex::RelativePlusImm(offset, held)))
                        if *offset < step =>
                    {
                        Some((held.clone(), *offset, step))
                    }
                    (Some(step), Some(OperandIndex::Relative(held))) => {
                        Some((held.clone(), 0, step))
                    }
                    _ => None,
                };
                let name = self
                    .names
                    .constants
                    .get(&slot)
                    .map_or_else(|| format!("cb{index}"), |buffer| buffer.name.clone());
                let expr = match inside {
                    Some((held, offset, step)) => {
                        let mut sourced = self.read(&held, &[0]);
                        self.strided(slot, sourced.owner, step);
                        reads.append(&mut sourced.reads);
                        let by_element = self.arrays.contains(&slot);
                        let element = coerce(sourced.expr, sourced.domain, Domain::Int);
                        let element = match by_element {
                            // The value is the element itself, since it was only ever used to pick
                            // one.
                            true => element.text(),
                            false => format!("{} + {offset}", element.text()),
                        };
                        let parts = by_element
                            .then(|| self.names.element(slot, &element, offset, comps))
                            .flatten();
                        // Whether the whole buffer can read this way is settled once every read has
                        // been seen, so each of them says only whether it could itself.
                        let held = self.reachable.entry(slot).or_insert(true);
                        *held &= sourced.owner.is_some();
                        match parts {
                            Some(parts) => gathered(parts),
                            None => Expr::Read {
                                base: format!("{name}[{element}]"),
                                swizzle: comps.to_vec(),
                            },
                        }
                    }
                    None => {
                        let (register, text, mut inner) = self.index(operand.indices.get(1));
                        reads.append(&mut inner);
                        if self.computed.contains(&slot) {
                            self.reachable.insert(slot, false);
                        }
                        match register.filter(|_| !self.computed.contains(&slot)) {
                            // Picked at run time, so the buffer is an array of registers and nothing
                            // names the one being read.
                            None => Expr::Read {
                                base: format!("{name}[{text}]"),
                                swizzle: comps.to_vec(),
                            },
                            Some(register) => match self.names.constant(slot, register, comps) {
                                Some(parts) => gathered(parts),
                                None => Expr::Read {
                                    base: format!("{name}[{register}]"),
                                    swizzle: comps.to_vec(),
                                },
                            },
                        }
                    }
                };
                return Sourced {
                    expr: self.modified(operand, expr),
                    domain: Domain::Float,
                    reads,
                    owner: None,
                };
            }
            RegisterType::ImmConstBuffer => {
                let (_, text, mut inner) = self.index(operand.indices.first());
                reads.append(&mut inner);
                format!("icb[{text}]")
            }
            RegisterType::IndexableTemp => {
                let (_, text, mut inner) = self.index(operand.indices.get(1));
                reads.append(&mut inner);
                // The component read is not known, so the whole array stands in for it and any write
                // to one ends every value that read it.
                reads.push((operand.reg_type.to_u32(), index, 0));
                format!("x{index}[{text}]")
            }
            other => match operand.indices.is_empty() {
                true => other.prefix().to_owned(),
                false => format!("{}{index}", other.prefix()),
            },
        };

        let swizzle = match operand.reg_type {
            RegisterType::Sampler | RegisterType::Resource => Vec::new(),
            _ => comps.to_vec(),
        };
        let expr = self.modified(operand, Expr::Read { base, swizzle });
        Sourced {
            expr,
            domain: Domain::Float,
            reads,
            owner: None,
        }
    }

    /// What a temporary register is called at this point in the body.
    fn spelling(&self, index: u32) -> String {
        match self.versions.get(&(RegisterType::Temp.to_u32(), index)) {
            None | Some(0) => format!("r{index}"),
            Some(version) => format!("r{index}_{}", version + 1),
        }
    }

    /// Which buffers can be read as arrays of structs, and which values hold an element of one.
    ///
    /// A value can be the element rather than the register it starts at only where picking an element
    /// is all it was ever used for — otherwise it still has to hold what the machine put in it. And a
    /// buffer can be read that way only where every one of its run-time reads came out that way, since
    /// it is declared once and a register array and a struct array are not the same declaration.
    fn reading_arrays(&mut self) {
        let elemental: Vec<u32> = self
            .defs
            .iter()
            .zip(&self.usage)
            .map(|(def, usage)| {
                let Some(step) = usage.stride else { return 0 };
                if usage.indexed == 0 || usage.indexed != usage.reads {
                    return 0;
                }
                match without_stride(def.expr.clone(), step) == def.expr {
                    true => 0,
                    false => step,
                }
            })
            .collect();
        self.arrays = self
            .reachable
            .iter()
            .filter(|(_, reached)| **reached)
            .map(|(slot, _)| *slot)
            .filter(|slot| self.element(*slot).is_some())
            .filter(|slot| {
                self.indexers.get(slot).is_some_and(|held| {
                    !held.is_empty() && held.iter().all(|id| elemental[*id] > 0)
                })
            })
            .collect();
        self.elemental = elemental;
    }

    /// Registers one element of a buffer takes, where the buffer holds more than one element of a
    /// struct whose fields are known. Nought where it does not, which is every buffer that is really
    /// an array of registers.
    fn element(&self, slot: u16) -> Option<u32> {
        let held = self.names.constants.get(&slot)?;
        let step = held.span();
        let span = *self.spans.get(&slot)?;
        (step > 0 && span > step && span % step == 0).then_some(step)
    }

    /// Note that a value was read as a register index into a buffer whose elements are structs of
    /// `stride` registers. A value every one of whose reads is that can be the element instead.
    fn strided(&mut self, slot: u16, owner: Option<usize>, stride: u32) {
        let Some(id) = owner else { return };
        self.indexers.entry(slot).or_default().push(id);
        let held = &mut self.usage[id];
        held.indexed += 1;
        held.stride = match held.stride {
            None if held.indexed == 1 => Some(stride),
            Some(held) if held == stride => Some(stride),
            _ => None,
        };
    }

    /// A register index, which may itself be computed.
    fn index(&mut self, index: Option<&OperandIndex>) -> (Option<u32>, String, Vec<Slot>) {
        match index {
            Some(OperandIndex::Imm32(value)) => (Some(*value), value.to_string(), Vec::new()),
            Some(OperandIndex::Imm64(value)) => (None, value.to_string(), Vec::new()),
            Some(OperandIndex::Relative(operand)) => {
                let sourced = self.read(operand, &[0]);
                let expr = coerce(sourced.expr, sourced.domain, Domain::Int);
                (None, expr.text(), sourced.reads)
            }
            Some(OperandIndex::RelativePlusImm(offset, operand)) => {
                let sourced = self.read(operand, &[0]);
                let expr = coerce(sourced.expr, sourced.domain, Domain::Int);
                (None, format!("{} + {offset}", expr.text()), sourced.reads)
            }
            None => (None, "0".to_owned(), Vec::new()),
        }
    }

    /// The negate and absolute-value modifiers an operand carries.
    fn modified(&self, operand: &Operand, expr: Expr) -> Expr {
        let width = expr.width();
        let expr = match operand.abs {
            true => call("abs", vec![expr], width),
            false => expr,
        };
        match operand.negate {
            true => Expr::Unary {
                op: "-",
                value: Box::new(expr),
            },
            false => expr,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn def(register: u32, lanes: &[u8], block: usize) -> Def {
        Def {
            base: String::new(),
            key: (RegisterType::Temp.to_u32(), register),
            lanes: lanes.to_vec(),
            expr: Expr::Literal {
                bits: vec![0],
                domain: Domain::Float,
            },
            domain: Domain::Float,
            reads: Vec::new(),
            block,
            fixed: false,
            local: false,
            looped: false,
        }
    }

    fn touch(register: u32, lanes: &[u8], at: usize) -> Vec<Touch> {
        lanes
            .iter()
            .map(|lane| ((RegisterType::Temp.to_u32(), register), *lane, at))
            .collect()
    }

    fn decided(defs: &[Def], touches: &[Touch]) -> Vec<bool> {
        let none = vec![false; defs.len()];
        splits(defs, touches, &none, &none)
    }

    /// The second value in a register is a different value, and reads as one.
    #[test]
    fn a_register_written_over_starts_again() {
        let defs = [def(0, &[0, 1, 2, 3], 0), def(0, &[0, 1, 2, 3], 0)];
        let touches = [touch(0, &[0], 1), touch(0, &[0], 2)].concat();
        assert_eq!(decided(&defs, &touches), [false, true]);
    }

    /// A component the run leaves alone still holds what was put there, and only the register's own
    /// name reaches it.
    #[test]
    fn what_the_run_does_not_write_keeps_the_name() {
        let defs = [def(0, &[0, 1, 2, 3], 0), def(0, &[0, 1], 0)];
        let touches = [touch(0, &[0], 1), touch(0, &[0, 2], 2)].concat();
        assert_eq!(decided(&defs, &touches), [false, false]);
    }

    /// Two writes finish the register between them, which is a fresh start at the first of them.
    #[test]
    fn a_run_of_writes_covers_between_them() {
        let defs = [
            def(0, &[0, 1, 2, 3], 0),
            def(0, &[0, 1], 0),
            def(0, &[2, 3], 0),
        ];
        let touches = [touch(0, &[0], 1), touch(0, &[0, 2], 3)].concat();
        assert_eq!(decided(&defs, &touches), [false, true, false]);
    }

    /// A read partway through the run is a read of the value the run is replacing.
    #[test]
    fn a_read_inside_the_run_stops_it() {
        let defs = [
            def(0, &[0, 1, 2, 3], 0),
            def(0, &[0, 1], 0),
            def(0, &[2, 3], 0),
        ];
        let touches = [touch(0, &[0], 1), touch(0, &[2], 2), touch(0, &[0, 2], 3)].concat();
        assert_eq!(decided(&defs, &touches), [false, false, false]);
    }

    /// A write under a branch does not happen on the way past it, so the value before it is still
    /// what a later read can find, and both of them answer to the register.
    #[test]
    fn a_write_inside_a_branch_is_not_a_fresh_start() {
        let defs = [def(0, &[0, 1, 2, 3], 0), def(0, &[0, 1, 2, 3], 1)];
        let touches = [touch(0, &[0], 1), touch(0, &[0], 2)].concat();
        assert_eq!(decided(&defs, &touches), [false, false]);
    }

    /// The value an instruction reads on its way to writing the register is the one being replaced,
    /// which is read before the new name takes over.
    #[test]
    fn reading_the_register_it_writes_does_not_stop_it() {
        let defs = [def(0, &[0, 1, 2, 3], 0), def(0, &[0, 1, 2, 3], 0)];
        let touches = [touch(0, &[0], 1), touch(0, &[0], 1), touch(0, &[0], 2)].concat();
        assert_eq!(decided(&defs, &touches), [false, true]);
    }

    /// A value written down under its own name never reached the register, so there is nothing
    /// there to start over from.
    #[test]
    fn a_value_with_a_name_of_its_own_is_not_a_fresh_start() {
        let defs = [def(0, &[0, 1, 2, 3], 0), def(0, &[0, 1, 2, 3], 0)];
        let touches = [touch(0, &[0], 1), touch(0, &[0], 2)].concat();
        let localised = [false, true];
        assert_eq!(
            splits(&defs, &touches, &[false, false], &localised),
            [false, false]
        );
    }

    /// A value written out again at each of its readers has to last until the last of them. These are
    /// the boundaries of that window, which the corpus does not happen to exercise.
    #[test]
    fn a_value_lasts_until_its_last_reader() {
        let slot = |register: u32, lane: u8| (RegisterType::Temp.to_u32(), register, lane);
        let mut held = def(9, &[0], 0);
        held.reads = vec![slot(1, 0)];
        // Nothing in between leaves it as it was.
        let clear = [held, def(2, &[0], 0), def(3, &[0], 0)];
        assert!(survives(&clear, &clear[0], 0, 2));
        // A write to what it reads, in between, ends it.
        let mut held = def(9, &[0], 0);
        held.reads = vec![slot(1, 0)];
        let over = [held, def(1, &[0], 0), def(3, &[0], 0)];
        assert!(!survives(&over, &over[0], 0, 2));
        // The reader's own write lands after it has read, so the reader itself does not end it.
        assert!(survives(&over, &over[0], 0, 1));
        // Neither does its own write, which is where it came from.
        let mut held = def(1, &[0], 0);
        held.reads = vec![slot(1, 0)];
        let itself = [held, def(3, &[0], 0)];
        assert!(survives(&itself, &itself[0], 0, 1));
        // Nothing read it at all, which the caller rules out before asking.
        assert!(survives(&over, &over[0], 0, 0));
    }

    /// Which element of an array a read took is not known, so a write anywhere in one ends every
    /// value that read from it.
    #[test]
    fn a_write_to_an_array_ends_everything_that_read_one() {
        let arrays = RegisterType::IndexableTemp.to_u32();
        let mut held = def(9, &[0], 0);
        held.reads = vec![(arrays, 0, 0)];
        let mut writer = def(0, &[0], 0);
        writer.key = (arrays, 7);
        let over = [held, writer, def(3, &[0], 0)];
        assert!(!survives(&over, &over[0], 0, 2));
    }
}
