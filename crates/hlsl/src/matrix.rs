//! Transform recovery — matrix multiplies and constructors, out of their unrolled pieces.
//!
//! Three things arrive here in pieces, since the machine has no instruction for a transform.
//! Sometimes it is one dot product per row of the result, each landing in its own component, and
//! sometimes it is each row of the matrix scaled by the component of the vector that picks it out,
//! summed in one expression. Either way, four of them against the same vector are the most
//! recognisable thing in a vertex shader, and saying so is the difference between reading a
//! transform and reading arithmetic. And a register filled a few components at a time is one value,
//! which a constructor says in a line.
//!
//! Rows are assembled with a constructor rather than by naming the matrix, except where the run
//! covers a declared matrix from its first row: a constructor takes its arguments in source order
//! whatever the packing, whereas naming the matrix means `mul` depends on how it was declared, and
//! picking wrong transposes the transform silently.
//!
//! Both read the same record of what each line assigns, the multiplies first, so that a transform
//! landing in three components can still join whatever fills the fourth.

use super::Names;
use super::expr::{Expr, call, lane, letters};

/// One emitted assignment, kept beside its text so a run of them can be recognised.
#[derive(Clone)]
pub struct Emitted {
    /// Block nesting it was written at, since a run has to sit in one block.
    pub depth: usize,
    /// The register it was assigned to, under the name it goes by from that line on.
    pub base: String,
    /// Components of `base` it wrote.
    pub lanes: Vec<u8>,
    /// The value assigned, which is what a run is recognised from.
    pub expr: Expr,
}

/// Shortest run worth folding. Two dot products are as clear written out, and a matrix of two rows
/// is rare enough that calling one would mislead more than it helped.
const SHORTEST: usize = 3;

/// A row of a matrix: where it is read from, which element of that, and how much of it is taken.
fn row(expr: &Expr) -> Option<(&str, &str, &[u8])> {
    let Expr::Read { base, swizzle } = expr else {
        return None;
    };
    // A row of a declared matrix arrives bracketed, since components cannot be taken off one
    // directly.
    let base = base.strip_prefix('(').unwrap_or(base);
    let base = base.strip_suffix(')').unwrap_or(base);
    // The last index is the row: an array of matrices carries the element in front of it, and that
    // element is what the row belongs to.
    let open = base.rfind('[')?;
    let inner = base.strip_suffix(']')?;
    Some((&base[..open], &inner[open + 1..], swizzle))
}

/// The terms of a sum, however it was bracketed.
fn terms<'a>(expr: &'a Expr, into: &mut Vec<&'a Expr>) {
    match expr {
        Expr::Binary {
            op: "+",
            left,
            right,
        } => {
            terms(left, into);
            terms(right, into);
        }
        held => into.push(held),
    }
}

/// One term of a weighted sum of rows: what does the weighting, which component of it, and the row
/// that weight applies to.
fn weight(expr: &Expr) -> Option<(&str, u8, &Expr)> {
    let Expr::Binary {
        op: "*",
        left,
        right,
    } = expr
    else {
        return None;
    };
    for (splat, held) in [(left, right), (right, left)] {
        let Expr::Read { base, swizzle } = &**splat else {
            continue;
        };
        // One component, however many times it is spelled, or it weights nothing in particular.
        let Some((first, rest)) = swizzle.split_first() else {
            continue;
        };
        if rest.iter().any(|lane| lane != first) || row(held).is_none() {
            continue;
        }
        return Some((base, *first, held));
    }
    None
}

/// A vector transformed by a matrix, which the machine writes as a weighted sum of the matrix's
/// rows: each row is scaled by one component of the vector, and the sum of those is `mul(v, M)`.
/// There is no instruction for the whole of it, so this is the shape a transform comes back in, and
/// it is the longest line in most vertex shaders. What the sum holds besides the rows — the
/// translation of an affine transform — is added to the multiply the same way it was added to the
/// rows.
pub fn transform(names: &Names, expr: Expr) -> Expr {
    summed(names, &expr).unwrap_or(expr)
}

fn summed(names: &Names, expr: &Expr) -> Option<Expr> {
    let mut parts = Vec::new();
    terms(expr, &mut parts);

    let mut weights = Vec::with_capacity(parts.len());
    let mut indices = Vec::with_capacity(parts.len());
    let mut rest = Vec::new();
    let mut matrix = None;
    let mut columns = None;
    for part in parts {
        // One matrix, read the same way each time, or these are separate multiplies standing side
        // by side rather than one transform.
        let held = weight(part)
            .and_then(|(base, lane, held)| Some((base, lane, held, row(held)?)))
            .filter(|(_, _, _, (name, _, swizzle))| {
                *matrix.get_or_insert(*name) == *name
                    && *columns.get_or_insert(*swizzle) == *swizzle
            });
        match held {
            Some((base, lane, held, (_, index, _))) => {
                weights.push((base, lane, held));
                indices.push(index);
            }
            None => rest.push(part),
        }
    }
    if weights.len() < SHORTEST {
        return None;
    }
    // In the order the matrix has its rows, which is the order `mul` takes them in whatever order
    // the compiler wrote them.
    let order = consecutive(&indices)?;
    let rows = order.len();
    let width = match columns? {
        [] => 4,
        held => held.len(),
    };

    // Naming the matrix is only safe where the rows are its own, from the first, in the order it
    // has them: a constructor takes its arguments in the order written whatever the packing says,
    // whereas what `mul` does with a named matrix depends on how that matrix was declared.
    let matrix = matrix?;
    // The first row of the matrix, not the first one written: the compiler emits them in whatever
    // order suits it, and the rows being consecutive is already settled.
    let own = indices[order[0]] == "0";
    let value = match declared(names, matrix).filter(|_| own) {
        Some(held) if held as usize == rows && width == 4 => Expr::Read {
            base: matrix.to_owned(),
            swizzle: Vec::new(),
        },
        // Less than the whole of it is a cast, which drops the rows and columns past the run.
        Some(held) if held as usize >= rows => Expr::Read {
            base: format!("(float{rows}x{width}){matrix}"),
            swizzle: Vec::new(),
        },
        _ => call(
            &format!("float{rows}x{width}"),
            order.iter().map(|at| weights[*at].2.clone()).collect(),
            width,
        ),
    };

    // The weights are the vector being transformed. Components of one value spell it; anything else
    // is gathered into one, since what the machine kept apart is still a vector.
    let taken = match weights.iter().all(|(base, ..)| *base == weights[0].0) {
        true => Expr::Read {
            base: weights[0].0.to_owned(),
            swizzle: order.iter().map(|at| weights[*at].1).collect(),
        },
        false => call(
            &format!("float{rows}"),
            order
                .iter()
                .map(|at| Expr::Read {
                    base: weights[*at].0.to_owned(),
                    swizzle: vec![weights[*at].1],
                })
                .collect(),
            rows,
        ),
    };
    let held = call("mul", vec![taken, value], width);
    Some(rest.into_iter().fold(held, |sum, part| Expr::Binary {
        op: "+",
        left: Box::new(sum),
        right: Box::new(part.clone()),
    }))
}

/// Replace the dot product inside a lane's expression with a read of where the multiply put it.
fn without_product(expr: &Expr, read: &Expr) -> Expr {
    if product(expr).is_some() {
        return read.clone();
    }
    match expr {
        Expr::Unary { op, value } => Expr::Unary {
            op,
            value: Box::new(without_product(value, read)),
        },
        Expr::Binary { op, left, right } => Expr::Binary {
            op,
            left: Box::new(without_product(left, read)),
            right: Box::new(without_product(right, read)),
        },
        Expr::Call { name, args, width } => Expr::Call {
            name: name.clone(),
            args: args.iter().map(|arg| without_product(arg, read)).collect(),
            width: *width,
        },
        Expr::Select { cond, then, els } => Expr::Select {
            cond: Box::new(without_product(cond, read)),
            then: Box::new(without_product(then, read)),
            els: Box::new(without_product(els, read)),
        },
        other => other.clone(),
    }
}

/// The dot product a lane is built on, whether it is the whole of the lane or sits inside something
/// else. A lane that holds more than one is not part of a single multiply.
fn inner(expr: &Expr) -> Option<(&str, &str, &[u8], &Expr)> {
    if let Some(held) = product(expr) {
        return Some(held);
    }
    let Expr::Call { args, .. } = expr else {
        return None;
    };
    let mut found = args.iter().filter_map(product);
    match (found.next(), found.next()) {
        (Some(held), None) => Some(held),
        _ => None,
    }
}

/// A dot product of a matrix row against a vector.
fn product(expr: &Expr) -> Option<(&str, &str, &[u8], &Expr)> {
    let Expr::Call { name, args, .. } = expr else {
        return None;
    };
    if name != "dot" || args.len() != 2 {
        return None;
    }
    // A row taken in part is a narrower matrix, not a different kind of thing: three rows of three
    // components are a `float3x3`, which is how a transform that leaves the translation alone comes
    // back. What the run has to agree on is that every row is taken the same way.
    let (matrix, index, swizzle) = row(&args[0])?;
    Some((matrix, index, swizzle, &args[1]))
}

/// An index split into where it starts and how far past that it reaches: `3` is nowhere plus three,
/// `r0.x + 2` is `r0.x` plus two. Rows of one matrix share a start and differ only in the offset.
fn offset(index: &str) -> (&str, u32) {
    if let Ok(step) = index.parse() {
        return ("", step);
    }
    let Some((base, step)) = index.rsplit_once(" + ") else {
        return (index, 0);
    };
    match step.parse() {
        Ok(step) => (base, step),
        Err(_) => (index, 0),
    }
}

/// Whether the indices name consecutive rows, whatever order they were written in. The compiler
/// emits the components in whatever order suits it, so this sorts them rather than requiring one.
fn consecutive(indices: &[&str]) -> Option<Vec<usize>> {
    let (start, _) = offset(indices.first()?);
    let mut steps: Vec<(u32, usize)> = Vec::with_capacity(indices.len());
    for (at, index) in indices.iter().enumerate() {
        let (base, step) = offset(index);
        if base != start {
            return None;
        }
        steps.push((step, at));
    }
    steps.sort_unstable();
    let first = steps[0].0;
    match steps
        .iter()
        .enumerate()
        .all(|(step, (held, _))| *held == first + step as u32)
    {
        true => Some(steps.into_iter().map(|(_, at)| at).collect()),
        false => None,
    }
}

/// Whether a name is, or is subscripted by, a register. What indexes a name is written into it
/// rather than kept beside it, so a register read there shows up in the text and nowhere else.
fn mentions(name: &str, base: &str) -> bool {
    let word = |char: Option<char>| char.is_some_and(|char| char.is_alphanumeric() || char == '_');
    let mut rest = name;
    while let Some(at) = rest.find(base) {
        let (before, after) = rest.split_at(at);
        let after = &after[base.len()..];
        if !word(before.chars().next_back()) && !word(after.chars().next()) {
            return true;
        }
        rest = after;
    }
    false
}

/// Whether the value being transformed reads the register the result is going into. Where it does,
/// each line sees what the one before it wrote, and folding them would have every row read the
/// original instead.
fn reads(expr: &Expr, base: &str) -> bool {
    match expr {
        Expr::Read { base: held, .. } => mentions(held, base),
        Expr::Literal { .. } => false,
        Expr::Unary { value, .. } | Expr::Swizzle { value, .. } => reads(value, base),
        Expr::Binary { left, right, .. } => reads(left, base) || reads(right, base),
        Expr::Call { args, .. } => args.iter().any(|arg| reads(arg, base)),
        Expr::Select { cond, then, els } => {
            reads(cond, base) || reads(then, base) || reads(els, base)
        }
        Expr::Vector(parts) => parts.iter().any(|part| reads(part, base)),
    }
}

/// The rows of a declared matrix, where the base names one. Telling a run that covers a whole matrix
/// from one taking part of it is what says whether the matrix can be named rather than rebuilt.
fn declared<'a>(names: &Names, base: &'a str) -> Option<u32> {
    // A buffer holding nothing but the matrix is declared without a struct, so the matrix stands
    // under the buffer's own name with nothing to reach through.
    let (buffer, member) = base.split_once('.').unwrap_or((base, base));
    // An array of matrices carries the element in the name; it is the matrix that is declared.
    let held = |name: &'a str| name.split_once('[').map_or(name, |(before, _)| before);
    names
        .constants
        .values()
        .find(|inner| inner.name == held(buffer))?
        .rows(held(member))
}

/// A run of assignments recognised as one multiply.
struct Fold<'a> {
    /// How many lines it covers.
    length: usize,
    /// The rows it reads, in row order.
    rows: Vec<&'a str>,
    /// Where each of those rows was written, as an index into the run.
    order: Vec<usize>,
    matrix: &'a str,
    vector: &'a Expr,
    /// Components of each row the run takes, which is the matrix's width.
    width: usize,
}

/// The run of assignments starting at `at` that together make one multiply, and what it multiplies.
fn run(lines: &[Option<Emitted>], at: usize) -> Option<Fold<'_>> {
    let first = lines.get(at)?.as_ref()?;
    let (matrix, _, columns, vector) = inner(&first.expr)?;
    let mut indices = Vec::new();
    let mut lanes: Vec<u8> = Vec::new();
    let mut end = at;

    while let Some(Some(held)) = lines.get(end) {
        let [lane] = held.lanes[..] else { break };
        if held.depth != first.depth || held.base != first.base || lanes.contains(&lane) {
            break;
        }
        let Some((held_matrix, index, held_columns, held_vector)) = inner(&held.expr) else {
            break;
        };
        // The same matrix, taken the same way, against the same vector, or these are separate
        // operations that happen to stand next to each other.
        if held_matrix != matrix || held_vector != vector || held_columns != columns {
            break;
        }
        indices.push(index);
        lanes.push(lane);
        end += 1;
    }

    if lanes.len() < SHORTEST || reads(vector, &first.base) {
        return None;
    }
    // In row order, which is the order the multiply produces its components in.
    let order = consecutive(&indices)?;
    Some(Fold {
        length: end - at,
        rows: order.iter().map(|at| indices[*at]).collect(),
        order,
        matrix,
        vector,
        width: match columns {
            [] => 4,
            held => held.len(),
        },
    })
}

/// Rewrite each run of dot products into the multiply it is, blanking the lines it replaces. The
/// assignments come back rewritten alongside, since a multiply is a value like any other and what
/// comes after may still have something to do with it.
pub fn fold(
    names: &Names,
    out: &mut Vec<String>,
    lines: &[Option<Emitted>],
) -> Vec<Option<Emitted>> {
    let mut records = lines.to_vec();
    let mut drop = vec![false; out.len()];
    let mut follow: Vec<(usize, Vec<String>)> = Vec::new();
    let mut at = 0;
    while at < lines.len() {
        let Some(Fold {
            length,
            rows: indices,
            order,
            matrix,
            vector,
            width,
        }) = run(lines, at)
        else {
            at += 1;
            continue;
        };
        let held = lines[at].as_ref().expect("the run starts on an assignment");
        // The component each row lands in, taken in row order rather than the order written.
        let lanes: Vec<u8> = order
            .iter()
            .filter_map(|position| lines[at + position].as_ref()?.lanes.first().copied())
            .collect();

        // Rows taken in part need the brackets a read of one already carries, so the row is spelled
        // the way the dot product spelled it rather than rebuilt.
        let rows: Vec<Expr> = indices
            .iter()
            .map(|index| Expr::Read {
                base: match width {
                    4 => format!("{matrix}[{index}]"),
                    _ => format!("({matrix}[{index}])"),
                },
                swizzle: match width {
                    4 => Vec::new(),
                    held => (0..held as u8).collect(),
                },
            })
            .collect();
        // Naming the matrix is only safe from its first row, and only where it is declared as one:
        // a constructor reads its rows in the order written whatever the packing says, whereas what
        // `mul` does with a named matrix depends on how that matrix was declared. Anything less than
        // the whole of it is a cast, which drops the rows and columns past the run.
        let whole = declared(names, matrix).filter(|_| indices.first() == Some(&"0"));
        let value = match whole {
            Some(rows) if rows as usize == length && width == 4 => Expr::Read {
                base: matrix.to_owned(),
                swizzle: Vec::new(),
            },
            Some(rows) if rows as usize >= length => Expr::Read {
                base: format!("(float{length}x{width}){matrix}"),
                swizzle: Vec::new(),
            },
            _ => call(&format!("float{length}x{width}"), rows, length),
        };
        let transform = call("mul", vec![value, vector.clone()], length);

        // Every lane in order is the whole register, which needs no swizzle to say so.
        let target = match lanes[..] {
            [0, 1, 2, 3] => held.base.clone(),
            _ => format!("{}.{}", held.base, letters(&lanes)),
        };
        out[at] = format!(
            "{}{target} = {};",
            "    ".repeat(held.depth),
            transform.text()
        );
        // The rows land in the order the matrix has them, which is not always the order of the
        // components they go to. Only where it is does the multiply stand for the components end to
        // end, which is the shape anything reading this back can build on.
        records[at] = lanes
            .windows(2)
            .all(|pair| pair[0] < pair[1])
            .then(|| Emitted {
                depth: held.depth,
                base: held.base.clone(),
                lanes: lanes.clone(),
                expr: transform.clone(),
            });
        for (line, record) in drop
            .iter_mut()
            .zip(records.iter_mut())
            .skip(at + 1)
            .take(length - 1)
        {
            *line = true;
            *record = None;
        }

        // A lane that wrapped its dot product in something else keeps that, applied to where the
        // multiply has just put the value. It cannot go inside the multiply, and the lanes are
        // independent, so following it is the same as doing it in place.
        let mut extra = Vec::new();
        for (position, held_at) in order.iter().enumerate() {
            let Some(line) = lines[at + held_at].as_ref() else {
                continue;
            };
            if product(&line.expr).is_some() {
                continue;
            }
            let letter = lane(lanes[position]);
            let read = Expr::Read {
                base: format!("{}.{letter}", held.base),
                swizzle: Vec::new(),
            };
            extra.push(format!(
                "{}{}.{letter} = {};",
                "    ".repeat(held.depth),
                held.base,
                without_product(&line.expr, &read).text()
            ));
        }
        follow.push((at, extra));
        at += length;
    }

    let mut keep = drop.iter();
    out.retain(|_| !keep.next().copied().unwrap_or(false));
    let mut keep = drop.iter();
    records.retain(|_| !keep.next().copied().unwrap_or(false));

    // Back to front, so that inserting does not move the places still to be filled.
    let mut shift: Vec<usize> = drop
        .iter()
        .scan(0usize, |kept, dropped| {
            let at = *kept;
            if !dropped {
                *kept += 1;
            }
            Some(at)
        })
        .collect();
    shift.push(out.len());
    for (at, extra) in follow.into_iter().rev() {
        let after = shift.get(at).copied().unwrap_or(out.len()) + 1;
        for line in extra.into_iter().rev() {
            let place = after.min(out.len());
            out.insert(place, line);
            records.insert(place, None);
        }
    }
    records
}

/// How far apart the writes filling one register may sit. A value assembled over more lines than
/// this was not going to read as one thing anyway, and every line in between is another that has to
/// be proved harmless.
const REACH: usize = 12;

/// The components a write covers, as a mask, where they are ones it can hold.
fn mask(lanes: &[u8]) -> Option<u8> {
    let mut held = 0u8;
    for lane in lanes {
        if *lane > 3 || held & (1 << lane) != 0 {
            return None;
        }
        held |= 1 << lane;
    }
    (held != 0).then_some(held)
}

/// The writes that fill one register between them, in the order their components sit in it.
fn whole(lines: &[Option<Emitted>], at: usize) -> Option<(Vec<usize>, usize)> {
    let first = lines[at].as_ref()?;
    let (base, depth) = (&first.base, first.depth);
    let mut covered = mask(&first.lanes)?;
    let mut members = vec![at];

    for pos in at + 1..(at + REACH).min(lines.len()) {
        let held = lines[pos].as_ref()?;
        if &held.base != base {
            // Everything in between ends up after the writes still waiting to join it, so it must
            // not read what they have already put in the register, and they must not read what it
            // leaves somewhere else.
            let mut moved = members.iter().filter_map(|line| lines[*line].as_ref());
            if held.depth != depth
                || reads(&held.expr, base)
                || moved.any(|held_at| reads(&held_at.expr, &held.base))
            {
                return None;
            }
            continue;
        }
        let lanes = mask(&held.lanes)?;
        if held.depth != depth || lanes & covered != 0 {
            return None;
        }
        covered |= lanes;
        members.push(pos);
        if covered == 0xF {
            return Some((ordered(lines, base, members)?, pos));
        }
    }
    None
}

/// The writes laid out in component order, where they really do lay the register out end to end.
///
/// A constructor takes its arguments in that order and no other, so a write covering components that
/// are not next to each other cannot be one of them. Nor can a write that reads the register it is
/// filling, since after the fold it would see what the register held before any of this.
fn ordered(lines: &[Option<Emitted>], base: &str, members: Vec<usize>) -> Option<Vec<usize>> {
    if members.len() < 2 {
        return None;
    }
    let held = |line: &usize| {
        lines[*line]
            .as_ref()
            .expect("every member assigns something")
    };
    let mut ordered = members;
    ordered.sort_by_key(|line| held(line).lanes.first().copied());

    let laid: Vec<u8> = ordered
        .iter()
        .flat_map(|line| held(line).lanes.clone())
        .collect();
    // A value narrower than the components it is written to is spread over them, which is not what a
    // constructor does with an argument.
    let sized = ordered
        .iter()
        .all(|line| held(line).expr.components() == held(line).lanes.len());
    let owned = ordered.iter().all(|line| !reads(&held(line).expr, base));
    (laid == [0, 1, 2, 3] && sized && owned).then_some(ordered)
}

/// Rewrite the writes that fill a register between them into the one value they build.
pub fn coalesce(out: &mut Vec<String>, lines: &[Option<Emitted>]) {
    let mut drop = vec![false; out.len()];
    let mut at = 0;
    while at < lines.len() {
        let Some((group, last)) = whole(lines, at) else {
            at += 1;
            continue;
        };
        let held = lines[last]
            .as_ref()
            .expect("the last member assigns something");
        let parts: Vec<Expr> = group
            .iter()
            .map(|line| {
                lines[*line]
                    .as_ref()
                    .expect("every member assigns")
                    .expr
                    .clone()
            })
            .collect();
        out[last] = format!(
            "{}{} = {};",
            "    ".repeat(held.depth),
            held.base,
            call("float4", parts, 4).text()
        );
        for line in group.into_iter().filter(|line| *line != last) {
            drop[line] = true;
        }
        at = last + 1;
    }

    let mut keep = drop.iter();
    out.retain(|_| !keep.next().copied().unwrap_or(false));
}

#[cfg(test)]
mod test {
    use super::*;

    fn dot(matrix: &str, index: &str, vector: &str) -> Expr {
        call(
            "dot",
            vec![
                Expr::Read {
                    base: format!("{matrix}[{index}]"),
                    swizzle: Vec::new(),
                },
                Expr::Read {
                    base: vector.to_owned(),
                    swizzle: Vec::new(),
                },
            ],
            1,
        )
    }

    fn assignment(base: &str, lane: u8, expr: Expr) -> Option<Emitted> {
        Some(Emitted {
            depth: 1,
            base: base.to_owned(),
            lanes: vec![lane],
            expr,
        })
    }

    fn folded(lines: Vec<Option<Emitted>>) -> Vec<String> {
        let mut out: Vec<String> = lines.iter().map(|_| "    original;".to_owned()).collect();
        fold(&Names::default(), &mut out, &lines);
        out
    }

    fn read(base: &str, swizzle: Vec<u8>) -> Expr {
        Expr::Read {
            base: base.to_owned(),
            swizzle,
        }
    }

    fn part(base: &str, lanes: Vec<u8>, expr: Expr) -> Option<Emitted> {
        Some(Emitted {
            depth: 1,
            base: base.to_owned(),
            lanes,
            expr,
        })
    }

    fn joined(lines: Vec<Option<Emitted>>) -> Vec<String> {
        let mut out: Vec<String> = (0..lines.len())
            .map(|at| format!("    line {at};"))
            .collect();
        coalesce(&mut out, &lines);
        out
    }

    /// A constructor lays its arguments out in component order, which is not always the order the
    /// components were written in.
    #[test]
    fn writes_filling_a_register_become_the_value_they_build() {
        let out = joined(vec![
            part("r1", vec![3], read("POSITION", vec![3])),
            part("r1", vec![0, 1, 2], read("POSITION", vec![0, 1, 2])),
        ]);
        assert_eq!(out, ["    r1 = float4(POSITION.xyz, POSITION.w);"]);
    }

    /// Anything in between ends up after the writes still waiting to join, so a line reading what
    /// they have already put in the register would go on to see the whole of it.
    #[test]
    fn a_read_of_the_register_in_between_stops_it() {
        let out = joined(vec![
            part("r2", vec![0, 1], read("a", vec![0, 1])),
            part("r0", vec![3], read("r2", vec![1])),
            part("r2", vec![2, 3], read("b", vec![0, 1])),
        ]);
        assert_eq!(out, ["    line 0;", "    line 1;", "    line 2;"]);
    }

    /// What indexes a name is written into it, so a register read there is in the text and nowhere
    /// else. Missing one of those is how a value gets moved past the write it was reading.
    #[test]
    fn a_read_hidden_in_an_index_stops_it_too() {
        let out = joined(vec![
            part("r2", vec![0, 1], read("a", vec![0, 1])),
            part("r0", vec![3], read("table[asint(r2.y)]", vec![1])),
            part("r2", vec![2, 3], read("b", vec![0, 1])),
        ]);
        assert_eq!(out, ["    line 0;", "    line 1;", "    line 2;"]);
    }

    /// A register whose name merely starts the same is a different register.
    #[test]
    fn a_longer_name_is_not_the_register_it_begins_with() {
        let out = joined(vec![
            part("r2", vec![0, 1], read("a", vec![0, 1])),
            part("r0", vec![3], read("r2_xy.x", Vec::new())),
            part("r2", vec![2, 3], read("b", vec![0, 1])),
        ]);
        assert_eq!(out, ["    line 1;", "    r2 = float4(a.xy, b.xy);"]);
    }

    /// One value written to several components is spread over them, which is not what a constructor
    /// does with an argument.
    #[test]
    fn a_value_spread_over_its_components_is_left_alone() {
        let out = joined(vec![
            part(
                "r1",
                vec![0, 1, 2],
                call("dot", vec![read("a", vec![0, 1, 2])], 3),
            ),
            part("r1", vec![3], read("b", vec![0])),
        ]);
        assert_eq!(out, ["    line 0;", "    line 1;"]);
    }

    #[test]
    fn a_row_at_a_time_becomes_one_multiply() {
        let out = folded(vec![
            assignment("o", 0, dot("M", "0", "r1")),
            assignment("o", 1, dot("M", "1", "r1")),
            assignment("o", 2, dot("M", "2", "r1")),
        ]);
        assert_eq!(out, ["    o.xyz = mul(float3x4(M[0], M[1], M[2]), r1);"]);
    }

    /// The compiler writes the components in whatever order suits it, and the destination simply
    /// follows.
    #[test]
    fn the_rows_keep_the_components_they_landed_in() {
        let out = folded(vec![
            assignment("o", 0, dot("M", "0", "r1")),
            assignment("o", 2, dot("M", "1", "r1")),
            assignment("o", 1, dot("M", "2", "r1")),
        ]);
        assert_eq!(out[0], "    o.xzy = mul(float3x4(M[0], M[1], M[2]), r1);");
    }

    /// Where the matrix is placed at run time the rows still step by one, and the index rides along.
    #[test]
    fn a_matrix_found_at_run_time_folds_the_same_way() {
        let out = folded(vec![
            assignment("r2", 0, dot("g_Instancing", "asint(r0.x)", "r1")),
            assignment("r2", 1, dot("g_Instancing", "asint(r0.x) + 1", "r1")),
            assignment("r2", 2, dot("g_Instancing", "asint(r0.x) + 2", "r1")),
        ]);
        assert_eq!(
            out[0],
            "    r2.xyz = mul(float3x4(g_Instancing[asint(r0.x)], \
             g_Instancing[asint(r0.x) + 1], g_Instancing[asint(r0.x) + 2]), r1);"
        );
    }

    /// The whole point of the guard: each line would read what the one before it wrote, so folding
    /// them would have every row see the original vector instead.
    #[test]
    fn a_vector_the_result_overwrites_is_left_alone() {
        let out = folded(vec![
            assignment("r2", 0, dot("M", "0", "r2")),
            assignment("r2", 1, dot("M", "1", "r2")),
            assignment("r2", 2, dot("M", "2", "r2")),
        ]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn rows_that_do_not_step_by_one_are_not_a_matrix() {
        let out = folded(vec![
            assignment("o", 0, dot("M", "0", "r1")),
            assignment("o", 1, dot("M", "2", "r1")),
            assignment("o", 2, dot("M", "4", "r1")),
        ]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn two_rows_are_left_as_they_are() {
        let out = folded(vec![
            assignment("o", 0, dot("M", "0", "r1")),
            assignment("o", 1, dot("M", "1", "r1")),
        ]);
        assert_eq!(out.len(), 2);
    }

    /// A row scaled by one component of a vector, summed over the rows, is a transform.
    fn scaled(vector: &str, lane: u8, matrix: &str, index: &str, columns: Vec<u8>) -> Expr {
        Expr::Binary {
            op: "*",
            left: Box::new(read(vector, vec![lane; columns.len().max(1)])),
            right: Box::new(read(&format!("{matrix}[{index}]"), columns)),
        }
    }

    fn sum(parts: Vec<Expr>) -> Expr {
        parts
            .into_iter()
            .reduce(|left, right| Expr::Binary {
                op: "+",
                left: Box::new(left),
                right: Box::new(right),
            })
            .expect("a sum of something")
    }

    fn matrixed() -> Names {
        let mut names = Names::default();
        names.constants.insert(
            0,
            super::super::Buffer::new(
                "g_B".to_owned(),
                vec![super::super::Field {
                    name: "m_M".to_owned(),
                    kind: "float4x4".to_owned(),
                    register: 0,
                    registers: 4,
                    mask: 0xF,
                }],
            ),
        );
        names
    }

    fn transformed(names: &Names, parts: Vec<Expr>) -> String {
        transform(names, sum(parts)).text()
    }

    /// The machine has no instruction for a transform, so it arrives as each row scaled by the
    /// component that picks it out.
    #[test]
    fn rows_scaled_by_a_vector_are_the_multiply_they_make() {
        let out = transformed(
            &Names::default(),
            vec![
                scaled("v", 0, "M", "0", Vec::new()),
                scaled("v", 1, "M", "1", Vec::new()),
                scaled("v", 2, "M", "2", Vec::new()),
            ],
        );
        assert_eq!(out, "mul(v.xyz, float3x4(M[0], M[1], M[2]))");
    }

    /// The rows arrive in whatever order suited the compiler, and the multiply takes them in the
    /// order the matrix has them.
    #[test]
    fn the_rows_go_in_the_order_the_matrix_has_them() {
        let out = transformed(
            &Names::default(),
            vec![
                scaled("v", 2, "M", "2", Vec::new()),
                scaled("v", 0, "M", "0", Vec::new()),
                scaled("v", 1, "M", "1", Vec::new()),
            ],
        );
        assert_eq!(out, "mul(v.xyz, float3x4(M[0], M[1], M[2]))");
    }

    /// A whole declared matrix can be named, which is what says the transform is that matrix rather
    /// than four rows that happen to sit together. The row it starts at is the matrix's first, not
    /// whichever one the compiler wrote first.
    #[test]
    fn a_whole_declared_matrix_is_named() {
        let out = transformed(
            &matrixed(),
            [3, 2, 0, 1]
                .map(|lane| scaled("v", lane, "g_B.m_M", &lane.to_string(), Vec::new()))
                .to_vec(),
        );
        assert_eq!(out, "mul(v, g_B.m_M)");
    }

    /// What the sum holds besides the rows is the translation of an affine transform, and it is
    /// added to the multiply the same way it was added to the rows.
    #[test]
    fn what_is_not_a_row_is_added_back() {
        let out = transformed(
            &Names::default(),
            vec![
                scaled("v", 0, "M", "0", vec![0, 1, 2]),
                scaled("v", 1, "M", "1", vec![0, 1, 2]),
                scaled("v", 2, "M", "2", vec![0, 1, 2]),
                read("M[3]", vec![0, 1, 2]),
            ],
        );
        assert_eq!(
            out,
            "mul(v.xyz, float3x3(M[0].xyz, M[1].xyz, M[2].xyz)) + M[3].xyz"
        );
    }

    /// The components of the vector need not have stayed together, and gathering them is what says
    /// they were one value.
    #[test]
    fn weights_kept_apart_are_gathered_into_the_vector() {
        let out = transformed(
            &Names::default(),
            vec![
                scaled("r1", 3, "M", "0", Vec::new()),
                scaled("r2", 3, "M", "1", Vec::new()),
                scaled("r0", 0, "M", "2", Vec::new()),
            ],
        );
        assert_eq!(
            out,
            "mul(float3(r1.w, r2.w, r0.x), float3x4(M[0], M[1], M[2]))"
        );
    }

    /// Rows of two different matrices are two multiplies standing side by side, and neither is
    /// long enough to be one.
    #[test]
    fn rows_of_different_matrices_are_not_one_multiply() {
        let parts = vec![
            scaled("v", 0, "M", "0", Vec::new()),
            scaled("v", 1, "N", "1", Vec::new()),
            scaled("v", 2, "M", "2", Vec::new()),
        ];
        let text = sum(parts.clone()).text();
        assert_eq!(transformed(&Names::default(), parts), text);
    }

    /// A dot product against part of a row is a narrower matrix, not a different kind of thing: a
    /// transform that leaves the translation alone comes back this way.
    #[test]
    fn rows_taken_in_part_are_a_narrower_matrix() {
        let narrow = |index: &str| {
            call(
                "dot",
                vec![
                    Expr::Read {
                        base: format!("(M[{index}])"),
                        swizzle: vec![0, 1, 2],
                    },
                    read("r1", vec![1, 2, 3]),
                ],
                1,
            )
        };
        let out = folded(vec![
            assignment("r0", 0, narrow("0")),
            assignment("r0", 1, narrow("1")),
            assignment("r0", 2, narrow("2")),
        ]);
        assert_eq!(
            out,
            ["    r0.xyz = mul(float3x3((M[0]).xyz, (M[1]).xyz, (M[2]).xyz), r1.yzw);"]
        );
    }

    /// Rows taken different ways are not one matrix, whatever else they have in common.
    #[test]
    fn rows_taken_differently_are_not_one_matrix() {
        let taken = |index: &str, swizzle: Vec<u8>| {
            call(
                "dot",
                vec![
                    Expr::Read {
                        base: format!("(M[{index}])"),
                        swizzle,
                    },
                    read("r1", vec![1, 2, 3]),
                ],
                1,
            )
        };
        let out = folded(vec![
            assignment("r0", 0, taken("0", vec![0, 1, 2])),
            assignment("r0", 1, taken("1", vec![0, 1])),
            assignment("r0", 2, taken("2", vec![0, 1, 2])),
        ]);
        assert_eq!(out.len(), 3);
    }
}
