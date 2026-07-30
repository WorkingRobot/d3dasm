//! Expression tree — the values a shader computes, rather than the registers they land in.
//!
//! Everything is a vector of up to four lanes, because that is the only thing the machine has. A
//! scalar is a one-lane value and broadcasts, which is how a dot product feeds a multiply without
//! any of it being written down.

/// How the bits in a value are meant to be read.
///
/// A register holds thirty-two untyped bits, so this travels with the expression rather than with
/// the register: the same `r0.x` is a float to `add` and a mask to `and`, and a cast goes in
/// wherever the two disagree.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Float,
    Int,
    Uint,
    Bool,
}

/// A value the shader computes. Every variant is a vector of one to four lanes.
#[derive(Clone, PartialEq)]
pub enum Expr {
    /// A register under whatever name the caller gave it, and the components taken from it.
    Read {
        base: String,
        swizzle: Vec<u8>,
    },
    /// Raw bits, one per lane; the domain decides how they read.
    Literal {
        bits: Vec<u32>,
        domain: Domain,
    },
    Unary {
        op: &'static str,
        value: Box<Expr>,
    },
    Binary {
        op: &'static str,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Call {
        name: String,
        args: Vec<Expr>,
        width: usize,
    },
    Select {
        cond: Box<Expr>,
        then: Box<Expr>,
        els: Box<Expr>,
    },
    /// Lanes of a value that is not a plain register read.
    Swizzle {
        value: Box<Expr>,
        swizzle: Vec<u8>,
    },
    /// Lanes taken from different places, side by side. A register read that straddles two constant
    /// buffer fields has no single name, so it is built out of the ones it does have.
    Vector(Vec<Expr>),
}

const LANES: [char; 4] = ['x', 'y', 'z', 'w'];

/// Binding strength, so a child only gets brackets where it would otherwise regroup the expression.
/// The numbers are C's, which HLSL inherits.
fn strength(op: &str) -> u8 {
    match op {
        "||" => 2,
        "&&" => 3,
        "|" => 4,
        "^" => 5,
        "&" => 6,
        "==" | "!=" => 7,
        "<" | ">" | "<=" | ">=" => 8,
        "<<" | ">>" => 9,
        "+" | "-" => 10,
        _ => 11,
    }
}

/// Lower than any operator, so a conditional always brackets inside one.
const TERNARY: u8 = 1;
/// Higher than any operator: what a swizzle or a call argument may hold without brackets.
const ATOM: u8 = 13;

fn pick<T: Copy>(from: &[T], lanes: &[u8]) -> Vec<T> {
    lanes
        .iter()
        .filter_map(|lane| from.get(usize::from(*lane)).copied())
        .collect()
}

/// Whether a swizzle names the lanes it has in their own order, which selects nothing.
pub(crate) fn identity(swizzle: &[u8]) -> bool {
    swizzle
        .iter()
        .enumerate()
        .all(|(at, lane)| usize::from(*lane) == at)
}

impl Expr {
    /// How many lanes the value is wide.
    pub fn width(&self) -> usize {
        match self {
            // Naming no components means the whole of a value that stands alone, which is one
            // thing wide. Nought would make everything sized from it collapse.
            Self::Read { swizzle, .. } => swizzle.len().max(1),
            Self::Swizzle { swizzle, .. } => swizzle.len(),
            Self::Literal { bits, .. } => bits.len(),
            Self::Unary { value, .. } => value.width(),
            Self::Binary { left, right, .. } => left.width().max(right.width()),
            Self::Call { width, .. } => *width,
            Self::Select { cond, then, els } => cond.width().max(then.width()).max(els.width()),
            Self::Vector(lanes) => lanes.iter().map(Self::width).sum(),
        }
    }

    /// How many components the written expression really has, or nought where that cannot be told.
    ///
    /// Not the same as `width`, which is how wide the value is being used as: a dot product written
    /// to three components is three wide and one component, and only one of those can stand as a
    /// constructor argument. Anything this cannot account for comes back nought rather than guessed.
    pub fn components(&self) -> usize {
        // Operands of different widths are the narrow one spread over the wide, which only works
        // where the narrow one is a single component.
        fn spread(parts: impl Iterator<Item = usize>) -> usize {
            let mut widest = 1;
            for part in parts {
                match part {
                    0 => return 0,
                    1 => continue,
                    _ if widest == 1 || widest == part => widest = part,
                    _ => return 0,
                }
            }
            widest
        }
        match self {
            Self::Read { swizzle, .. } => swizzle.len().max(1),
            Self::Swizzle { swizzle, .. } => swizzle.len(),
            Self::Literal { bits, .. } => match uniform(bits) {
                true => 1,
                false => bits.len(),
            },
            Self::Unary { value, .. } => value.components(),
            Self::Binary { left, right, .. } => {
                spread([left.components(), right.components()].into_iter())
            }
            Self::Select { cond, then, els } => {
                spread([cond.components(), then.components(), els.components()].into_iter())
            }
            Self::Vector(lanes) => lanes.iter().map(Self::components).sum(),
            Self::Call { name, args, width } => match name.as_str() {
                // The two that answer with one component whatever they were given.
                "dot" | "length" => 1,
                // A multiply is as wide as the matrix it was built for, and a constructor says so
                // in its own name.
                "mul" => *width,
                _ if name.starts_with("float") => *width,
                _ => spread(args.iter().map(Self::components)),
            },
        }
    }

    /// The value narrowed to the lanes given, by index into its own.
    ///
    /// Every operator here is elementwise, so narrowing one pushes down into its operands and comes
    /// out as a swizzle on the leaves rather than a bracket around the whole expression. What is not
    /// elementwise — a call, a value already swizzled — takes the swizzle on the outside.
    pub fn select(self, lanes: &[u8]) -> Self {
        if self.width() <= 1 {
            return self;
        }
        if lanes.len() == self.width() && identity(lanes) {
            return self;
        }
        match self {
            Self::Read { base, swizzle } => Self::Read {
                base,
                swizzle: pick(&swizzle, lanes),
            },
            Self::Literal { bits, domain } => Self::Literal {
                bits: pick(&bits, lanes),
                domain,
            },
            Self::Unary { op, value } => Self::Unary {
                op,
                value: Box::new(value.select(lanes)),
            },
            Self::Binary { op, left, right } => Self::Binary {
                op,
                left: Box::new(left.select(lanes)),
                right: Box::new(right.select(lanes)),
            },
            Self::Select { cond, then, els } => Self::Select {
                cond: Box::new(cond.select(lanes)),
                then: Box::new(then.select(lanes)),
                els: Box::new(els.select(lanes)),
            },
            Self::Swizzle { value, swizzle } => Self::Swizzle {
                value,
                swizzle: pick(&swizzle, lanes),
            },
            // Every lane already stands alone, so narrowing is just keeping some of them.
            Self::Vector(parts) if parts.iter().all(|part| part.width() == 1) => Self::Vector(
                lanes
                    .iter()
                    .filter_map(|lane| parts.get(usize::from(*lane)).cloned())
                    .collect(),
            ),
            value => Self::Swizzle {
                value: Box::new(value),
                swizzle: lanes.to_vec(),
            },
        }
    }

    /// The expression as HLSL, bracketed only where precedence needs it.
    pub fn text(&self) -> String {
        self.render(TERNARY)
    }

    fn render(&self, least: u8) -> String {
        match self {
            // Taking all four components in order is what the register already is.
            Self::Read { base, swizzle } => match swizzle.is_empty() || swizzle[..] == [0, 1, 2, 3]
            {
                true => base.clone(),
                false => format!("{base}.{}", letters(swizzle)),
            },
            Self::Literal { bits, domain } => literal(bits, *domain),
            Self::Unary { op, value } => bracket(format!("{op}{}", value.render(12)), 12, least),
            Self::Binary { op, left, right } => {
                let here = strength(op);
                bracket(
                    format!("{} {op} {}", left.render(here), right.render(here + 1)),
                    here,
                    least,
                )
            }
            Self::Call { name, args, .. } => {
                let args: Vec<String> = args.iter().map(|arg| arg.render(TERNARY)).collect();
                format!("{name}({})", args.join(", "))
            }
            // A per-component choice is not the short-circuiting operator, and modern HLSL insists
            // on the difference.
            Self::Select { cond, then, els } if cond.width() > 1 => format!(
                "select({}, {}, {})",
                cond.render(TERNARY),
                then.render(TERNARY),
                els.render(TERNARY)
            ),
            Self::Select { cond, then, els } => bracket(
                format!(
                    "{} ? {} : {}",
                    cond.render(TERNARY + 1),
                    then.render(TERNARY + 1),
                    els.render(TERNARY)
                ),
                TERNARY,
                least,
            ),
            // Taking every lane of a value in order is the value itself.
            Self::Swizzle { value, swizzle }
                if swizzle.len() == value.width() && identity(swizzle) =>
            {
                value.render(least)
            }
            Self::Swizzle { value, swizzle } => {
                format!("{}.{}", value.render(ATOM), letters(swizzle))
            }
            Self::Vector(parts) => match parts.as_slice() {
                [only] => only.render(least),
                parts => {
                    let width: usize = parts.iter().map(Self::width).sum();
                    let parts: Vec<String> =
                        parts.iter().map(|part| part.render(TERNARY)).collect();
                    format!("float{width}({})", parts.join(", "))
                }
            },
        }
    }
}

fn bracket(text: String, here: u8, least: u8) -> String {
    match here < least {
        true => format!("({text})"),
        false => text,
    }
}

/// The letter naming one component.
pub(crate) fn lane(at: u8) -> char {
    LANES[usize::from(at) % LANES.len()]
}

/// A swizzle as the component letters that name it — `[0, 3]` is `"xw"`.
pub(crate) fn letters(swizzle: &[u8]) -> String {
    swizzle.iter().copied().map(lane).collect()
}

/// Whether a literal is one value repeated, which is written as that value and nothing more.
fn uniform(bits: &[u32]) -> bool {
    bits.split_first()
        .is_some_and(|(first, rest)| rest.iter().all(|value| value == first))
}

/// Denominators a shader divides by often enough that the quotient is worth recognising. Powers of
/// two are left out: those come back as short decimals already, and `1.0 / 2.0` says less than
/// `0.5`.
const OVER: [u32; 5] = [3, 6, 7, 9, 255];

/// A constant that is exactly a small fraction, written as the division that produced it.
///
/// Only where dividing really does give these bits back. A value merely close to one over something
/// is a different number, and saying otherwise would misstate what the shader computes. The
/// brackets matter: without them a division would bind to whatever the constant is multiplied by.
fn fraction(value: f32) -> Option<String> {
    fn shared(left: u32, right: u32) -> u32 {
        match right {
            0 => left,
            _ => shared(right, left % right),
        }
    }
    OVER.iter().find_map(|over| {
        let above = (value * *over as f32).round();
        let held = above / *over as f32;
        // In lowest terms, or the fraction says less than the decimal it replaces: three sixths is
        // a half, and nobody wrote it the long way.
        (held.to_bits() == value.to_bits()
            && above != 0.0
            && above.abs() < 1e7
            && shared(above.abs() as u32, *over) == 1)
            .then(|| format!("({above:.1} / {:.1})", *over as f32))
    })
}

/// A constant as HLSL writes it. Lanes that are all the same collapse to a scalar, which is what
/// they were before the compiler splatted them across a register.
fn literal(bits: &[u32], domain: Domain) -> String {
    let one = |value: &u32| -> String {
        match domain {
            Domain::Float => {
                let float = f32::from_bits(*value);
                if !float.is_finite() {
                    return format!("asfloat({value:#010x}u)");
                }
                match float == float.trunc() && float.abs() < 1e7 {
                    true => format!("{float:.1}"),
                    false => fraction(float).unwrap_or_else(|| format!("{float:?}")),
                }
            }
            Domain::Int => (*value as i32).to_string(),
            Domain::Uint | Domain::Bool => match *value > 9 {
                true => format!("{value:#x}u"),
                false => format!("{value}u"),
            },
        }
    };
    match bits {
        [] => String::new(),
        [first, ..] if uniform(bits) => one(first),
        _ => {
            let name = match domain {
                Domain::Float => "float",
                Domain::Int => "int",
                Domain::Uint | Domain::Bool => "uint",
            };
            let values: Vec<String> = bits.iter().map(one).collect();
            format!("{name}{}({})", bits.len(), values.join(", "))
        }
    }
}

/// The same value read in another domain, with the cast that makes it legal.
///
/// A comparison leaves a boolean rather than the all-ones mask the machine really holds, so taking
/// one as bits spells the mask out rather than reinterpreting it.
pub fn coerce(value: Expr, from: Domain, to: Domain) -> Expr {
    if from == to {
        return value;
    }
    // A constant is only ever bits until something reads it, so it takes the domain asked for
    // rather than a cast around it.
    if let Expr::Literal { bits, .. } = &value
        && to != Domain::Bool
    {
        return Expr::Literal {
            bits: bits.clone(),
            domain: to,
        };
    }
    let width = value.width();
    match (from, to) {
        // As a number a boolean is one or nought; as bits it is the mask a comparison would have
        // left.
        (Domain::Bool, _) => {
            let set = match to {
                Domain::Float => 1f32.to_bits(),
                _ => u32::MAX,
            };
            Expr::Select {
                cond: Box::new(value),
                then: Box::new(Expr::Literal {
                    bits: vec![set; width],
                    domain: to,
                }),
                els: Box::new(Expr::Literal {
                    bits: vec![0; width],
                    domain: to,
                }),
            }
        }
        (_, Domain::Bool) => Expr::Binary {
            op: "!=",
            left: Box::new(value),
            right: Box::new(Expr::Literal {
                bits: vec![0; width],
                domain: from,
            }),
        },
        // The bits are the same either way round; only the reading of them changes.
        (Domain::Int, Domain::Uint) | (Domain::Uint, Domain::Int) => value,
        (_, Domain::Float) => call("asfloat", vec![value], width),
        (_, Domain::Int) => call("asint", vec![value], width),
        (_, Domain::Uint) => call("asuint", vec![value], width),
    }
}

/// A call to `name`, whose result is `width` lanes wide.
pub fn call(name: &str, args: Vec<Expr>, width: usize) -> Expr {
    Expr::Call {
        name: name.to_owned(),
        args,
        width,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn read(base: &str, swizzle: &[u8]) -> Expr {
        Expr::Read {
            base: base.to_owned(),
            swizzle: swizzle.to_vec(),
        }
    }

    #[test]
    fn narrowing_reaches_the_leaves() {
        let sum = Expr::Binary {
            op: "+",
            left: Box::new(read("r0", &[0, 1, 2])),
            right: Box::new(read("r1", &[3, 3, 3])),
        };
        assert_eq!(sum.select(&[0, 2]).text(), "r0.xz + r1.ww");
    }

    #[test]
    fn narrowing_a_call_stays_outside_it() {
        let sample = call("t0.Sample", vec![read("v1", &[0, 1])], 4);
        assert_eq!(sample.select(&[2]).text(), "t0.Sample(v1.xy).z");
    }

    #[test]
    fn a_scalar_broadcasts_rather_than_narrowing() {
        let dot = call("dot", vec![read("r0", &[0, 1, 2])], 1);
        assert_eq!(dot.select(&[1, 1]).text(), "dot(r0.xyz)");
    }

    #[test]
    fn brackets_follow_binding_strength() {
        let inner = Expr::Binary {
            op: "+",
            left: Box::new(read("a", &[0])),
            right: Box::new(read("b", &[0])),
        };
        let outer = Expr::Binary {
            op: "*",
            left: Box::new(inner.clone()),
            right: Box::new(read("c", &[0])),
        };
        assert_eq!(outer.text(), "(a.x + b.x) * c.x");
        let flat = Expr::Binary {
            op: "+",
            left: Box::new(inner),
            right: Box::new(read("c", &[0])),
        };
        assert_eq!(flat.text(), "a.x + b.x + c.x");
    }

    /// Subtraction and division do not reassociate, so a right operand of the same strength keeps
    /// its brackets where an addition would drop them.
    #[test]
    fn the_right_of_a_subtraction_keeps_its_brackets() {
        let inner = Expr::Binary {
            op: "-",
            left: Box::new(read("b", &[0])),
            right: Box::new(read("c", &[0])),
        };
        let outer = Expr::Binary {
            op: "-",
            left: Box::new(read("a", &[0])),
            right: Box::new(inner),
        };
        assert_eq!(outer.text(), "a.x - (b.x - c.x)");
    }

    #[test]
    fn constants_collapse_when_every_lane_agrees() {
        let same = Expr::Literal {
            bits: vec![0.5f32.to_bits(); 4],
            domain: Domain::Float,
        };
        assert_eq!(same.text(), "0.5");
        let mixed = Expr::Literal {
            bits: vec![0, 1f32.to_bits()],
            domain: Domain::Float,
        };
        assert_eq!(mixed.text(), "float2(0.0, 1.0)");
    }

    #[test]
    fn a_mask_taken_as_bits_spells_itself_out() {
        let mask = Expr::Binary {
            op: "<",
            left: Box::new(read("r0", &[0])),
            right: Box::new(read("r1", &[0])),
        };
        assert_eq!(
            coerce(mask, Domain::Bool, Domain::Uint).text(),
            "r0.x < r1.x ? 0xffffffffu : 0u"
        );
    }

    /// A constant that is exactly a small fraction says more as the division that made it, but only
    /// where the decimal was the worse of the two: three sixths is a half.
    #[test]
    fn a_constant_that_is_a_fraction_is_written_as_one() {
        let render = |value: f32| literal(&[value.to_bits()], Domain::Float);
        assert_eq!(render(1.0 / 255.0), "(1.0 / 255.0)");
        assert_eq!(render(254.0 / 255.0), "(254.0 / 255.0)");
        assert_eq!(render(1.0 / 3.0), "(1.0 / 3.0)");
        assert_eq!(render(0.5), "0.5");
        assert_eq!(render(0.25), "0.25");
        assert_eq!(render(2.0), "2.0");
        // Near enough is a different number, and saying otherwise would misstate the shader.
        assert_eq!(render(0.0039), "0.0039");
    }
}
