//! Idiom recovery — the shapes the compiler expanded, put back.
//!
//! `pow` leaves no trace of itself in the bytecode: it is a log, a multiply and an exponent, and
//! `normalize` is a dot product, a reciprocal square root and a multiply. Recognising those runs is
//! the last thing standing between an expression that is merely correct and one that says what the
//! shader meant.
//!
//! Every rule here is conditioned on the pieces being the *same* value, not merely alike, which is
//! what keeps `a * rsqrt(dot(b, b))` from being read as normalising anything. That the pieces are
//! present at all already means each was used exactly once, since a value used twice keeps its
//! register and never reaches this.

use super::expr::{Domain, Expr, call, identity};

/// Both readings of a commutative operator, so a rule can be written once.
fn either<'a>(left: &'a Expr, right: &'a Expr) -> [(&'a Expr, &'a Expr); 2] {
    [(left, right), (right, left)]
}

/// The lanes of a constant, where every one of them is a negative float.
fn negated(expr: &Expr) -> Option<Expr> {
    let Expr::Literal {
        bits,
        domain: Domain::Float,
    } = expr
    else {
        return None;
    };
    bits.iter()
        .all(|value| f32::from_bits(*value) < 0.0)
        .then(|| Expr::Literal {
            bits: bits.iter().map(|value| value ^ 0x8000_0000).collect(),
            domain: Domain::Float,
        })
}

/// A value with its leading negation taken off, if it has one.
fn without_sign(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Unary { op: "-", value } => Some((**value).clone()),
        other => negated(other),
    }
}

fn is_literal(expr: &Expr, value: f32) -> bool {
    matches!(expr, Expr::Literal { bits, domain: Domain::Float }
        if bits.iter().all(|bit| f32::from_bits(*bit) == value))
}

/// The condition behind a weight that is only ever one or nought, which is a choice written as
/// arithmetic because the machine had no branch to spare for it.
fn either_way(expr: &Expr, exact: bool) -> Option<&Expr> {
    let Expr::Select { cond, then, els } = expr else {
        return None;
    };
    // Reading a weight of one or nought back as a branch is only the same arithmetic for values that
    // are finite: nought times an infinity is not nothing. An exact reading leaves it as written.
    (!exact && is_literal(then, 1.0) && is_literal(els, 0.0)).then_some(&**cond)
}

/// A conditional standing in for a comparison's all-ones mask.
fn mask(expr: &Expr) -> Option<&Expr> {
    let Expr::Select { cond, then, els } = expr else {
        return None;
    };
    let ones =
        matches!(&**then, Expr::Literal { bits, .. } if bits.iter().all(|bit| *bit == u32::MAX));
    let zeros = matches!(&**els, Expr::Literal { bits, .. } if bits.iter().all(|bit| *bit == 0));
    (ones && zeros).then_some(&**cond)
}

fn unary(op: &'static str, value: Expr) -> Expr {
    Expr::Unary {
        op,
        value: Box::new(value),
    }
}

fn binary(op: &'static str, left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// The one argument of a call by this name.
fn argument<'a>(expr: &'a Expr, name: &str) -> Option<&'a Expr> {
    match expr {
        Expr::Call {
            name: called, args, ..
        } if called == name && args.len() == 1 => args.first(),
        _ => None,
    }
}

/// A dot product of one value with itself, which is a length squared however it is written.
fn square(expr: &Expr) -> Option<&Expr> {
    let Expr::Call { name, args, .. } = expr else {
        return None;
    };
    match (name.as_str(), args.as_slice()) {
        ("dot", [left, right]) if left == right => Some(left),
        _ => None,
    }
}

/// Rewrite an expression into the idioms it was expanded from, innermost first. Under `exact`,
/// nothing is recognised and only the broadcast collapse applies.
pub fn simplify(expr: Expr, exact: bool) -> Expr {
    let expr = descend(expr, exact);
    collapsed(rewrite(expr, exact))
}

/// One component spelled once rather than repeated to the width of what it is combined with. HLSL
/// spreads a single component over the other side of an operator by itself, so the repetition adds
/// nothing, and it is on nearly every line.
///
/// Only against a side that really is wider, and only on one of them: two repeats against each
/// other are what sets the width of the result, and collapsing both would narrow it.
fn collapsed(expr: Expr) -> Expr {
    // A sign or a complement applies to each component alike, so the repeat inside one collapses the
    // same way it would on its own.
    fn single(held: &Expr) -> Option<Expr> {
        match held {
            Expr::Read { base, swizzle } => match swizzle.split_first() {
                Some((first, rest))
                    if !rest.is_empty() && rest.iter().all(|lane| lane == first) =>
                {
                    Some(Expr::Read {
                        base: base.clone(),
                        swizzle: vec![*first],
                    })
                }
                _ => None,
            },
            Expr::Unary { op, value } => Some(Expr::Unary {
                op,
                value: Box::new(single(value)?),
            }),
            _ => None,
        }
    }

    match expr {
        // A sign lifted out of a product leaves the product where this would not otherwise look.
        Expr::Unary { op, value } => Expr::Unary {
            op,
            value: Box::new(collapsed(*value)),
        },
        // A choice made per component by one repeated condition is the same choice made once, and
        // the language spells that as a conditional rather than as a call.
        Expr::Select { cond, then, els } => {
            let wider = then.components() > 1 || els.components() > 1;
            let cond = match single(&cond).filter(|_| wider) {
                Some(held) => Box::new(held),
                None => cond,
            };
            Expr::Select { cond, then, els }
        }
        Expr::Binary { op, left, right } => {
            let (left, right) = if let Some(held) = single(&left).filter(|_| right.components() > 1)
            {
                (held, *right)
            } else if let Some(held) = single(&right).filter(|_| left.components() > 1) {
                (*left, held)
            } else {
                (*left, *right)
            };
            Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        leaf => leaf,
    }
}

fn descend(expr: Expr, exact: bool) -> Expr {
    match expr {
        Expr::Unary { op, value } => Expr::Unary {
            op,
            value: Box::new(simplify(*value, exact)),
        },
        Expr::Binary { op, left, right } => Expr::Binary {
            op,
            left: Box::new(simplify(*left, exact)),
            right: Box::new(simplify(*right, exact)),
        },
        Expr::Call { name, args, width } => Expr::Call {
            name,
            args: args.into_iter().map(|arg| simplify(arg, exact)).collect(),
            width,
        },
        Expr::Select { cond, then, els } => Expr::Select {
            cond: Box::new(simplify(*cond, exact)),
            then: Box::new(simplify(*then, exact)),
            els: Box::new(simplify(*els, exact)),
        },
        Expr::Swizzle { value, swizzle } => Expr::Swizzle {
            value: Box::new(simplify(*value, exact)),
            swizzle,
        },
        leaf => leaf,
    }
}

fn rewrite(expr: Expr, exact: bool) -> Expr {
    // Every rule here recognises a run of instructions as the operation it came from, and the
    // operation compiles to its own sequence rather than to the run. That is the point of them, and
    // it is also why an exact reading leaves them all off: what it says the shader computes, it
    // computes instruction for instruction.
    if exact {
        return expr;
    }
    match &expr {
        Expr::Binary {
            op: "+",
            left,
            right,
        } => {
            // Adding a negative is subtracting, which is how the source had it before the compiler
            // folded the sign into an operand.
            if let Some(value) = without_sign(right) {
                return rewrite(binary("-", (**left).clone(), value), exact);
            }
            if let Some(value) = without_sign(left) {
                return rewrite(binary("-", (**right).clone(), value), exact);
            }
            // Scaling by one or nought and adding is the branch the machine had none to spare for:
            // either the sum, or what it was added to on its own.
            for (product, addend) in either(left, right) {
                if let Expr::Binary {
                    op: "*",
                    left: scaled,
                    right: weight,
                } = product
                {
                    for (weight, value) in either(scaled, weight) {
                        let Some(cond) = either_way(weight, exact) else {
                            continue;
                        };
                        // Scaling the distance between two values and adding the nearer back is how
                        // the machine chooses between them, so the choice is the two themselves
                        // rather than one of them plus a difference that cancels.
                        let then = match value {
                            Expr::Binary {
                                op: "-",
                                left,
                                right,
                            } if &**right == addend => (**left).clone(),
                            held => rewrite(binary("+", held.clone(), addend.clone()), exact),
                        };
                        return Expr::Select {
                            cond: Box::new(cond.clone()),
                            then: Box::new(then),
                            els: Box::new(addend.clone()),
                        };
                    }
                    for (difference, weight) in either(scaled, weight) {
                        let Expr::Binary {
                            op: "-",
                            left: to,
                            right: from,
                        } = difference
                        else {
                            continue;
                        };
                        // The value subtracted has to be the one added back, or this is an
                        // interpolation between something and something else entirely.
                        if &**from == addend {
                            return rewrite(
                                call(
                                    "lerp",
                                    vec![(**from).clone(), (**to).clone(), (*weight).clone()],
                                    expr.width(),
                                ),
                                exact,
                            );
                        }
                    }
                }
            }
            expr
        }
        // A negation inside a product belongs outside it, where an addition can turn it into a
        // subtraction.
        Expr::Binary {
            op: "*",
            left,
            right,
        } => {
            if let Some(value) = without_sign(left) {
                return unary("-", rewrite(binary("*", value, (**right).clone()), exact));
            }
            if let Some(value) = without_sign(right) {
                return unary("-", rewrite(binary("*", (**left).clone(), value), exact));
            }
            // The natural logarithm, the other way about: the base-two one scaled back down.
            for (value, scale) in either(left, right) {
                if is_literal(scale, std::f32::consts::LN_2)
                    && let Some(inner) = argument(value, "log2")
                {
                    return call("log", vec![inner.clone()], expr.width());
                }
            }
            for (vector, scale) in either(left, right) {
                let Some(length) = argument(scale, "rsqrt") else {
                    continue;
                };
                if square(length) == Some(vector) {
                    return call("normalize", vec![(*vector).clone()], expr.width());
                }
            }
            expr
        }
        // Masking a value with a comparison keeps it or zeroes it, which is the conditional the
        // shader was written with.
        Expr::Binary {
            op: "&",
            left,
            right,
        } => {
            for (mask_side, value) in either(left, right) {
                let Some(cond) = mask(mask_side) else {
                    continue;
                };
                return Expr::Select {
                    cond: Box::new(cond.clone()),
                    then: Box::new((*value).clone()),
                    els: Box::new(Expr::Literal {
                        bits: vec![0; value.width()],
                        domain: Domain::Uint,
                    }),
                };
            }
            expr
        }
        // A raw buffer read narrowed to its leading components is a narrower read. The language has
        // one for each width, and fetching four only to drop some says nothing the shorter one does
        // not, while the offset and its alignment are the same either way.
        Expr::Swizzle { value, swizzle }
            if swizzle.len() < 4
                && identity(swizzle)
                && matches!(&**value, Expr::Call { name, .. } if name.ends_with(".Load4")) =>
        {
            let Expr::Call { name, args, .. } = &**value else {
                unreachable!("just matched a call")
            };
            let held = name.strip_suffix('4').unwrap_or(name);
            let taken = match swizzle.len() {
                1 => held.to_owned(),
                lanes => format!("{held}{lanes}"),
            };
            call(&taken, args.clone(), swizzle.len())
        }
        Expr::Call { name, args, width } => match (name.as_str(), args.as_slice()) {
            // Interpolating all the way to one end or the other is not interpolating: it is the
            // choice the shader was written with, which the machine had no branch to spare for.
            ("lerp", [from, to, weight]) => match either_way(weight, exact) {
                Some(cond) => Expr::Select {
                    cond: Box::new(cond.clone()),
                    then: Box::new(to.clone()),
                    els: Box::new(from.clone()),
                },
                None => expr,
            },
            // Negating both sides of a dot product cancels, and hiding that keeps a normalise from
            // being recognised.
            ("dot", [left, right])
                if matches!(left, Expr::Unary { op: "-", .. })
                    && matches!(right, Expr::Unary { op: "-", .. }) =>
            {
                let strip = |expr: &Expr| match expr {
                    Expr::Unary { value, .. } => (**value).clone(),
                    other => other.clone(),
                };
                call("dot", vec![strip(left), strip(right)], *width)
            }
            // Reinterpreting is free, so a cast of a constant is that constant and a cast straight
            // back is nothing at all.
            ("asfloat" | "asint" | "asuint", [Expr::Literal { bits, .. }]) => Expr::Literal {
                bits: bits.clone(),
                domain: match name.as_str() {
                    "asfloat" => Domain::Float,
                    "asint" => Domain::Int,
                    _ => Domain::Uint,
                },
            },
            // A cast of a choice is a choice between casts, which is what lets the two sides of a
            // masked value simplify separately.
            ("asfloat" | "asint" | "asuint", [Expr::Select { cond, then, els }]) => Expr::Select {
                cond: cond.clone(),
                then: Box::new(rewrite(call(name, vec![(**then).clone()], *width), exact)),
                els: Box::new(rewrite(call(name, vec![(**els).clone()], *width), exact)),
            },
            ("asfloat", [inner]) => argument(inner, "asuint").cloned().unwrap_or(expr),
            ("asuint", [inner]) => argument(inner, "asfloat").cloned().unwrap_or(expr),
            (
                "exp2",
                [
                    Expr::Binary {
                        op: "*",
                        left,
                        right,
                    },
                ],
            ) => {
                for (logarithm, power) in either(left, right) {
                    if let Some(base) = argument(logarithm, "log2") {
                        return call("pow", vec![base.clone(), (*power).clone()], *width);
                    }
                }
                // The machine has only the base-two exponent, so the natural one arrives scaled by
                // the logarithm of two. Nothing but that exact constant means this.
                for (value, scale) in either(left, right) {
                    if is_literal(scale, std::f32::consts::LOG2_E) {
                        return call("exp", vec![(*value).clone()], *width);
                    }
                }
                expr
            }
            ("sqrt", [inner]) => match square(inner) {
                // The length of a difference is how far apart the two ends are.
                Some(Expr::Binary {
                    op: "-",
                    left,
                    right,
                }) => call(
                    "distance",
                    vec![(**left).clone(), (**right).clone()],
                    *width,
                ),
                Some(vector) => call("length", vec![vector.clone()], *width),
                None => expr,
            },
            (
                "min",
                [
                    Expr::Call {
                        name: inner,
                        args: bounds,
                        ..
                    },
                    high,
                ],
            ) if inner == "max" => {
                let [first, second] = bounds.as_slice() else {
                    return expr;
                };
                // The bounds are the constants; without exactly one of them being constant there is
                // no telling which operand is being clamped and which is clamping it.
                let constant = |expr: &Expr| matches!(expr, Expr::Literal { .. });
                let (value, low) = match (constant(first), constant(second)) {
                    (false, true) => (first, second),
                    (true, false) => (second, first),
                    _ => return expr,
                };
                if !constant(high) {
                    return expr;
                }
                match is_literal(low, 0.0) && is_literal(high, 1.0) {
                    true => call("saturate", vec![value.clone()], *width),
                    false => call(
                        "clamp",
                        vec![value.clone(), low.clone(), high.clone()],
                        *width,
                    ),
                }
            }
            _ => expr,
        },
        _ => expr,
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

    fn float(value: f32, lanes: usize) -> Expr {
        Expr::Literal {
            bits: vec![value.to_bits(); lanes],
            domain: Domain::Float,
        }
    }

    #[test]
    fn adding_a_negative_constant_subtracts_it() {
        let expr = binary("+", read("r0", &[0]), float(-1.0, 1));
        assert_eq!(simplify(expr, false).text(), "r0.x - 1.0");
    }

    #[test]
    fn a_sign_inside_a_product_comes_out_of_it() {
        let expr = binary(
            "+",
            binary("*", read("r0", &[0]), float(-2.0, 1)),
            float(1.5, 1),
        );
        assert_eq!(simplify(expr, false).text(), "1.5 - r0.x * 2.0");
    }

    #[test]
    fn a_log_scaled_and_exponentiated_is_a_power() {
        let expr = call(
            "exp2",
            vec![binary(
                "*",
                call("log2", vec![read("r0", &[0])], 1),
                read("r1", &[1]),
            )],
            1,
        );
        assert_eq!(simplify(expr, false).text(), "pow(r0.x, r1.y)");
    }

    #[test]
    fn a_vector_over_its_own_length_normalises() {
        let vector = read("r0", &[0, 1, 2]);
        let expr = binary(
            "*",
            vector.clone(),
            call(
                "rsqrt",
                vec![call("dot", vec![vector.clone(), vector], 1)],
                1,
            ),
        );
        assert_eq!(simplify(expr, false).text(), "normalize(r0.xyz)");
    }

    /// The whole precondition: scaling one vector by another's reciprocal length is not normalising.
    #[test]
    fn a_different_vector_does_not_normalise() {
        let other = read("r1", &[0, 1, 2]);
        let expr = binary(
            "*",
            read("r0", &[0, 1, 2]),
            call("rsqrt", vec![call("dot", vec![other.clone(), other], 1)], 1),
        );
        assert!(!simplify(expr, false).text().contains("normalize"));
    }

    #[test]
    fn a_square_root_of_a_self_dot_is_a_length() {
        let vector = read("r0", &[0, 1, 2]);
        let expr = call(
            "sqrt",
            vec![call("dot", vec![vector.clone(), vector], 1)],
            1,
        );
        assert_eq!(simplify(expr, false).text(), "length(r0.xyz)");
    }

    #[test]
    fn a_difference_scaled_and_added_back_interpolates() {
        let from = read("r0", &[0]);
        let to = read("r1", &[0]);
        let expr = binary(
            "+",
            binary("*", binary("-", to, from.clone()), read("r2", &[0])),
            from,
        );
        assert_eq!(simplify(expr, false).text(), "lerp(r0.x, r1.x, r2.x)");
    }

    #[test]
    fn a_masked_value_becomes_the_choice_it_stood_for() {
        let cond = binary("<", read("r0", &[0]), float(0.0, 1));
        let mask = Expr::Select {
            cond: Box::new(cond),
            then: Box::new(Expr::Literal {
                bits: vec![u32::MAX],
                domain: Domain::Uint,
            }),
            els: Box::new(Expr::Literal {
                bits: vec![0],
                domain: Domain::Uint,
            }),
        };
        let expr = call(
            "asfloat",
            vec![binary("&", call("asuint", vec![read("r1", &[3])], 1), mask)],
            1,
        );
        assert_eq!(simplify(expr, false).text(), "r0.x < 0.0 ? r1.w : 0.0");
    }

    #[test]
    fn a_bounded_value_clamps() {
        let expr = call(
            "min",
            vec![
                call("max", vec![read("r0", &[0]), float(0.0, 1)], 1),
                float(1.0, 1),
            ],
            1,
        );
        assert_eq!(simplify(expr, false).text(), "saturate(r0.x)");
    }

    #[test]
    fn a_repeated_component_collapses_against_a_wider_side() {
        let read = |base: &str, swizzle: &[u8]| Expr::Read {
            base: base.to_owned(),
            swizzle: swizzle.to_vec(),
        };
        let product = binary("*", read("r2", &[0, 0, 0]), read("r3", &[0, 1, 2]));
        assert_eq!(simplify(product, false).text(), "r2.x * r3.xyz");
        let signed = binary(
            "*",
            unary("-", read("r2", &[0, 0, 0])),
            read("r3", &[0, 1, 2]),
        );
        assert_eq!(simplify(signed, false).text(), "-(r2.x * r3.xyz)");
        let sum = binary(
            "+",
            read("r0", &[1, 2, 3]),
            binary(
                "*",
                unary("-", read("r2", &[0, 0, 0])),
                read("r3", &[0, 1, 2]),
            ),
        );
        assert_eq!(simplify(sum, false).text(), "r0.yzw - r2.x * r3.xyz");
    }

    /// A weight that is only ever one or nought is not a weight, and the interpolation it drives is
    /// the choice the machine had no branch to spare for.
    #[test]
    fn interpolating_by_one_or_nought_is_a_choice() {
        let read = |base: &str| Expr::Read {
            base: base.to_owned(),
            swizzle: Vec::new(),
        };
        let pick = Expr::Select {
            cond: Box::new(read("held")),
            then: Box::new(Expr::Literal {
                bits: vec![1f32.to_bits()],
                domain: Domain::Float,
            }),
            els: Box::new(Expr::Literal {
                bits: vec![0],
                domain: Domain::Float,
            }),
        };
        let held = call("lerp", vec![read("a"), read("b"), pick.clone()], 4);
        assert_eq!(simplify(held, false).text(), "held ? b : a");
        // A weight that is anything else stays an interpolation.
        let open = call("lerp", vec![read("a"), read("b"), read("t")], 4);
        assert_eq!(simplify(open, false).text(), "lerp(a, b, t)");
    }

    /// Scaling the distance between two values by nought or one and adding the nearer back is how
    /// the machine chooses between them without a branch.
    #[test]
    fn scaling_a_difference_by_a_choice_is_the_choice() {
        let read = |base: &str| Expr::Read {
            base: base.to_owned(),
            swizzle: Vec::new(),
        };
        let pick = Expr::Select {
            cond: Box::new(read("held")),
            then: Box::new(Expr::Literal {
                bits: vec![1f32.to_bits()],
                domain: Domain::Float,
            }),
            els: Box::new(Expr::Literal {
                bits: vec![0],
                domain: Domain::Float,
            }),
        };
        // mask * (a - b) + b is a choice between a and b, with the difference cancelling.
        let whole = binary(
            "+",
            binary("*", pick.clone(), binary("-", read("a"), read("b"))),
            read("b"),
        );
        assert_eq!(simplify(whole, false).text(), "held ? a : b");
        // Anything else scaled by it is that plus what it was added to, or just that.
        let part = binary("+", binary("*", pick, read("a")), read("b"));
        assert_eq!(simplify(part, false).text(), "held ? a + b : b");
    }

    /// One condition repeated to the width of the arms is one condition, and the language spells that
    /// as a conditional rather than as a per-component call.
    #[test]
    fn a_repeated_condition_makes_a_conditional_not_a_call() {
        let held = Expr::Select {
            cond: Box::new(Expr::Read {
                base: "r0".to_owned(),
                swizzle: vec![3, 3, 3],
            }),
            then: Box::new(Expr::Read {
                base: "a".to_owned(),
                swizzle: vec![0, 1, 2],
            }),
            els: Box::new(Expr::Read {
                base: "b".to_owned(),
                swizzle: vec![0, 1, 2],
            }),
        };
        assert_eq!(simplify(held, false).text(), "r0.w ? a.xyz : b.xyz");
        // Components that really differ stay a per-component choice.
        let apart = Expr::Select {
            cond: Box::new(Expr::Read {
                base: "r0".to_owned(),
                swizzle: vec![0, 1, 2],
            }),
            then: Box::new(Expr::Read {
                base: "a".to_owned(),
                swizzle: vec![0, 1, 2],
            }),
            els: Box::new(Expr::Read {
                base: "b".to_owned(),
                swizzle: vec![0, 1, 2],
            }),
        };
        assert_eq!(
            simplify(apart, false).text(),
            "select(r0.xyz, a.xyz, b.xyz)"
        );
    }

    /// A raw buffer read narrowed to its leading components is a narrower read, and the language has
    /// one for each width.
    #[test]
    fn a_raw_load_narrowed_is_a_narrower_load() {
        let load = call(
            "g_Data.Load4",
            vec![Expr::Read {
                base: "off".to_owned(),
                swizzle: Vec::new(),
            }],
            4,
        );
        assert_eq!(
            simplify(load.clone().select(&[0]), false).text(),
            "g_Data.Load(off)"
        );
        assert_eq!(
            simplify(load.clone().select(&[0, 1]), false).text(),
            "g_Data.Load2(off)"
        );
        assert_eq!(
            simplify(load.clone().select(&[0, 1, 2]), false).text(),
            "g_Data.Load3(off)"
        );
        // Taking the whole of it, or components out of order, is not a narrower read.
        assert_eq!(simplify(load.clone(), false).text(), "g_Data.Load4(off)");
        assert_eq!(
            simplify(load.select(&[1, 2]), false).text(),
            "g_Data.Load4(off).yz"
        );
    }
}
