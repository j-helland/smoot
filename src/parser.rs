use std::str::FromStr;

use num_traits::Float;

use crate::base_unit::BaseUnit;
use crate::registry::{Registry, REGISTRY};
use crate::{error::SmootError, quantity::Quantity, unit::Unit};

trait ParsableFloat: FromStr + Float {}
impl ParsableFloat for f64 {}

// Unit and quantity expression parser.
peg::parser! {
    pub(crate) grammar expression_parser() for str {
        /// Matches whitespace.
        rule __()
            = [' ' | '\n' | '\t']*
            {}

        rule sign() = ['-' | '+']
        rule digits() = [c if c.is_ascii_digit()]+

        pub rule integer() -> i32
            = num:$("-"?digits()) !['.']
            {? num.parse::<i32>().or(Err("Invalid integer number")) }

        pub rule decimal<N: ParsableFloat>() -> N
            = num:$(sign()?(digits()".")?digits()(['e' | 'E']sign()?digits())?)
            {? num.parse::<N>().or(Err("Invalid decimal number")) }

        /// Parses a basic unit. Will fail if the unit string is not found in the specified unit cache.
        rule unit(registry: &Registry) -> Unit<f64>
            = u:$(['a'..='z' | 'A'..='Z' | '_']+)
            {?
                registry
                    .get_unit(u)
                    .map(|u| BaseUnit::clone(u))
                    .map(|u| Unit::new(vec![u], vec![]))
                    .ok_or("Unknown unit")
            }

        /// Core parser with operator precedence.
        /// We only need to support a few basic operators; addition and subtraction don't make sense for units.
        pub rule unit_expression(unit_cache: &Registry) -> Unit<f64>
            = precedence!
            {
                u1:(@) __ "*" __ u2:@ { u1 * u2 }
                u1:(@) __ "/" __ u2:@ { u1 / u2 }
                --
                u:@ __ "**" __ n:integer() { u.powi(n) }
                u:@ __ "^" __ n:integer() { u.powi(n) }
                u:@ __ "**" __ n:decimal() { u.powf(n) }
                u:@ __ "^" __ n:decimal() { u.powf(n) }
                --
                u:unit(unit_cache) { u }
                "(" __ expr:unit_expression(unit_cache) __ ")" { expr }
            }

        rule quantity(unit_cache: &Registry) -> Quantity<f64, f64>
            = n:decimal() __ u:unit_expression(unit_cache)
            { Quantity::new(n, u) }

        /// Add operator requires conditional parsing to handle incompatible units.
        rule quantity_add(unit_cache: &Registry) -> Quantity<f64, f64>
            = q1:quantity(unit_cache) __ "+" __ q2:expression(unit_cache)
            {? (&q1 + &q2).or(Err("Incompatible units")) }

        /// Subtract operator requires conditional parsing to handle incompatible units.
        rule quantity_sub(unit_cache: &Registry) -> Quantity<f64, f64>
            = q1:quantity(unit_cache) __ "-" __ q2:expression(unit_cache)
            {? (&q1 - &q2).or(Err("Incompatible units")) }

        pub rule expression(unit_cache: &Registry) -> Quantity<f64, f64>
            = precedence!
            {
                q:quantity_add(unit_cache) { q }
                q:quantity_sub(unit_cache) { q }
                --
                // TODO(jwh): implement owning operators to reduce copies
                q1:(@) __ "*" __ q2:@ { &q1 * &q2 }
                q1:(@) __ "/" __ q2:@ { &q1 / &q2 }
                --
                q1:@ __ "**" __ n:integer() !['.'] { q1.powi(n) }
                q1:@ __ "^" __ n:integer() !['.'] { q1.powi(n) }
                q1:@ __ "**" __ n:decimal() { q1.powf(n) }
                q1:@ __ "^" __ n:decimal() { q1.powf(n) }
                --
                q:quantity(unit_cache) { q }
                "-" q:expression(unit_cache) { -q }
                "(" __ expr:expression(unit_cache) __ ")" { expr }
                n:decimal() { Quantity::new_dimensionless(n) }
            }
    }
}

/// Implement `.parse()` for strings into units.
impl FromStr for Unit<f64> {
    type Err = SmootError;

    fn from_str(s: &str) -> Result<Unit<f64>, Self::Err> {
        // TODO(jwh): get cache from non-global scope
        expression_parser::unit_expression(s, &REGISTRY)
            .map_err(|_e| SmootError::InvalidUnitExpression(0, s.into()))
    }
}

/// Implement `.parse()` for strings into quantities.
impl FromStr for Quantity<f64, f64> {
    type Err = SmootError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        expression_parser::expression(s, &REGISTRY)
            .map(|mut q| {
                q.ito_reduced_units();
                q
            })
            .map_err(|_| SmootError::InvalidQuantityExpression(0, s.into()))
    }
}

#[cfg(test)]
mod test_expression_parser {
    use crate::error::SmootResult;

    use super::*;
    use peg::{error::ParseError, str::LineCol};
    use std::sync::{Arc, LazyLock};
    use test_case::case;

    static UNIT_METER: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("meter").expect("No unit 'meter'"));
    static UNIT_KILOMETER: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("kilometer").expect("No unit 'kilometer'"));
    static UNIT_SECOND: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("second").expect("No unit 'second'"));
    static UNIT_GRAM: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("gram").expect("No unit 'gram'"));

    #[case("1", Some(1); "Basic")]
    #[case("100", Some(100); "Multiple digits")]
    #[case("-1", Some(-1); "Negative number")]
    #[case("asdf", None; "Not a number")]
    #[case("1.0", None; "Not an integer")]
    #[case("9", Some(9); "Digit at bounds")]
    fn test_parse_integer(s: &str, expected: Option<i32>) -> Result<(), ParseError<LineCol>> {
        let result = expression_parser::integer(s);
        if let Some(expected) = expected {
            assert_eq!(result?, expected);
        } else {
            assert!(result.is_err());
        }
        Ok(())
    }

    #[case("1.0", Some(1.0); "Basic")]
    #[case("-1.0", Some(-1.0); "Negative")]
    #[case("asdf", None; "Not a number")]
    #[case("1.0.0", None; "Too many decimal points")]
    #[case("9.9", Some(9.9); "Float at bounds")]
    fn test_parse_number(s: &str, expected: Option<f64>) -> Result<(), ParseError<LineCol>> {
        let result = expression_parser::decimal::<f64>(s);
        if let Some(expected) = expected {
            assert_eq!(result?, expected);
        } else {
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    /// A unit with a name whose characters are on the boundaries of the allowed characters parses correctly.
    fn test_parse_unit_at_bounds() -> Result<(), ParseError<LineCol>> {
        // Given a unit cache with a unit that has boundary characters
        let mut registry = Registry::new();
        let _ = registry
            .units
            .insert("zZ_".into(), Arc::new(BaseUnit::clone(&UNIT_METER)));

        // The unit parses as expected.
        let result = expression_parser::unit_expression("zZ_", &registry)?;
        assert_eq!(
            result,
            Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![])
        );
        Ok(())
    }

    #[case("meter", Some(Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![])); "Basic")]
    #[case("gram", Some(Unit::new(vec![BaseUnit::clone(&UNIT_GRAM)], vec![])); "Gram")]
    #[case("meter / second", Some(Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![BaseUnit::clone(&UNIT_SECOND)])); "Division")]
    #[case("meter * second", Some(Unit::new(vec![BaseUnit::clone(&UNIT_METER), BaseUnit::clone(&UNIT_SECOND)], vec![])); "Multiplication")]
    #[case("meter ** 2", Some(Unit::new(vec![UNIT_METER.powf(2.0)], vec![])); "Exponentiation")]
    #[case(
        "meter^2",
        Some(Unit::new(vec![UNIT_METER.powf(2.0)], vec![]))
        ; "Exponentiation alternate notation"
    )]
    #[case("meter ** 0.5", Some(Unit::new(vec![UNIT_METER.powf(0.5)], vec![])); "Fractional power")]
    #[case("meter + meter", None; "Invalid operator plus")]
    #[case(
        "(meter * gram) / second",
        Some(Unit::new(
            vec![
                BaseUnit::clone(&UNIT_METER),
                BaseUnit::clone(&UNIT_GRAM),
            ],
            vec![BaseUnit::clone(&UNIT_SECOND)],
        ))
        ; "Parentheses"
    )]
    #[case(
        "(meter * second) ** 2",
        Some(Unit::new(
            vec![UNIT_METER.powf(2.0), UNIT_SECOND.powf(2.0)],
            vec![]
        ))
        ; "Parentheses with exponentiation"
    )]
    #[case(
        "((meter) * (second))",
        Some(Unit::new(vec![BaseUnit::clone(&UNIT_METER), BaseUnit::clone(&UNIT_SECOND)], vec![]))
        ; "Parentheses with nesting"
    )]
    #[case(
        "(meter / second) * (second / meter)",
        Some(Unit::new(vec![], vec![]))
        ; "Simplification"
    )]
    #[case(
        "meter / kilometer",
        Some(Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![BaseUnit::clone(&UNIT_KILOMETER)]))
        ; "No reduction"
    )]
    fn test_unit_parsing(s: &str, expected: Option<Unit<f64>>) -> SmootResult<()> {
        let result = s.parse::<Unit<f64>>();
        if let Some(expected) = expected {
            let result = result?;
            assert_eq!(result, expected, "{:#?} != {:#?}", result, expected);
        } else {
            assert!(result.is_err());
        }
        Ok(())
    }

    #[case(
        "1 meter",
        Some(Quantity::new(1.0, Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![])))
        ; "Basic"
    )]
    #[case(
        "1meter",
        Some(Quantity::new(1.0, Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![])))
        ; "Spaces should not matter"
    )]
    #[case(
        "1.0 meter",
        Some(Quantity::new(1.0, Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![])))
        ; "Decimal point"
    )]
    #[case(
        "1 meter + 1 meter",
        Some(Quantity::new(2.0, Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![])))
        ; "Addition"
    )]
    #[case(
        "1 meter + 1 kilometer",
        Some(Quantity::new(1.0 + 1e-3, Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![])))
        ; "Addition with conversion"
    )]
    #[case(
        "1 meter + 1 second",
        None
        ; "Addition with incompatible units"
    )]
    #[case(
        "1 meter - 1 meter",
        Some(Quantity::new(0.0, Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![])))
        ; "Subtraction"
    )]
    #[case(
        "1 meter - 1 kilometer",
        Some(Quantity::new(1.0 - 1e-3, Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![])))
        ; "Subtraction with conversion"
    )]
    #[case(
        "1 meter - 1 second",
        None
        ; "Subtraction with incompatible units"
    )]
    #[case(
        "1 meter * 2 second",
        Some(Quantity::new(
            2.0,
            Unit::new(vec![BaseUnit::clone(&UNIT_METER), BaseUnit::clone(&UNIT_SECOND)], vec![]),
        ))
        ; "Multiplication"
    )]
    #[case(
        "1 meter * 1 kilometer",
        Some(Quantity::new(
            1e-3,
            Unit::new(vec![BaseUnit::clone(&UNIT_KILOMETER)], vec![]).powf(2.0),
        ))
        ; "Multiplication with conversion"
    )]
    #[case(
        "1 kilometer * 1 meter",
        Some(Quantity::new(
            1e3,
            Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]).powf(2.0),
        ))
        ; "Multiplication with conversion flipped"
    )]
    #[case(
        "1 meter * 1 second * 1 kilometer",
        Some(Quantity::new(
            1e-3,
            Unit::new(vec![BaseUnit::clone(&UNIT_KILOMETER)], vec![]).powf(2.0)
                * Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]),
        ))
        ; "Multiplication with other unit in the middle"
    )]
    #[case(
        "1 meter / 2 second",
        Some(Quantity::new(
            0.5,
            Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![BaseUnit::clone(&UNIT_SECOND)]),
        ))
        ; "Division"
    )]
    #[case(
        "1 meter / 1 kilometer",
        Some(Quantity::new(
            1e-3,
            Unit::new(vec![], vec![]),
        ))
        ; "Division with conversion"
    )]
    #[case(
        "1 kilometer / 1 meter",
        Some(Quantity::new(
            1e3,
            Unit::new(vec![], vec![]),
        ))
        ; "Division with conversion flipped"
    )]
    #[case(
        "1 meter ** 2",
        Some(Quantity::new(
            1.0,
            Unit::new(
                vec![BaseUnit::clone(&UNIT_METER).powf(2.0)],
                vec![]
            ),
        ))
        ; "Exponentiation"
    )]
    #[case(
        "1 meter ** 0.5",
        Some(Quantity::new(
            1.0,
            Unit::new(
                vec![BaseUnit::clone(&UNIT_METER).powf(0.5)],
                vec![]
            ),
        ))
        ; "Fractional powers"
    )]
    #[case(
        "1 meter ** -1",
        Some(Quantity::new(
            1.0,
            Unit::new(
                vec![],
                vec![BaseUnit::clone(&UNIT_METER)],
            ),
        ))
        ; "Negative powers"
    )]
    #[case(
        "(1 meter * 2 second)^2",
        Some(Quantity::new(
            4.0,
            Unit::new(
                vec![
                    BaseUnit::clone(&UNIT_METER).powf(2.0),
                    BaseUnit::clone(&UNIT_SECOND).powf(2.0),
                ],
                vec![]
            ),
        ))
        ; "Exponentiation with parentheses"
    )]
    #[case(
        "1 (meter ** 2) * (meter ** 0.5)",
        Some(Quantity::new(
            1.0,
            Unit::new(
                vec![BaseUnit::clone(&UNIT_METER).powf(2.5)],
                vec![]
            ),
        ))
        ; "Exponents add"
    )]
    fn test_expression_parsing(s: &str, expected: Option<Quantity<f64, f64>>) -> SmootResult<()> {
        let result = s.parse::<Quantity<f64, f64>>();
        if let Some(expected) = expected {
            let result = result?;
            assert_eq!(result, expected, "{:#?} != {:#?}", result, expected);
        } else {
            assert!(result.is_err());
        }
        Ok(())
    }
}
