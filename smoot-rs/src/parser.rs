use std::str::FromStr;

use num_traits::Float;

use crate::registry::Registry;
use crate::utils::Powi;
use crate::{quantity::Quantity, unit::Unit};

trait ParsableFloat: FromStr + Float {}
impl ParsableFloat for f64 {}

// Unit and quantity expression parser.
peg::parser! {
    pub(crate) grammar expression_parser() for str {
        /// Whitespace.
        rule __() = [' ' | '\n' | '\t']*

        /// Allow any non-whitespace character except for reserved characters.
        rule symbol() -> &'input str
            = sym:$([^ ' ' | '\n' | '\t' | '-' | '[' | ']' | '(' | ')' | '=' | ';' | ':' | ',' | '*' | '/' | '^' | '#']+)
            { sym }
        rule sign() = ['-' | '+']
        rule digits() = [c if c.is_ascii_digit()]+

        pub rule integer() -> i32
            = num:$(sign()?digits()) !['.' | 'e' | 'E']
            {? num.parse::<i32>().or(Err("Invalid integer number")) }

        pub rule decimal<N: ParsableFloat>() -> N
            = num:$(sign()?(digits()".")?digits()(['e' | 'E']sign()?digits())?)
            {? num.parse::<N>().or(Err("Invalid decimal number")) }

        /// Parses a basic unit. Will fail if the unit string is not found in the specified unit cache.
        rule unit(registry: &Registry) -> Unit
            = sym:symbol()
            {?
                registry
                    .get_unit(sym)
                    .cloned()
                    .map(|u| Unit::new(vec![u], vec![]))
                    .ok_or("Unknown unit")
            }

        rule reciprocal_unit(registry: &Registry) -> Unit
            = d:decimal::<f64>() __ "/" __ u:unit_expression(registry)
            {
                let mut u = u.powi(-1);
                u.iscale(d);
                u
            }

        /// Core parser with operator precedence.
        /// We only need to support a few basic operators; addition and subtraction don't make sense for units.
        pub rule unit_expression(registry: &Registry) -> Unit
            = precedence!
            {
                // multiplication
                u1:(@) __ "*" __ u2:@ { u1 * u2 }
                u1:(@) __ u2:@ { u1 * u2 }
                d:decimal() __ "*" __ u:@ { Unit::new_constant(d) * u }
                d:decimal() __ u:@ { Unit::new_constant(d) * u }
                u:@ __ "*" __ d:decimal() { u * Unit::new_constant(d) }
                u:@ __ d:decimal() { u * Unit::new_constant(d) }
                // division
                u1:(@) __ "/" __ u2:@ { u1 / u2 }
                d:decimal() __ "/" __ u:@ { Unit::new_constant(d) / u }
                u:@ __ "/" __ d:decimal() { u / Unit::new_constant(d) }
                --
                u:@ __ ("**" / "^") __ n:integer() { u.powi(n) }
                --
                u:unit(registry) { u }
                "(" __ expr:unit_expression(registry) __ ")" { expr }
            }

        rule quantity(registry: &Registry) -> Quantity<f64, f64>
            = n:decimal()? __ u:unit_expression(registry)
            { Quantity::new(n.unwrap_or(1.0), u) }

        /// Add operator requires conditional parsing to handle incompatible units.
        rule quantity_add(registry: &Registry) -> Quantity<f64, f64>
            = q1:quantity(registry) __ "+" __ q2:expression(registry)
            {? (q1 + q2).or(Err("Incompatible units")) }

        /// Subtract operator requires conditional parsing to handle incompatible units.
        rule quantity_sub(registry: &Registry) -> Quantity<f64, f64>
            = q1:quantity(registry) __ "-" __ q2:expression(registry)
            {? (q1 - q2).or(Err("Incompatible units")) }

        pub rule expression(registry: &Registry) -> Quantity<f64, f64>
            = precedence!
            {
                q:quantity_add(registry) { q }
                q:quantity_sub(registry) { q }
                --
                q1:(@) __ "*" __ q2:@ { q1 * q2 }
                q1:(@) __ "/" __ q2:@ { q1 / q2 }
                d:decimal::<f64>() __ "/" __ u:unit_expression(registry) { d / u }
                --
                q1:@ __ "**" __ n:integer() !['.'] { q1.powi(n) }
                q1:@ __ "^" __ n:integer() !['.'] { q1.powi(n) }
                --
                q:quantity(registry) { q }
                "-" q:expression(registry) { -q }
                "(" __ expr:expression(registry) __ ")" { expr }
                n:decimal() { Quantity::new_dimensionless(n) }
            }
    }
}

#[cfg(test)]
mod test_expression_parser {
    use crate::{
        base_unit::BaseUnit,
        error::{SmootError, SmootResult},
        test_utils::TEST_REGISTRY,
    };

    use super::*;
    use peg::{error::ParseError, str::LineCol};
    use std::sync::LazyLock;
    use test_case::case;

    static UNIT_METER: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("meter").expect("No unit 'meter'"));
    static UNIT_KILOMETER: LazyLock<&BaseUnit> = LazyLock::new(|| {
        TEST_REGISTRY
            .get_unit("kilometer")
            .expect("No unit 'kilometer'")
    });
    static UNIT_SECOND: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("second").expect("No unit 'second'"));
    static UNIT_GRAM: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("gram").expect("No unit 'gram'"));
    static UNIT_NEWTON: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("newton").expect("No unit 'newton'"));
    static UNIT_JOULE: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("joule").expect("No unit 'joule'"));
    static UNIT_PERCENT: LazyLock<&BaseUnit> = LazyLock::new(|| {
        TEST_REGISTRY
            .get_unit("percent")
            .expect("No unit 'percent'")
    });

    fn parse_unit(s: &str, registry: &Registry) -> SmootResult<Unit> {
        expression_parser::unit_expression(s, registry)
            .map_err(|_e| SmootError::ExpressionError(format!("Invalid unit expression {}", s)))
    }

    fn parse_quantity(s: &str, registry: &Registry) -> SmootResult<Quantity<f64, f64>> {
        expression_parser::expression(s, registry)
            .map(|mut q| {
                q.ito_reduced_units();
                q
            })
            .map_err(|_| SmootError::ExpressionError(format!("Invalid quantity expression {}", s)))
    }

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
            .insert("zZ_".into(), BaseUnit::clone(&UNIT_METER));

        // The unit parses as expected.
        let result = expression_parser::unit_expression("zZ_", &registry)?;
        assert_eq!(
            result,
            Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![])
        );
        Ok(())
    }

    #[case("meter", Some(Unit::new(vec![UNIT_METER.clone()], vec![])); "Basic")]
    #[case("gram", Some(Unit::new(vec![UNIT_GRAM.clone()], vec![])); "Gram")]
    #[case("meter / second", Some(Unit::new(vec![UNIT_METER.clone()], vec![UNIT_SECOND.clone()])); "Division")]
    #[case("meter * second", Some(Unit::new(vec![UNIT_METER.clone(), UNIT_SECOND.clone()], vec![])); "Multiplication")]
    #[case("meter second", Some(Unit::new(vec![UNIT_METER.clone(), UNIT_SECOND.clone()], vec![])); "Implicit multiplication")]
    #[case("meter ** 2", Some(Unit::new(vec![UNIT_METER.powi(2)], vec![])); "Exponentiation")]
    #[case(
        "meter^2",
        Some(Unit::new(vec![UNIT_METER.powi(2)], vec![]))
        ; "Exponentiation alternate notation"
    )]
    #[case(
        "meter^-1",
        Some(Unit::new(vec![], vec![UNIT_METER.clone()]))
        ; "Reciprocal power"
    )]
    #[case("meter ** 0.5", None; "Fractional power is invalid")]
    #[case("meter + meter", None; "Invalid operator plus")]
    #[case(
        "(meter * gram) / second",
        Some(Unit::new(
            vec![UNIT_METER.clone(), UNIT_GRAM.clone()],
            vec![BaseUnit::clone(&UNIT_SECOND)],
        ))
        ; "Parentheses"
    )]
    #[case(
        "(meter * second) ** 2",
        Some(Unit::new(
            vec![UNIT_METER.powi(2), UNIT_SECOND.powi(2)],
            vec![]
        ))
        ; "Parentheses with exponentiation"
    )]
    #[case(
        "((meter) * (second))",
        Some(Unit::new(vec![UNIT_METER.clone(), UNIT_SECOND.clone()], vec![]))
        ; "Parentheses with nesting"
    )]
    #[case(
        "(meter / second) * (second / meter)",
        Some(Unit::new_dimensionless())
        ; "Simplification"
    )]
    #[case(
        "meter / kilometer",
        Some(Unit::new(vec![UNIT_METER.clone()], vec![UNIT_KILOMETER.clone()]))
        ; "No reduction"
    )]
    #[case(
        "joule / newton",
        Some(Unit::new(vec![UNIT_JOULE.clone()], vec![UNIT_NEWTON.clone()]))
        ; "Composite base units"
    )]
    #[case(
        "1 / meter",
        Some(Unit::new(vec![BaseUnit::new_constant(1.0)], vec![UNIT_METER.clone()]))
        ; "Reciprocal unit"
    )]
    #[case(
        "2 / meter",
        Some(Unit::new(vec![BaseUnit::new_constant(2.0)], vec![UNIT_METER.clone()]))
        ; "Scaled reciprocal unit"
    )]
    #[case(
        "2 * meter",
        Some(Unit::new(vec![BaseUnit::new_constant(2.0), UNIT_METER.clone()], vec![]))
        ; "Scaled unit with operator"
    )]
    #[case(
        "2 meter",
        Some(Unit::new(vec![BaseUnit::new_constant(2.0), UNIT_METER.clone()], vec![]))
        ; "Scaled unit with implicit operator"
    )]
    #[case(
        "%",
        Some(Unit::new(vec![UNIT_PERCENT.clone()], vec![]))
        ; "Special symbol"
    )]
    fn test_unit_parsing(s: &str, expected: Option<Unit>) -> SmootResult<()> {
        let result = parse_unit(s, &TEST_REGISTRY);
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
        Some(Quantity::new(1.0, Unit::new(vec![UNIT_METER.clone()], vec![])))
        ; "Basic"
    )]
    #[case(
        "1meter",
        Some(Quantity::new(1.0, Unit::new(vec![UNIT_METER.clone()], vec![])))
        ; "Spaces should not matter"
    )]
    #[case(
        "1.0 meter",
        Some(Quantity::new(1.0, Unit::new(vec![UNIT_METER.clone()], vec![])))
        ; "Decimal point"
    )]
    #[case(
        "1 meter + 1 meter",
        Some(Quantity::new(2.0, Unit::new(vec![UNIT_METER.clone()], vec![])))
        ; "Addition"
    )]
    #[case(
        "1 meter + 1 kilometer",
        Some(Quantity::new(1001.0, Unit::new(vec![UNIT_METER.clone()], vec![])))
        ; "Addition with conversion"
    )]
    #[case(
        "1 meter + 1 second",
        None
        ; "Addition with incompatible units"
    )]
    #[case(
        "1 meter - 1 meter",
        Some(Quantity::new(0.0, Unit::new(vec![UNIT_METER.clone()], vec![])))
        ; "Subtraction"
    )]
    #[case(
        "1 meter - 1 kilometer",
        Some(Quantity::new(-999.0, Unit::new(vec![UNIT_METER.clone()], vec![])))
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
            Unit::new(vec![UNIT_METER.clone(), UNIT_SECOND.clone()], vec![]),
        ))
        ; "Multiplication"
    )]
    #[case(
        "1 meter * 1 kilometer",
        Some(Quantity::new(
            1e-3,
            Unit::new(vec![UNIT_KILOMETER.clone()], vec![]).powi(2),
        ))
        ; "Multiplication with conversion"
    )]
    #[case(
        "1 meter * 1 kilometer**2",
        Some(Quantity::new(
            1e-3,
            Unit::new(vec![UNIT_KILOMETER.clone()], vec![]).powi(3),
        ))
        ; "Multiplication with conversion 2"
    )]
    #[case(
        "1 meter**2 * 1 kilometer**2",
        Some(Quantity::new(
            1e-6,
            Unit::new(vec![UNIT_KILOMETER.clone()], vec![]).powi(4),
        ))
        ; "Multiplication with conversion 3"
    )]
    #[case(
        "1 kilometer * 1 meter",
        Some(Quantity::new(
            1e3,
            Unit::new(vec![UNIT_METER.clone()], vec![]).powi(2),
        ))
        ; "Multiplication with conversion flipped"
    )]
    #[case(
        "1 meter * 1 second * 1 kilometer",
        Some(Quantity::new(
            1e-3,
            Unit::new(vec![UNIT_KILOMETER.clone()], vec![]).powi(2)
                * Unit::new(vec![UNIT_SECOND.clone()], vec![]),
        ))
        ; "Multiplication with other unit in the middle"
    )]
    #[case(
        "1 meter / 2 second",
        Some(Quantity::new(
            0.5,
            Unit::new(vec![UNIT_METER.clone()], vec![UNIT_SECOND.clone()]),
        ))
        ; "Division"
    )]
    #[case(
        "1 meter / 1 kilometer",
        Some(Quantity::new(
            1e-3,
            Unit::new_dimensionless(),
        ))
        ; "Division with conversion"
    )]
    #[case(
        "1 kilometer / 1 meter",
        Some(Quantity::new(
            1e3,
            Unit::new_dimensionless(),
        ))
        ; "Division with conversion flipped"
    )]
    #[case(
        "1 meter ** 2",
        Some(Quantity::new(
            1.0,
            Unit::new(
                vec![UNIT_METER.powi(2)],
                vec![]
            ),
        ))
        ; "Exponentiation"
    )]
    #[case(
        "1 meter ** 0.5",
        None
        ; "Fractional powers are invalid"
    )]
    #[case(
        "1 meter ** -1",
        Some(Quantity::new(
            1.0,
            Unit::new(
                vec![],
                vec![UNIT_METER.clone()],
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
                    UNIT_METER.powi(2),
                    UNIT_SECOND.powi(2),
                ],
                vec![]
            ),
        ))
        ; "Exponentiation with parentheses"
    )]
    #[case(
        "1 / meter",
        Some(Quantity::new(
            1.0,
            Unit::new(
                vec![],
                vec![UNIT_METER.clone()]
            ),
        ))
        ; "Reciprocal unit"
    )]
    fn test_expression_parsing(s: &str, expected: Option<Quantity<f64, f64>>) -> SmootResult<()> {
        let result = parse_quantity(s, &TEST_REGISTRY);
        if let Some(expected) = expected {
            let result = result?;
            assert_eq!(result, expected, "{:#?} != {:#?}", result, expected);
        } else {
            assert!(result.is_err());
        }
        Ok(())
    }
}
