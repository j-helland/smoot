#[derive(Debug, PartialEq)]
pub(crate) struct PrefixDefinition<'a> {
    pub(crate) name: &'a str,
    pub(crate) multiplier: f64,
    pub(crate) aliases: Vec<&'a str>,
    pub(crate) lineno: usize,
}

#[derive(Debug, PartialEq)]
pub(crate) struct DimensionDefinition<'a> {
    pub(crate) name: &'a str,
    pub(crate) dimension: &'a str,
    pub(crate) modifiers: Vec<(&'a str, f64)>,
    pub(crate) aliases: Vec<&'a str>,
    pub(crate) lineno: usize,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct UnitDefinition<'a> {
    pub(crate) name: String,
    pub(crate) expression: ParseTree,
    pub(crate) modifiers: Vec<(&'a str, f64)>,
    pub(crate) aliases: Vec<&'a str>,
    pub(crate) lineno: usize,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Operator {
    Invalid(String),
    Mul,
    Div,
    Pow,
    Assign,
    AssignAlias,
}
impl From<&str> for Operator {
    fn from(value: &str) -> Self {
        match value {
            "*" => Self::Mul,
            "/" => Self::Div,
            "^" | "**" => Self::Pow,
            "=" => Self::Assign,
            _ => Self::Invalid(value.into()),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum NodeData {
    Op(Operator),
    Integer(i32),
    Decimal(f64),
    Symbol(String),
    Dimension(String),
}
impl From<i32> for NodeData {
    fn from(value: i32) -> Self {
        Self::Integer(value)
    }
}
impl From<f64> for NodeData {
    fn from(value: f64) -> Self {
        Self::Decimal(value)
    }
}
impl<'a> From<&'a str> for NodeData {
    fn from(value: &'a str) -> Self {
        Self::Symbol(value.into())
    }
}
impl From<String> for NodeData {
    fn from(value: String) -> Self {
        Self::Symbol(value)
    }
}
impl From<Operator> for NodeData {
    fn from(value: Operator) -> Self {
        Self::Op(value)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct ParseTree {
    pub(crate) val: NodeData,
    pub(crate) left: Option<Box<ParseTree>>,
    pub(crate) right: Option<Box<ParseTree>>,
}
impl ParseTree {
    pub fn new(val: NodeData, left: ParseTree, right: ParseTree) -> Self {
        Self {
            val,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }

    pub fn node(val: NodeData) -> Self {
        Self {
            val,
            left: None,
            right: None,
        }
    }
}
impl<T> From<T> for ParseTree
where
    T: Into<NodeData>,
{
    fn from(value: T) -> Self {
        ParseTree::node(value.into())
    }
}

peg::parser! {
    pub(crate) grammar registry_parser() for str {
        /// Whitespace.
        rule __() = [' ' | '\n' | '\t']*


        /// Allow any non-whitespace character except for reserved characters.
        rule symbol() -> &'input str
            = sym:$([^ ' ' | '\n' | '\t' | '-' | '[' | ']' | '(' | ')' | '=' | ';' | ':' | ',' | '*' | '/' | '^' | '#']+)
            { sym }
        rule dimension() -> &'input str
            = dim:$("[" symbol()? "]")
            { dim }

        rule eq() = __ "=" __
        rule sign() = ['-' | '+']
        rule digits() = [c if c.is_ascii_digit()]+
        rule integer() -> i32
            = num:$("-"?digits()) !['.' | 'e' | 'E']
            {? num.parse::<i32>().or(Err("Invalid integer number")) }
        rule decimal() -> f64
            = num:$(sign()?(digits()".")?digits()(['e' | 'E']sign()?digits())?)
            {? num.parse::<f64>().or(Err("Invalid decimal number")) }

        rule numeric_expression() -> f64
            = precedence! {
                n1:(@) __ "+" __ n2:@ { n1 + n2 }
                n1:(@) __ "-" __ n2:@ { n1 - n2 }
                --
                n1:(@) __ "*" __ n2:@ { n1 * n2 }
                n1:(@) __ "/" __ n2:@ { n1 / n2 }
                --
                n1:(@) __ "**" __ n2:@ { n1.powf(n2) }
                n1:(@) __ "^" __ n2:@ { n1.powf(n2) }
                --
                n1:decimal() { n1 }
            }

        rule modifiers() -> (&'input str, f64)
            = __ ";" __ modifier:symbol() ":" __ value:numeric_expression()
            { (modifier, value) }

        rule aliases() -> &'input str = eq() s:symbol() { s }

        /// Matches comments of the form `# comment comment comment`.
        pub rule comment() -> &'input str
            = __ "#"+ __ rest:$([^'\n']*)
            { rest }

        pub rule unit_expression() -> ParseTree
            // Do not handle assignment operator `=` here because it conflicts with alias definitions
            // e.g. `unit = <expr> = alias1 = alias2 = ...`
            = precedence! {
                u1:(@) __ u2:@
                    { ParseTree::new(Operator::Mul.into(), u1, u2) }
                u1:(@) __ op:$("*" / "/") __ u2:@
                    { ParseTree::new(Operator::from(op).into(), u1, u2) }
                --
                u1:(@) __ op:$("^" / "**") __ u2:@
                    { ParseTree::new(Operator::from(op).into(), u1, u2) }
                --
                "(" __ expr:unit_expression() __ ")" { expr }
                --
                i:integer() { i.into() }
                d:decimal() { d.into() }
                sym:symbol() { NodeData::Symbol(sym.into()).into() }
                dim:dimension() { NodeData::Dimension(dim.into()).into() }
            }

        pub rule unit_with_aliases(lineno: usize) -> UnitDefinition<'input>
            = name:symbol() eq() unit:unit_expression()
                modifiers:(modifiers()*)
                aliases:(aliases()*)
                comment()?
            {
                UnitDefinition {
                    name: name.into(),
                    expression: ParseTree::new(
                        Operator::Assign.into(),
                        NodeData::Symbol(name.into()).into(),
                        unit,
                    ),
                    modifiers,
                    aliases,
                    lineno,
                }
            }

        pub rule dimension_definition(lineno: usize) -> DimensionDefinition<'input>
            = name:symbol() eq() dimension:dimension()
                modifiers:(modifiers()*)
                aliases:(aliases()*)
                comment()?
            { DimensionDefinition { name, dimension, modifiers, aliases, lineno } }

        pub rule derived_dimension() -> ParseTree
            = dim:dimension() eq() expr:unit_expression()
                comment()?
            {
                ParseTree::new(
                    Operator::Assign.into(),
                    NodeData::Dimension(dim.into()).into(),
                    expr,
                )
            }

        pub rule prefix_definition(lineno: usize) -> PrefixDefinition<'input>
            = name:symbol() "-" eq() multiplier:decimal()
                // Prefix aliases are special: they have a `-` suffix.
                aliases:((eq() s:symbol() "-"? { s })*)
                comment()?
            { PrefixDefinition { name, multiplier, aliases, lineno } }
    }
}

#[cfg(test)]
mod test_unit_parser {
    use super::*;

    use peg::{error::ParseError, str::LineCol};
    use test_case::case;

    use super::registry_parser;

    // #[test]
    #[case(
        "1.380649e-23 J K^-1",
        Some(ParseTree::new(
            Operator::Mul.into(),
            ParseTree::new(
                Operator::Mul.into(),
                1.380649e-23.into(),
                "J".into(),
            ),
            ParseTree::new(
                Operator::Pow.into(),
                "K".into(),
                (-1).into(),
            )
        ))
        ; "Exponent takes precendence"
    )]
    #[case(
        "ℎ / (2 * π)",
        Some(ParseTree::new(
            Operator::Div.into(),
            "ℎ".into(),
            ParseTree::new(
                Operator::Mul.into(),
                2.into(),
                "π".into(),
            )
        ))
        ; "Parentheticals have precendence"
    )]
    #[case(
        "a / (b / (c * d))",
        Some(ParseTree::new(
            Operator::Div.into(),
            "a".into(),
            ParseTree::new(
                Operator::Div.into(),
                "b".into(),
                ParseTree::new(
                    Operator::Mul.into(),
                    "c".into(),
                    "d".into(),
                )
            )
        ))
        ; "Nested parentheticals"
    )]
    #[case(
        "1 / [viscosity]",
        Some(ParseTree::new(
            Operator::Div.into(),
            1.into(),
            NodeData::Dimension("[viscosity]".into()).into(),
        ))
        ; "Dimension expression"
    )]
    #[case(
        "5 / 9 * kelvin",
        Some(ParseTree::new(
            Operator::Mul.into(),
            ParseTree::new(
                Operator::Div.into(),
                5.into(),
                9.into(),
            ),
            "kelvin".into(),
        ))
        ; "Associativity"
    )]
    #[case(
        "meter ** 2",
        Some(ParseTree::new(
            Operator::Pow.into(),
            "meter".into(),
            2.into(),
        ))
        ; "Power"
    )]
    #[case(
        "1 / second ** 2",
        Some(ParseTree::new(
            Operator::Div.into(),
            1.into(),
            ParseTree::new(
                Operator::Pow.into(),
                "second".into(),
                2.into(),
            ),
        ))
        ; "Inverted power"
    )]
    fn test_unit_expression(
        expression: &str,
        expected: Option<ParseTree>,
    ) -> Result<(), ParseError<LineCol>> {
        let result = registry_parser::unit_expression(expression);
        if let Some(expected) = expected {
            assert_eq!(result?, expected);
        } else {
            assert!(result.is_err(), "Expected an error but got:\n{:#?}", result);
        }
        Ok(())
    }

    #[case(
        "speed_of_light = 299792458 m/s = c = c_0",
        Some(UnitDefinition {
            name: "speed_of_light".into(),
            expression: ParseTree::new(
                Operator::Assign.into(),
                "speed_of_light".into(),
                ParseTree::new(
                    Operator::Div.into(),
                    ParseTree::new(
                        Operator::Mul.into(),
                        299792458.into(),
                        "m".into(),
                    ),
                    "s".into()
                )
            ),
            modifiers: vec![],
            aliases: vec!["c", "c_0"],
            lineno: 0,
        })
        ; "Assignment operator is handled and aliases are parsed"
    )]
    #[case(
        "tansec = 4.8481368111333441675396429478852851658848753880815e-6  # comment",
        Some(UnitDefinition {
            name: "tansec".into(),
            expression: ParseTree::new(
                Operator::Assign.into(),
                "tansec".into(),
                4.8481368111333441675396429478852851658848753880815e-6.into(),
            ),
            modifiers: vec![],
            aliases: vec![],
            lineno: 0,
        })
        ; "Assignment with trivial expression and comment"
    )]
    #[case(
        "decibel = 1 ; logbase: 10; logfactor: 10 = dB",
        Some(UnitDefinition {
            name: "decibel".into(),
            expression: ParseTree::new(
                Operator::Assign.into(),
                "decibel".into(),
                1.into(),
            ),
            modifiers: vec![("logbase", 10.0), ("logfactor", 10.0)],
            aliases: vec!["dB"],
            lineno: 0,
        })
        ; "Modifiers"
    )]
    #[case(
        "x =  1e-21",
        Some(UnitDefinition {
            name: "x".into(),
            expression: ParseTree::new(
                Operator::Assign.into(),
                "x".into(),
                1e-21.into(),
            ),
            modifiers: vec![],
            aliases: vec![],
            lineno: 0,
        })
        ; "Scientific notation"
    )]
    fn unit_with_aliases(
        expression: &str,
        expected: Option<UnitDefinition>,
    ) -> Result<(), ParseError<LineCol>> {
        let result = registry_parser::unit_with_aliases(expression, 0);
        if let Some(expected) = expected {
            assert_eq!(result?, expected);
        } else {
            assert!(result.is_err(), "Expected error but got:\n{:#?}", result);
        }
        Ok(())
    }

    #[case(
        "[momentum] = [length] * [mass] / [time]",
        Some(ParseTree::new(
            Operator::Assign.into(),
            NodeData::Dimension("[momentum]".into()).into(),
            ParseTree::new(
                Operator::Div.into(),
                ParseTree::new(
                    Operator::Mul.into(),
                    NodeData::Dimension("[length]".into()).into(),
                    NodeData::Dimension("[mass]".into()).into(),
                ),
                NodeData::Dimension("[time]".into()).into(),
            )
        ))
        ; "Basic derived dimension expression"
    )]
    fn test_derived_dimension(
        expression: &str,
        expected: Option<ParseTree>,
    ) -> Result<(), ParseError<LineCol>> {
        let result = registry_parser::derived_dimension(expression);
        if let Some(expected) = expected {
            assert_eq!(result?, expected);
        } else {
            assert!(result.is_err(), "Expected error but got:\n{:#?}", result);
        }
        Ok(())
    }

    #[case(
        "micro- = 1e-6  = µ- = μ- = u- = mu- = mc-  # comment",
        Some(PrefixDefinition {
            name: "micro",
            multiplier: 1e-6,
            aliases: vec!["µ", "μ", "u", "mu", "mc"],
            lineno: 0,
        })
        ; "Prefix definition with aliases and comment"
    )]
    #[case(
        "semi- = 0.5 = _ = demi-",
        Some(PrefixDefinition {
            name: "semi",
            multiplier: 0.5,
            aliases: vec!["_", "demi"],
            lineno: 0,
        })
    )]
    fn test_prefix_definition(
        expression: &str,
        expected: Option<PrefixDefinition>,
    ) -> Result<(), ParseError<LineCol>> {
        let result = registry_parser::prefix_definition(expression, 0);
        if let Some(expected) = expected {
            assert_eq!(result?, expected);
        } else {
            assert!(result.is_err(), "Expected error but got:\n{:#?}", result);
        }
        Ok(())
    }
}
