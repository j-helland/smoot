use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::Hash;
use std::io::Write;
use std::sync::LazyLock;
use std::{collections::HashMap, fs::read_to_string, sync::Arc};

use bitcode::{Decode, Encode};

use crate::base_unit::{BaseUnit, DIMENSIONLESS};
use crate::error::{SmootError, SmootResult};

use super::registry_parser::{
    registry_parser, NodeData, Operator, ParseTree, PrefixDefinition, UnitDefinition,
};

pub static REGISTRY: LazyLock<Registry> =
    LazyLock::new(|| Registry::default().expect("Failed to load default registry"));

#[derive(Encode, Decode)]
pub struct Registry {
    pub units: HashMap<String, Arc<BaseUnit<f64>>>,
}
impl Registry {
    const DEFAULT_UNITS_FILE: &str = "default_en.txt";
    const REGISTRY_CACHE_FILE: &str = ".registry_cache.smoot";

    pub fn new() -> Self {
        Self {
            units: HashMap::new(),
        }
    }

    pub fn default() -> SmootResult<Self> {
        std::fs::read(Self::REGISTRY_CACHE_FILE).map_or_else(
            |_| {
                let mut new = Self::new();
                new.load_from_file(Self::DEFAULT_UNITS_FILE)?;

                std::fs::File::create(Self::REGISTRY_CACHE_FILE)
                    .and_then(|mut f| f.write(&bitcode::encode(&new)))
                    .map_err(|_| SmootError::FailedToWriteCache(Self::REGISTRY_CACHE_FILE))
                    .map(|_| new)
            },
            |data| {
                bitcode::decode(&data)
                    .map_err(|_| SmootError::FailedToDecodeCache(Self::REGISTRY_CACHE_FILE))
            },
        )
    }

    pub fn clear_cache() {
        std::fs::remove_file(Self::REGISTRY_CACHE_FILE).unwrap_or(())
    }

    pub fn get_unit(&self, key: &str) -> Option<&Arc<BaseUnit<f64>>> {
        self.units.get(key)
    }

    pub fn load_from_file(&mut self, path: &str) -> SmootResult<()> {
        let data = read_to_string(path).unwrap();
        self.parse_definitions(&data)?;
        Ok(())
    }

    fn parse_definitions(&mut self, data: &str) -> SmootResult<()> {
        let lines = data
            .lines()
            .map(|line| line.trim())
            .enumerate()
            .map(|(lineno, line)| (lineno + 1, line))
            .filter(|(_, line)| !line.is_empty() && !Self::is_comment(line));

        let mut dim_defs = Vec::new();
        let mut unit_defs: HashMap<String, UnitDefinition> = HashMap::new();
        let mut derived_dim_defs = HashMap::new();
        let mut prefix_defs = HashMap::new();

        for (lineno, line) in lines {
            if let Ok(dim_def) = registry_parser::dimension_definition(line, lineno) {
                dim_defs.push(dim_def);
            } else if let Ok(unit_def) = registry_parser::unit_with_aliases(line, lineno) {
                for &alias in unit_def.aliases.iter().filter(|&&a| a.ne("_")) {
                    Self::try_insert(lineno, &mut unit_defs, alias.into(), unit_def.clone())?;
                }
                Self::try_insert(lineno, &mut unit_defs, unit_def.name.clone(), unit_def)?;
            } else if let Ok(expr) = registry_parser::derived_dimension(line) {
                if let NodeData::Op(Operator::Assign) = &expr.val {
                    if let NodeData::Dimension(dimension) = &expr.left.as_ref().unwrap().val {
                        // TODO: handle better
                        if let Some(overwrite) = derived_dim_defs.insert(dimension.clone(), expr) {
                            panic!(
                                "Line '{}' overwrote previous derived dimension {:?}",
                                line, overwrite
                            );
                        }
                    } else {
                        println!("{:#?}", expr);
                        panic!(
                            "Derived dimension expression {} does not assign to a dimension",
                            line
                        );
                    }
                }
            } else if let Ok(prefix_def) = registry_parser::prefix_definition(line, lineno) {
                Self::try_insert(lineno, &mut prefix_defs, prefix_def.name, prefix_def)?;
            } else {
                println!("line:{} Unhandled line '{}'", lineno, line);
            }
        }

        //==================================================
        // Generate prefix assignment expressions
        //==================================================
        // Add prefixes for regular unit definitions
        for name in unit_defs.keys().cloned().collect::<Vec<_>>() {
            let def = unit_defs.get(&name).unwrap().clone();

            for prefix_def in prefix_defs.values() {
                let f_make_new_def = |new_name: String| UnitDefinition {
                    name: new_name.clone(),
                    // Create an expression like `kilometer = 1000 * meter`, where `M` is the prefix multiplier.
                    expression: ParseTree::new(
                        Operator::Assign.into(),
                        new_name.into(),
                        ParseTree::new(
                            Operator::Mul.into(),
                            ParseTree::from(prefix_def.multiplier),
                            name.clone().into(),
                        ),
                    ),
                    modifiers: def.modifiers.clone(),
                    // Clear the aliases to avoid redefining them
                    aliases: vec![],
                    lineno: def.lineno,
                };

                let f_make_new_alias = |new_name: String, from: String| UnitDefinition {
                    name: new_name.clone(),
                    // Create an expression like `km = kilometer`.
                    expression: ParseTree::new(
                        Operator::AssignAlias.into(),
                        new_name.into(),
                        from.into(),
                    ),
                    modifiers: def.modifiers.clone(),
                    // Clear the aliases to avoid redefining them
                    aliases: vec![],
                    lineno: def.lineno,
                };

                // We only need to do the prefix aliases because the unit aliases have already been added to the unit_defs map.
                let mut prefix_name: String = prefix_def.name.into();
                prefix_name.push_str(&name);

                for &prefix_alias in prefix_def.aliases.iter().filter(|&&a| a.ne("_")) {
                    let mut alias_name: String = prefix_alias.into();
                    alias_name.push_str(&name);

                    let new_alias = f_make_new_alias(alias_name.clone(), prefix_name.clone());
                    Self::insert_or_warn(def.lineno, &mut unit_defs, alias_name, new_alias);
                }

                let new_def = f_make_new_def(prefix_name.clone());
                Self::insert_or_warn(def.lineno, &mut unit_defs, prefix_name, new_def);
            }
        }

        // Add prefixes for dimension definitions
        for def in dim_defs.iter() {
            let name = def.name;

            for prefix_def in prefix_defs.values() {
                let f_make_new_def = |name: String| UnitDefinition {
                    name: name.clone(),
                    // Create an expression like `kilometer = 1000 * meter`, where `M` is the prefix multiplier.
                    expression: ParseTree::new(
                        Operator::Assign.into(),
                        name.into(),
                        ParseTree::new(
                            Operator::Mul.into(),
                            ParseTree::from(prefix_def.multiplier),
                            def.name.into(),
                        ),
                    ),
                    modifiers: def.modifiers.clone(),
                    // Clear the aliases to avoid redefining them
                    aliases: vec![],
                    lineno: def.lineno,
                };

                let f_make_new_alias = |name: String, from: String| UnitDefinition {
                    name: name.clone(),
                    // Create an expression like `km = kilometer`.
                    expression: ParseTree::new(
                        Operator::AssignAlias.into(),
                        name.into(),
                        from.into(),
                    ),
                    modifiers: def.modifiers.clone(),
                    // Clear the aliases to avoid redefining them
                    aliases: vec![],
                    lineno: def.lineno,
                };

                let mut prefix_name: String = prefix_def.name.into();
                prefix_name.push_str(name);

                // Add all combinations of prefix aliases and unit aliases.
                for alias in def.aliases.iter().filter(|&&a| a.ne("_")) {
                    let mut alias_name: String = prefix_def.name.into();
                    alias_name.push_str(alias);

                    for &prefix_alias in prefix_def.aliases.iter().filter(|&&a| a.ne("_")) {
                        let mut alias_name: String = prefix_alias.into();
                        alias_name.push_str(alias);

                        // Add prefix aliases
                        let new_alias = f_make_new_alias(alias_name.clone(), prefix_name.clone());
                        Self::insert_or_warn(def.lineno, &mut unit_defs, alias_name, new_alias);
                    }

                    // Add prefixed unit aliases
                    let new_alias = f_make_new_alias(alias_name.clone(), prefix_name.clone());
                    Self::insert_or_warn(def.lineno, &mut unit_defs, alias_name, new_alias);
                }

                // Add all prefix aliases to the unit name
                for &prefix_alias in prefix_def.aliases.iter().filter(|&&a| a.ne("_")) {
                    let mut alias_name: String = prefix_alias.into();
                    alias_name.push_str(def.name);

                    let new_alias = f_make_new_alias(alias_name.clone(), prefix_name.clone());
                    Self::insert_or_warn(def.lineno, &mut unit_defs, alias_name, new_alias);
                }

                // Add prefixed unit
                let new_def = f_make_new_def(prefix_name.clone());
                Self::insert_or_warn(def.lineno, &mut unit_defs, prefix_name.clone(), new_def);
            }
        }

        //==================================================
        // Define dimensions
        //==================================================
        let mut next_dimension = 1;
        let mut dimensions = HashMap::new();
        dimensions.insert("[dimensionless]", DIMENSIONLESS);

        // TODO: base units
        dim_defs.into_iter().for_each(|def| {
            let unit = BaseUnit::new(def.name.into(), 1.0, next_dimension);
            self.insert_def(def.name.into(), &def.aliases, Arc::new(unit));

            dimensions.insert(def.dimension, next_dimension);
            next_dimension <<= 1;
        });

        //==================================================
        // Evaluate unit definitions
        //==================================================
        // Make sure that a dimensionless unit is added.
        Self::try_insert(
            0,
            &mut self.units,
            "dimensionless".into(),
            Arc::new(BaseUnit {
                name: "dimensionless".into(),
                multiplier: 1.0,
                power: None,
                unit_type: DIMENSIONLESS,
                dimensionality: vec![],
            }),
        )?;

        self.define_units(&unit_defs, &prefix_defs)
    }

    #[inline(always)]
    fn try_insert<K, V>(
        lineno: usize,
        map: &mut HashMap<K, V>,
        key: K,
        val: V,
    ) -> Result<&mut V, SmootError>
    where
        K: Eq + Hash + Debug,
        V: Debug,
    {
        match map.entry(key) {
            Entry::Vacant(entry) => Ok(entry.insert(val)),
            Entry::Occupied(entry) => Err(SmootError::ConflictingDefiniition(
                lineno,
                format!(
                    "Attempted to overwrite value {:?}\nwith {:?}:{:?}",
                    entry.get(),
                    entry.key(),
                    val
                ),
            )),
        }
    }

    #[inline(always)]
    fn insert_or_warn<K, V>(_lineno: usize, map: &mut HashMap<K, V>, key: K, val: V)
    where
        K: Eq + Hash + Debug,
        V: Debug,
    {
        match map.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(val);
            }
            Entry::Occupied(_entry) => {
                // // TODO: actual logging
                // println!("line:{} Attempted to overwrite value {:?}\nwith {:?}:{:?}", lineno, entry.get(), entry.key(), val);
            }
        }
    }

    fn insert_def(
        &mut self,
        name: String,
        aliases: &Vec<&str>,
        unit: Arc<BaseUnit<f64>>,
    ) -> &Arc<BaseUnit<f64>> {
        for &alias in aliases.iter().filter(|&&a| a.ne("_")) {
            let _ = self.units.entry(alias.into()).or_insert(unit.clone());
        }
        self.units.entry(name).or_insert(unit)
    }

    fn define_units<'a>(
        &mut self,
        unit_defs: &HashMap<String, UnitDefinition>,
        prefix_defs: &'a HashMap<&'a str, PrefixDefinition<'a>>,
    ) -> Result<(), SmootError> {
        for (name, def) in unit_defs.iter() {
            if self.units.contains_key(name) {
                continue;
            }

            let rtree = def.expression.right.as_ref().unwrap();
            let unit = self.traverse_expression_tree(def.lineno, rtree, unit_defs, prefix_defs)?;

            let unit = match def.expression.val {
                // Regular unit definitions like `byte = 8 * bit` should assign the lhs name `byte`.
                NodeData::Op(Operator::Assign) => {
                    let mut unit = BaseUnit::clone(&unit);
                    // Make sure the official name is correct
                    unit.name = name.clone();
                    Arc::new(unit)
                }
                // Generated alias expressions like `@alias km = kilometer` should use the rhs name `kilometer` for consistency.
                NodeData::Op(Operator::AssignAlias) => unit,
                _ => {
                    return Err(SmootError::InvalidUnitExpression(
                        def.lineno,
                        format!("{:#?}", def.expression),
                    ))
                }
            };

            let _ = self.insert_def(def.name.clone(), &def.aliases, unit);
        }
        Ok(())
    }

    fn get_or_create_unit<'a>(
        &mut self,
        lineno: usize,
        symbol: &'a str,
        unit_defs: &HashMap<String, UnitDefinition>,
        prefix_defs: &'a HashMap<&'a str, PrefixDefinition<'a>>,
    ) -> Result<&Arc<BaseUnit<f64>>, SmootError> {
        if self.units.contains_key(symbol) {
            return Ok(self.units.get(symbol).unwrap());
        }

        let def = unit_defs
            .get(symbol)
            .ok_or_else(|| SmootError::UnknownUnit(lineno, symbol.into()))?;
        let rtree = def.expression.right.as_ref().ok_or_else(|| {
            SmootError::InvalidUnitExpression(lineno, "Missing right expression tree".into())
        })?;

        // Make sure the official name is correct
        let unit = self.traverse_expression_tree(def.lineno, rtree, unit_defs, prefix_defs)?;
        let mut unit = BaseUnit::clone(&unit);
        unit.name = def.name.clone();

        // Insert name and aliases
        Ok(self.insert_def(def.name.clone(), &def.aliases, Arc::new(unit)))
    }

    fn traverse_expression_tree<'a>(
        &mut self,
        lineno: usize,
        tree: &'a ParseTree,
        unit_defs: &HashMap<String, UnitDefinition>,
        prefix_defs: &'a HashMap<&'a str, PrefixDefinition<'a>>,
    ) -> Result<Arc<BaseUnit<f64>>, SmootError> {
        if tree.left.is_none() && tree.right.is_none() {
            // Leaf
            let unit = match &tree.val {
                NodeData::Integer(int) => Arc::new(BaseUnit::new_constant(f64::from(*int))),
                NodeData::Decimal(float) => Arc::new(BaseUnit::new_constant(*float)),
                NodeData::Symbol(symbol) => self
                    .get_or_create_unit(lineno, symbol, unit_defs, prefix_defs)?
                    .clone(),
                NodeData::Op(op) => {
                    return Err(SmootError::InvalidOperator(lineno, format!("{:?}", op)));
                }
                NodeData::Dimension(dim) => {
                    return Err(SmootError::UnexpectedDimension(lineno, dim.to_owned()));
                }
            };
            return Ok(unit);
        }
        if tree.right.is_none() || tree.left.is_none() {
            panic!("Intermediate node {:?} has a None child", tree.val);
        }

        let right_unit = self.traverse_expression_tree(
            lineno,
            tree.right.as_ref().unwrap().as_ref(),
            unit_defs,
            prefix_defs,
        )?;
        let right_unit = BaseUnit::clone(&right_unit);

        let left_unit = self.traverse_expression_tree(
            lineno,
            tree.left.as_ref().unwrap().as_ref(),
            unit_defs,
            prefix_defs,
        )?;
        let mut left_unit = BaseUnit::clone(&left_unit);

        if let NodeData::Op(op) = &tree.val {
            return Ok(Arc::new(match op {
                Operator::Div => {
                    left_unit /= right_unit;
                    left_unit
                }
                Operator::Mul => {
                    left_unit *= right_unit;
                    left_unit
                }
                Operator::Pow => {
                    // We should never have an expression like `meter ** seconds`, only numerical exponents like `meter ** 2`.
                    // TODO: check this during parsing
                    left_unit.multiplier = left_unit.multiplier.powf(right_unit.multiplier);
                    left_unit
                }
                _ => panic!("Invalid operator: {:?}", op),
            }));
        }
        unreachable!();
    }

    #[inline(always)]
    fn is_comment(line: &str) -> bool {
        line.is_empty() || registry_parser::comment(line).is_ok()
    }
}

#[cfg(test)]
mod test_registry {
    use std::time::Instant;

    use super::*;

    #[test]
    fn test() -> Result<(), SmootError> {
        // let mut registry = Registry::new();

        // let start = Instant::now();
        // registry.load_from_file("default_en.txt")?;
        // println!("File load {:?} ms", start.elapsed().as_millis());

        // let encoded = bitcode::encode(&registry);
        // let mut tmp = NamedTempFile::new().unwrap();
        // let _ = tmp.write(&encoded).unwrap();
        // let _ = tmp.seek(std::io::SeekFrom::Start(0)).unwrap();

        // let start = Instant::now();
        // let mut buf = Vec::new();
        // let _ = tmp.read_to_end(&mut buf).unwrap();
        // let decoded: Registry = bitcode::decode(&buf).unwrap();
        // println!("Bitcode load {:?} ms", start.elapsed().as_millis());

        Registry::clear_cache();

        let start = Instant::now();
        let _ = Registry::default();
        println!("First load {:?} ms", start.elapsed().as_millis());

        let start = Instant::now();
        let _ = Registry::default();
        println!("Second load {:?} ms", start.elapsed().as_millis());

        assert!(false);
        Ok(())
    }
}
