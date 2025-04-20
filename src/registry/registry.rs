use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::io::Write;
use std::path::Path;

use bitcode::{Decode, Encode};
use xxhash_rust::xxh3::xxh3_64;

use crate::base_unit::{BaseUnit, DimensionType, DIMENSIONLESS_TYPE};
use crate::error::{SmootError, SmootResult};

use super::registry_parser::{
    registry_parser, DimensionDefinition, NodeData, Operator, ParseTree, PrefixDefinition,
    UnitDefinition,
};

/// Registry holds all BaseUnit definitions.
///
/// The registry is intended to be cached to disk in a format that is faster to load than
/// re-parsing the unit definitions file.
#[derive(Encode, Decode)]
pub struct Registry {
    pub(crate) units: HashMap<String, BaseUnit>,

    /// A root unit is the canonical unit for a particular dimension (e.g. [length] -> meter).
    // Don't bother with sharing memory with units because there's relatively few dimensions,
    // making this cheap.
    root_units: HashMap<DimensionType, BaseUnit>,
    dimensions: HashMap<DimensionType, String>,
    prefix_definitions: HashMap<String, PrefixDefinition>,
    checksum: u64,
}
impl Registry {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            units: HashMap::new(),
            root_units: HashMap::new(),
            dimensions: HashMap::new(),
            prefix_definitions: HashMap::new(),
            checksum: 0,
        }
    }

    /// Load a registry from a string containing definitions.
    pub fn new_from_str(data: &str) -> SmootResult<Self> {
        let mut new = Self::new();
        new.parse_definitions(data)?;
        Ok(new)
    }

    pub fn new_from_file(path: &Path) -> SmootResult<Self> {
        let mut new = Self::new();
        new.load_from_file(path)?;
        Ok(new)
    }

    pub fn new_from_cache_or_file(cache_path: &Path, file_path: &Path) -> SmootResult<Self> {
        Self::load_cache_or(cache_path, file_path)
    }

    /// Extend this registry with additional definitions.
    pub fn extend(&mut self, data: &str) -> SmootResult<()> {
        self.parse_definitions(data)
    }

    /// Forcibly delete the local cache file if it exists.
    pub fn clear_cache(path: &Path) {
        std::fs::remove_file(path).unwrap_or(())
    }

    pub fn get_unit(&self, key: &str) -> Option<&BaseUnit> {
        self.units.get(key)
    }

    pub fn get_root_unit(&self, dimension: &DimensionType) -> &BaseUnit {
        self.root_units.get(dimension).expect("Missing root unit")
    }

    pub fn get_dimension(&self, dimension: &DimensionType) -> &String {
        self.dimensions.get(dimension).expect("Missing dimension")
    }

    pub fn len(&self) -> usize {
        self.units.len()
    }

    pub fn all_keys(&self) -> Vec<String> {
        self.units.keys().cloned().collect()
    }

    /// Load a units file and cache it.
    fn load_and_cache(path: &Path, cache_path: &Path) -> SmootResult<Self> {
        let mut new = Self::new();
        new.load_from_file(path)?;

        std::fs::File::create(cache_path)
            .and_then(|mut f| f.write(&bitcode::encode(&new)))
            .map_err(|e| {
                SmootError::CacheError(format!(
                    "Failed to write cache file {}: {}",
                    cache_path.display(),
                    e
                ))
            })
            .map(|_| new)
    }

    /// Load registry from cache data.
    /// If the cache data is invalid or does not match the specified file path, fall back to loading from the file instead.
    fn load_cache_or(cache_path: &Path, path: &Path) -> SmootResult<Self> {
        let registry = std::fs::read(cache_path)
            .map_err(|e| {
                SmootError::CacheError(format!(
                    "Failed to read cache file {}: {}",
                    cache_path.display(),
                    e
                ))
            })
            .and_then(|data| {
                bitcode::decode::<Registry>(&data)
                    .map_err(|e| SmootError::CacheError(format!("Failed to decode cache: {}", e)))
            });
        if registry.is_err() {
            // No cache, create a new one.
            return Self::load_and_cache(path, cache_path);
        }

        let checksum = std::fs::read(path)
            .map_err(|e| {
                SmootError::FileError(format!(
                    "Failed to load unit definitions file {}: {}",
                    path.display(),
                    e
                ))
            })
            .map(|data| Self::compute_checksum(&data))?;

        // If the checksum doesn't match, there was a change to the units. Load from scratch and create a new cache.
        if checksum != registry.as_ref().unwrap().checksum {
            Self::load_and_cache(path, cache_path)
        } else {
            registry
        }
    }

    /// Parse a unit definitions file, populating this registry.
    fn load_from_file(&mut self, path: &Path) -> SmootResult<()> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| SmootError::FileError(format!("{}: {}", path.display(), e)))?;
        self.parse_definitions(&data)?;
        self.checksum = Self::compute_checksum(data.as_bytes());
        Ok(())
    }

    fn compute_checksum(data: &[u8]) -> u64 {
        // We just need it to be fast, not cryptographically secure.
        xxh3_64(data)
    }

    /// Internal definitions file parsing logic.
    fn parse_definitions(&mut self, data: &str) -> SmootResult<()> {
        // Iterate unit definition lines, ignoring comments and blank lines.
        // Unit definition expressions are strictly contained in a single line, although the expression
        // may reference other lines implicitly.
        let lines = data
            .lines()
            .map(|line| line.trim())
            .enumerate()
            .map(|(lineno, line)| (lineno + 1, line))
            .filter(|(_, line)| !line.is_empty() && !Self::is_comment(line));

        // Containers for parsed unit constructs. These are not the "real" units, we can only build those
        // in a second pass since parsed expressions may reference units that are defined later in the file.
        // Dimensions like `[length]` define the category that a unit belongs to.
        let mut dim_defs: Vec<DimensionDefinition> = Vec::new();
        let mut unit_defs: HashMap<String, UnitDefinition> = HashMap::new();
        // Derived dimensions are defined by expressions of base dimensions e.g. `[length] / [time]`.
        let mut derived_dim_defs: HashMap<String, ParseTree> = HashMap::new();

        //==================================================
        // First pass - get raw definitions
        //==================================================
        // Parse all abstract "definition" objects, which will be wired together into "real" objects during
        // the second pass.
        for (lineno, line) in lines {
            if let Ok(dim_def) = registry_parser::dimension_definition(line, lineno) {
                dim_defs.push(dim_def);
            } else if let Ok(unit_def) = registry_parser::unit_with_aliases(line, lineno) {
                // Ignore any aliases named `_`
                for &alias in unit_def.aliases.iter().filter(|&&a| a.ne("_")) {
                    Self::try_insert(lineno, &mut unit_defs, alias.into(), unit_def.clone())?;
                }

                // Plural form e.g. "meters"
                // Don't add plural suffix to aliases because this could create confusing conflicts like "ms" for "meters" / "milliseconds".
                let mut plural_unit_def = unit_def.clone();
                // No need for aliases since the non-plural definition already covers it.
                plural_unit_def.aliases.clear();
                Self::try_insert(
                    lineno,
                    &mut unit_defs,
                    plural_unit_def.name.clone() + "s",
                    plural_unit_def,
                )?;

                // Non-plural form e.g. "meter"
                Self::try_insert(lineno, &mut unit_defs, unit_def.name.clone(), unit_def)?;
            } else if let Ok(expr) = registry_parser::derived_dimension(line) {
                if let NodeData::Op(Operator::Assign) = &expr.val {
                    if let NodeData::Dimension(Some(dimension)) = &expr.left.as_ref().unwrap().val {
                        if let Some(overwrite) = derived_dim_defs.insert(dimension.clone(), expr) {
                            return Err(SmootError::DimensionError(format!(
                                "line:{} Line '{}' overwrote previous derived dimension {:?}",
                                lineno, line, overwrite
                            )));
                        }
                    } else {
                        return Err(SmootError::DimensionError(format!(
                            "line:{} Derived dimension expression {} does not assign to a dimension",
                            lineno,
                            line
                        )));
                    }
                }
            } else if let Ok(prefix_def) = registry_parser::prefix_definition(line, lineno) {
                Self::try_insert(
                    lineno,
                    &mut self.prefix_definitions,
                    prefix_def.name.clone(),
                    prefix_def,
                )?;
            } else {
                return Err(SmootError::ParseTreeError(format!(
                    "line:{} Unhandled line '{}'",
                    lineno, line
                )));
            }
        }

        //==================================================
        // Prefix assignment expressions
        //==================================================
        let f_add_prefix = |mut prefix: String, name: &str| {
            prefix.push_str(name);
            prefix
        };

        // Add prefixes for regular unit definitions
        // Prefixes are combinatorial. Every prefix (and prefix alias) must be attached to every unit (and unit alias).
        // Add all prefix combos as regular units so that the unit definition stage instantiates everything seamlessly.
        //
        // This works by grafting existing unit definition parse trees with an additional multiplicative factor for each
        // prefix (e.g. kilo corresponds to 1000, so we graft on a `1000 * ` operation to the expression's parse tree).
        for name in unit_defs.keys().cloned().collect::<Vec<_>>() {
            let def = unit_defs.get(&name).unwrap().clone();

            for prefix_def in self.prefix_definitions.values() {
                let f_make_new_def = |new_name: String| UnitDefinition {
                    name: new_name.clone(),
                    // Create an expression like `kilometer = 1000 * meter`, where `M` is the prefix multiplier.
                    expression: ParseTree::new(
                        Operator::Assign.into(),
                        new_name.into(),
                        ParseTree::new(
                            Operator::Mul.into(),
                            prefix_def.multiplier.into(),
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
                let prefix_name = f_add_prefix(prefix_def.name.clone(), &name);

                // Ignore any aliases named `_`
                for prefix_alias in prefix_def.aliases.iter().filter(|&a| a.ne("_")) {
                    let alias_name = f_add_prefix(prefix_alias.clone(), &name);

                    let new_alias = f_make_new_alias(alias_name.clone(), prefix_name.clone());
                    Self::insert_or_warn(def.lineno, &mut unit_defs, alias_name, new_alias);
                }

                let new_def = f_make_new_def(prefix_name.clone());
                Self::insert_or_warn(def.lineno, &mut unit_defs, prefix_name, new_def);
            }
        }

        // Add prefixes for dimension definitions
        // Dimension definitions like `meter = [length] = m = metre` need the same treatment.
        for def in dim_defs.iter() {
            let name = def.name;

            for prefix_def in self.prefix_definitions.values() {
                let f_make_new_def = |new_name: String| UnitDefinition {
                    name: new_name.clone(),
                    // Create an expression like `kilometer = 1000 * meter`, where `M` is the prefix multiplier.
                    expression: ParseTree::new(
                        Operator::Assign.into(),
                        new_name.into(),
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

                let prefix_name = f_add_prefix(prefix_def.name.clone(), def.name);

                // Add all combinations of prefix aliases and unit aliases.
                // Ignore any aliases named `_`
                for alias in def.aliases.iter().filter(|&&a| a.ne("_")) {
                    let alias_name = f_add_prefix(prefix_def.name.clone(), alias);

                    // Ignore any aliases named `_`
                    for prefix_alias in prefix_def.aliases.iter().filter(|&a| a.ne("_")) {
                        let alias_name = f_add_prefix(prefix_alias.clone(), alias);

                        // Add prefix aliases
                        let new_alias = f_make_new_alias(alias_name.clone(), prefix_name.clone());
                        Self::insert_or_warn(def.lineno, &mut unit_defs, alias_name, new_alias);
                    }

                    // Add prefixed unit aliases
                    let new_alias = f_make_new_alias(alias_name.clone(), prefix_name.clone());
                    Self::insert_or_warn(def.lineno, &mut unit_defs, alias_name, new_alias);
                }

                // Add all prefix aliases to the unit name
                // Ignore any aliases named `_`
                for prefix_alias in prefix_def.aliases.iter().filter(|&a| a.ne("_")) {
                    let alias_name = f_add_prefix(prefix_alias.clone(), def.name);

                    let new_alias = f_make_new_alias(alias_name.clone(), prefix_name.clone());
                    Self::insert_or_warn(def.lineno, &mut unit_defs, alias_name, new_alias);
                }

                // Add prefixed unit
                let new_def = f_make_new_def(prefix_name.clone());
                // Plural
                let mut plural_new_def = new_def.clone();
                // No aliases needed in plural since non-plural covers it
                plural_new_def.aliases.clear();
                Self::insert_or_warn(
                    def.lineno,
                    &mut unit_defs,
                    f_add_prefix(prefix_def.name.clone(), name) + "s",
                    plural_new_def,
                );
                // Non-plural
                Self::insert_or_warn(def.lineno, &mut unit_defs, prefix_name, new_def);
            }
        }

        //==================================================
        // Define dimensions
        //==================================================
        let mut next_dimension = 1;
        let mut dimensions = HashMap::new();

        dim_defs.into_iter().for_each(|def| {
            let dim = if def.is_dimensionless() {
                DIMENSIONLESS_TYPE
            } else {
                let dim = next_dimension;
                next_dimension <<= 1;
                dim
            };

            let unit = BaseUnit::new(def.name.into(), 1.0, dim);

            self.insert_root_unit(unit.clone());
            // Plural unit
            // No aliases needed since non-plural covers it
            self.insert_def(def.name.to_string() + "s", &vec![], unit.clone());
            // Non-plural
            self.insert_def(def.name.to_string(), &def.aliases, unit);

            dimensions.insert(def.dimension, dim);
            self.dimensions.insert(dim, def.dimension.to_string());
        });

        //==================================================
        // Evaluate unit definitions
        //==================================================
        self.define_units(&unit_defs)
    }

    /// Insert a value into a map, returning an error on conflict without clobbering the value.
    /// This differs from the builtin hashmap insertion, which clobbers existing values.
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
            Entry::Occupied(entry) => Err(SmootError::ExpressionError(format!(
                "line:{} Attempted to overwrite value {:?}\nwith {:?}:{:?}",
                lineno,
                entry.get(),
                entry.key(),
                val
            ))),
        }
    }

    /// Insert a value into a map, silently ignoring conflicts (no clobbering of existing values).
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

    /// Insert a unit definition.
    fn insert_def(&mut self, name: String, aliases: &Vec<&str>, unit: BaseUnit) -> &BaseUnit {
        // Ignore any aliases named `_`
        for &alias in aliases.iter().filter(|&&a| a.ne("_")) {
            let _ = self.units.entry(alias.into()).or_insert(unit.clone());
        }
        self.units.entry(name).or_insert(unit)
    }

    /// Insert only if the entry doesn't exist
    fn insert_root_unit(&mut self, unit: BaseUnit) {
        self.root_units.entry(unit.unit_type).or_insert(unit);
    }

    /// Take all parsed unit definitions and create real units from them.
    fn define_units(
        &mut self,
        unit_defs: &HashMap<String, UnitDefinition>,
    ) -> Result<(), SmootError> {
        for (name, def) in unit_defs.iter() {
            if self.units.contains_key(name) {
                continue;
            }

            let rtree = def.expression.right.as_ref().unwrap();
            let unit = self.eval_expression_tree(def.lineno, rtree, unit_defs)?;

            let unit = match def.expression.val {
                // Regular unit definitions like `byte = 8 * bit` should assign the lhs name `byte`.
                NodeData::Op(Operator::Assign) => {
                    let mut unit = BaseUnit::clone(&unit);
                    // Make sure the official name is correct
                    unit.name = def.name.clone();
                    unit
                }
                // Generated alias expressions like `@alias km = kilometer` should use the rhs name `kilometer` for consistency.
                NodeData::Op(Operator::AssignAlias) => unit,
                _ => {
                    return Err(SmootError::ExpressionError(format!(
                        "line:{} Invalid unit expression {:#?}",
                        def.lineno, def.expression,
                    )));
                }
            };

            let _ = self.insert_def(name.clone(), &def.aliases, unit);
        }
        Ok(())
    }

    /// Get a previously defined unit, defining it if no previous definition exists.
    fn get_or_create_unit(
        &mut self,
        lineno: usize,
        symbol: &str,
        unit_defs: &HashMap<String, UnitDefinition>,
    ) -> Result<&BaseUnit, SmootError> {
        if self.units.contains_key(symbol) {
            return Ok(self.units.get(symbol).unwrap());
        }

        let def = unit_defs.get(symbol).ok_or_else(|| {
            SmootError::ExpressionError(format!("line:{} Unknown unit {}", lineno, symbol))
        })?;
        let rtree = def.expression.right.as_ref().ok_or_else(|| {
            SmootError::ExpressionError(format!(
                "line:{} Unit expression missing right expression tree",
                lineno
            ))
        })?;

        // Make sure the official name is correct
        let unit = self.eval_expression_tree(def.lineno, rtree, unit_defs)?;
        let mut unit = BaseUnit::clone(&unit);
        unit.name = def.name.clone();

        // Insert name and aliases
        Ok(self.insert_def(def.name.clone(), &def.aliases, unit))
    }

    /// Evaluate a parsed unit expression into a unit definition.
    fn eval_expression_tree(
        &mut self,
        lineno: usize,
        tree: &ParseTree,
        unit_defs: &HashMap<String, UnitDefinition>,
    ) -> SmootResult<BaseUnit> {
        if tree.left.is_none() && tree.right.is_none() {
            // Leaf
            let unit = match &tree.val {
                NodeData::Integer(int) => BaseUnit::new_constant(f64::from(*int)),
                NodeData::Decimal(float) => BaseUnit::new_constant(*float),
                NodeData::Symbol(symbol) => {
                    self.get_or_create_unit(lineno, symbol, unit_defs)?.clone()
                }
                NodeData::Op(op) => {
                    return Err(SmootError::ParseTreeError(format!(
                        "line:{} Invalid operator {:?}",
                        lineno, op
                    )));
                }
                NodeData::Dimension(dim) => {
                    return Err(SmootError::ParseTreeError(format!(
                        "line:{} Unexpected dimension {:?}",
                        lineno,
                        dim.to_owned()
                    )));
                }
            };
            return Ok(unit);
        }
        if tree.right.is_none() || tree.left.is_none() {
            return Err(SmootError::ParseTreeError(format!(
                "line:{} Intermediate node {:?} has a None child",
                lineno, tree.val
            )));
        }

        let right_unit =
            self.eval_expression_tree(lineno, tree.right.as_ref().unwrap().as_ref(), unit_defs)?;
        let right_unit = BaseUnit::clone(&right_unit);

        let left_unit =
            self.eval_expression_tree(lineno, tree.left.as_ref().unwrap().as_ref(), unit_defs)?;
        let mut left_unit = BaseUnit::clone(&left_unit);

        if let NodeData::Op(op) = &tree.val {
            return match op {
                Operator::Div => {
                    left_unit /= right_unit;
                    Ok(left_unit)
                }
                Operator::Mul => {
                    left_unit *= right_unit;
                    Ok(left_unit)
                }
                Operator::Pow => {
                    // We should never have an expression like `meter ** seconds`, only numerical exponents like `meter ** 2`.
                    if !right_unit.name.is_empty() {
                        return Err(SmootError::ParseTreeError(format!(
                            "line:{} Invalid operator {} ** {}",
                            lineno, left_unit.name, right_unit.name
                        )));
                    }
                    left_unit.multiplier = left_unit.multiplier.powf(right_unit.multiplier);
                    left_unit.mul_dimensionality(right_unit.multiplier);
                    Ok(left_unit)
                }
                _ => Err(SmootError::ParseTreeError(format!(
                    "line:{} Invalid operator {:?}",
                    lineno, op
                ))),
            };
        }
        unreachable!();
    }

    #[inline(always)]
    fn is_comment(line: &str) -> bool {
        line.is_empty() || registry_parser::comment(line).is_ok()
    }
}

//==================================================
// Unit tests
//==================================================
#[cfg(test)]
mod test_registry {
    use test_case::case;

    use crate::test_utils::{DEFAULT_UNITS_FILE, TEST_CACHE_PATH};

    use super::*;

    #[test]
    fn test_registry_loads_default() -> SmootResult<()> {
        let cache_path = Path::new(TEST_CACHE_PATH);
        Registry::clear_cache(cache_path);
        let _ = Registry::new_from_cache_or_file(cache_path, Path::new(DEFAULT_UNITS_FILE))?;
        // Second load from cached file
        let _ = Registry::new_from_cache_or_file(cache_path, Path::new(DEFAULT_UNITS_FILE))?;
        Ok(())
    }

    #[case(
        "percent = 0.01 = %",
        Some(HashMap::from([
            ("percent".to_string(), BaseUnit::new("percent".to_string(), 0.01, 0)),
            ("percents".to_string(), BaseUnit::new("percent".to_string(), 0.01, 0)),
            ("%".to_string(), BaseUnit::new("percent".to_string(), 0.01, 0)),
        ]))
        ; "Dimensionless unit with alias parses"
    )]
    #[case(
        "# ignored comment\npercent = 0.01 = %  # ignored comment",
        Some(HashMap::from([
            ("percent".to_string(), BaseUnit::new("percent".to_string(), 0.01, 0)),
            ("percents".to_string(), BaseUnit::new("percent".to_string(), 0.01, 0)),
            ("%".to_string(), BaseUnit::new("percent".to_string(), 0.01, 0)),
        ]))
        ; "Comments are ignored"
    )]
    #[case(
        "kilo- = 1e3\nmeter = [length]",
        Some(HashMap::from([
            ("meter".to_string(), BaseUnit::new("meter".to_string(), 1.0, 1)),
            ("meters".to_string(), BaseUnit::new("meter".to_string(), 1.0, 1)),
            ("kilometer".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, 1)),
            ("kilometers".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, 1)),
        ]))
        ; "Prefixes are applied"
    )]
    #[case(
        "unit2 = 2 * unit1\nunit1 = 1.0",
        Some(HashMap::from([
            ("unit1".to_string(), BaseUnit::new("unit1".to_string(), 1.0, 0)),
            ("unit1s".to_string(), BaseUnit::new("unit1".to_string(), 1.0, 0)),
            ("unit2".to_string(), BaseUnit::new("unit2".to_string(), 2.0, 0)),
            ("unit2s".to_string(), BaseUnit::new("unit2".to_string(), 2.0, 0)),
        ]))
        ; "Derived units can be defined in any order"
    )]
    #[case(
        "radian = []\nsteradian = radian ** 2",
        Some(HashMap::from([
            ("radian".to_string(), BaseUnit::new("radian".to_string(), 1.0, 0)),
            ("radians".to_string(), BaseUnit::new("radian".to_string(), 1.0, 0)),
            ("steradian".to_string(), BaseUnit::new("steradian".to_string(), 1.0, 0)),
            ("steradians".to_string(), BaseUnit::new("steradian".to_string(), 1.0, 0)),
        ]))
        ; "Derived dimensionless units stay unitless"
    )]
    fn test_registry_load_from_str(
        data: &str,
        expected_units: Option<HashMap<String, BaseUnit>>,
    ) -> SmootResult<()> {
        let res = Registry::new_from_str(data);
        if let Some(expected_units) = expected_units {
            assert_eq!(res?.units, expected_units);
        } else {
            assert!(res.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_extend() -> SmootResult<()> {
        let mut registry = Registry::new_from_str("nano- = 1e-9\nmeter = [length]")?;
        registry.extend("kilometer = 1000 * meter")?;

        // Old units are present
        assert!(registry.get_unit("meter").is_some());
        assert!(registry.get_unit("nanometer").is_some());
        // Extension units are present
        assert!(registry.get_unit("kilometer").is_some());
        // Prefixes apply to extension units
        assert!(registry.get_unit("nanokilometer").is_some());
        Ok(())
    }
}
