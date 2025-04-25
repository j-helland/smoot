use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash};
use std::io::Write;
use std::path::Path;

use bitcode::{Decode, Encode};
use xxhash_rust::xxh3::{Xxh3Builder, xxh3_64};

use crate::base_unit::{BaseUnit, DIMENSIONLESS_TYPE, DimensionType};
use crate::error::{SmootError, SmootResult};

use super::registry_parser::{
    DimensionDefinition, NodeData, Operator, ParseTree, PrefixDefinition, UnitDefinition,
    registry_parser,
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
    root_units: HashMap<DimensionType, BaseUnit, Xxh3Builder>,
    dimensions: HashMap<DimensionType, String, Xxh3Builder>,
    prefix_definitions: HashMap<String, PrefixDefinition, Xxh3Builder>,
    symbols: HashMap<String, String, Xxh3Builder>,
    checksum: u64,
}
impl Registry {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            units: HashMap::new(),
            root_units: HashMap::default(),
            dimensions: HashMap::default(),
            prefix_definitions: HashMap::default(),
            symbols: HashMap::default(),
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

    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
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

        //==================================================
        // First pass - get abstract definitions
        //==================================================
        // Containers for parsed unit constructs. These are not the "real" units, we can only build those
        // in a second pass since parsed expressions may reference units that are defined later in the file.
        let (mut unit_defs, dim_defs) = self.parse_abstract_definitions(lines)?;

        //==================================================
        // Prefix assignment expressions
        //==================================================
        // Add prefixes and suffixes for regular unit definitions.
        //
        // Prefixes are combinatorial. Every prefix (and prefix alias) must be attached to every unit (and unit alias).
        // Add all prefix combos as regular units so that the unit definition stage instantiates everything seamlessly.
        for name in unit_defs.keys().cloned().collect::<Vec<_>>() {
            let def = unit_defs.get(&name).unwrap().clone();

            // Plural form of base unit e.g. 'seconds'
            Self::insert_or_warn(
                def.lineno,
                &mut unit_defs,
                def.name.clone() + "s",
                def.clone(),
            );

            // Add symbol e.g. 's' for 'second'
            let data: UnitAliasData = def.into();
            if let Some(symbol) = data.symbol {
                let new_alias = data.to_alias(symbol.to_string(), data.name.clone());
                Self::insert_or_warn(data.lineno, &mut unit_defs, symbol.to_string(), new_alias);
            };

            self.create_unit_defs_from_prefixes(&mut unit_defs, name, data);
        }

        // Add prefixes and suffixes for dimension definitions
        // Dimension definitions like `meter = [length] = m = metre`
        for data in dim_defs.iter().cloned().map(UnitAliasData::from) {
            self.create_unit_defs_from_prefixes(&mut unit_defs, data.name.clone(), data);
        }

        // Avoid excessive re-hashing of tables from resizing.
        self.dimensions.reserve(dim_defs.len());
        self.units.reserve(unit_defs.len() + dim_defs.len());

        //==================================================
        // Define dimensions
        //==================================================
        // These are the base cases for recursive unit definitions like `smoot = 1.7018 * meter`.
        self.define_root_units(&dim_defs);

        //==================================================
        // Evaluate unit definitions
        //==================================================
        // Wire together the remaining abstract definitions into concrete units.
        // For example, `smoot = 1.7018 * meter` will be recursively linked to `meter = [length]`,
        // which was defined as a root unit previously.
        self.define_units(&unit_defs)
    }

    /// Insert a value into a map, returning an error on conflict without clobbering the value.
    /// This differs from the builtin hashmap insertion, which clobbers existing values.
    fn try_insert<K, V, S>(
        lineno: usize,
        map: &mut HashMap<K, V, S>,
        key: K,
        val: V,
    ) -> Result<&mut V, SmootError>
    where
        K: Eq + Hash + Debug,
        V: Debug,
        S: BuildHasher,
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
    fn insert_or_warn<K, V, S>(_lineno: usize, map: &mut HashMap<K, V, S>, key: K, val: V)
    where
        K: Eq + Hash + Debug,
        V: Debug,
        S: BuildHasher,
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
            // Symbol
            if let Some(symbol) = def.symbol {
                Self::insert_or_warn(
                    def.lineno,
                    &mut self.symbols,
                    def.name.clone(),
                    symbol.to_string(),
                );
            }
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

    #[inline(always)]
    fn add_prefix(prefix: String, name: &str) -> String {
        prefix + name
    }

    /// Parse all abstract unit and dimension definitions e.g. `meter = [length]`.
    ///
    /// These definitions reference each other (e.g. `smoot = 1.7018 * meter`) and can
    /// be declared in any order. The abstract definitions returned from this function
    /// are not yet wired together, they are essentially just parse trees with symbols
    /// that still need to be linked.
    fn parse_abstract_definitions<'input, L>(
        &mut self,
        lines: L,
    ) -> SmootResult<(
        HashMap<String, UnitDefinition<'input>>,
        Vec<DimensionDefinition<'input>>,
        // HashMap<String, ParseTree>,
    )>
    where
        L: Iterator<Item = (usize, &'input str)>,
    {
        let mut unit_defs: HashMap<String, UnitDefinition> = HashMap::new();
        // Dimensions like `[length]` define the category that a unit belongs to.
        let mut dim_defs: Vec<DimensionDefinition> = Vec::new();
        // // Derived dimensions are defined by expressions of base dimensions e.g. `[length] / [time]`.
        // let mut derived_dim_defs: HashMap<String, ParseTree> = HashMap::new();

        for (lineno, line) in lines {
            if let Ok(dim_def) = registry_parser::dimension_definition(line, lineno) {
                dim_defs.push(dim_def);
            } else if let Ok(unit_def) = registry_parser::unit_with_aliases(line, lineno) {
                Self::try_insert(lineno, &mut unit_defs, unit_def.name.clone(), unit_def)?;
            } else if let Ok(_expr) = registry_parser::derived_dimension(line) {
                // TODO(jwh): Left here for posterity. We don't actually use derived dimensions in Smoot but might in the future.
                //     if let NodeData::Op(Operator::Assign) = &expr.val {
                //         if let NodeData::Dimension(Some(dimension)) = &expr.left.as_ref().unwrap().val {
                //             if let Some(overwrite) = derived_dim_defs.insert(dimension.clone(), expr) {
                //                 return Err(SmootError::DimensionError(format!(
                //                     "line:{} Line '{}' overwrote previous derived dimension {:?}",
                //                     lineno, line, overwrite
                //                 )));
                //             }
                //         } else {
                //             return Err(SmootError::DimensionError(format!(
                //                 "line:{} Derived dimension expression {} does not assign to a dimension",
                //                 lineno, line
                //             )));
                //         }
                //     }
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
        Ok((
            unit_defs, dim_defs,
            // derived_dim_defs,
        ))
    }

    /// Take a unit definition and add all prefixes and suffixes as new unit definitions.
    ///
    /// If we have an alias like `kilo- = 1000`, then the newly generated prefixed unit
    /// definitions contain ParseTrees for expressions like `kilometer = 1000 * meter`.
    ///
    /// Examples
    /// --------
    /// If we have a suffix `kilo-` and a unit `meter`, then we will generate
    /// `kilometer` and `kilometers`. Note that `kilometers` will be an alias of `kilometer`.
    fn create_unit_defs_from_prefixes<'a>(
        &self,
        unit_defs: &mut HashMap<String, UnitDefinition<'a>>,
        name: String,
        data: UnitAliasData<'a>,
    ) {
        // For now, hard-code plural suffix 's'.
        // TODO(jwh): This should probably be defined somewhere else, although it's unlikely that
        //            we'll have any more suffixes.
        for &suffix in ["", "s"].iter() {
            // Add aliases e.g. 'sec' and 'secs' for 'second'
            for &alias in data.aliases.iter() {
                let new_alias = data.to_alias(alias.to_string(), data.name.clone());
                Self::insert_or_warn(
                    data.lineno,
                    unit_defs,
                    alias.to_string() + suffix,
                    new_alias,
                );
            }

            for prefix_def in self.prefix_definitions.values() {
                let f_make_new_def = |new_name: String| UnitDefinition {
                    name: new_name.clone(),
                    symbol: None,
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
                    modifiers: data.modifiers.clone(),
                    // Clear the aliases to avoid redefining them
                    aliases: vec![],
                    lineno: data.lineno,
                };

                let prefix_name = Self::add_prefix(prefix_def.name.clone(), &name);

                // Add prefixed symbol e.g. 'millis' and 'ms' for 'second'.
                // No plural forms for symbols e.g. 'ms' [length] would conflict with 'ms' [time].
                if let Some(symbol) = data.symbol {
                    for prefix_alias in prefix_def.aliases.iter() {
                        let alias_name = Self::add_prefix(prefix_alias.clone(), symbol);
                        let new_alias = data.to_alias(alias_name.clone(), prefix_name.clone());
                        Self::insert_or_warn(data.lineno, unit_defs, alias_name, new_alias);
                    }

                    let alias_name = Self::add_prefix(prefix_def.name.clone(), symbol);
                    let new_alias = data.to_alias(alias_name.clone(), prefix_name.clone());
                    Self::insert_or_warn(data.lineno, unit_defs, alias_name, new_alias);
                };

                // Add all combinations of prefix aliases and unit aliases e.g. 'msec' and 'msecs'.
                for &alias in data.aliases.iter() {
                    for prefix_alias in prefix_def.aliases.iter() {
                        let alias_name = Self::add_prefix(prefix_alias.clone(), alias);
                        let new_alias = data.to_alias(alias_name.clone(), prefix_name.clone());
                        Self::insert_or_warn(
                            data.lineno,
                            unit_defs,
                            alias_name + suffix,
                            new_alias,
                        );
                    }

                    let alias_name = Self::add_prefix(prefix_def.name.clone(), alias);
                    let new_alias = data.to_alias(alias_name.clone(), prefix_name.clone());
                    Self::insert_or_warn(data.lineno, unit_defs, alias_name + suffix, new_alias);
                }

                // Add all prefix aliases to the unit name e.g. 'msecond'.
                for prefix_alias in prefix_def.aliases.iter() {
                    let alias_name = Self::add_prefix(prefix_alias.clone(), &data.name);
                    let new_alias = data.to_alias(alias_name.clone(), prefix_name.clone());
                    Self::insert_or_warn(data.lineno, unit_defs, alias_name + suffix, new_alias);
                }

                // Add prefixed unit e.g. 'millisecond'.
                let new_def = f_make_new_def(prefix_name.clone());
                Self::insert_or_warn(data.lineno, unit_defs, prefix_name + suffix, new_def);
            }
        }
    }

    /// Create root units from dimension definitions like `meter = [length]`.
    fn define_root_units(&mut self, dim_defs: &Vec<DimensionDefinition>) {
        let mut next_dimension = 1;
        let mut dimensions = HashMap::new();

        for def in dim_defs.into_iter() {
            let dim = if def.is_dimensionless() {
                DIMENSIONLESS_TYPE
            } else {
                let dim = next_dimension;
                next_dimension <<= 1;
                dim
            };

            let unit = BaseUnit::new(def.name.into(), 1.0, dim);

            self.insert_root_unit(unit.clone());
            // Symbol
            if let Some(symbol) = def.symbol {
                self.insert_def(symbol.to_string(), &vec![], unit.clone());
                Self::insert_or_warn(
                    def.lineno,
                    &mut self.symbols,
                    def.name.to_string(),
                    symbol.to_string(),
                );
            }
            // Plural unit
            // No aliases needed since non-plural covers it
            self.insert_def(def.name.to_string() + "s", &vec![], unit.clone());
            // Non-plural
            self.insert_def(def.name.to_string(), &def.aliases, unit);

            dimensions.insert(def.dimension, dim);
            self.dimensions.insert(dim, def.dimension.to_string());
        }
    }
}

/// Common data shared between various unit-like definitions.
struct UnitAliasData<'a> {
    name: String,
    lineno: usize,
    aliases: Vec<&'a str>,
    modifiers: Vec<(&'a str, f64)>,
    symbol: Option<&'a str>,
}
impl<'a> From<UnitDefinition<'a>> for UnitAliasData<'a> {
    fn from(value: UnitDefinition<'a>) -> Self {
        UnitAliasData {
            name: value.name,
            lineno: value.lineno,
            aliases: value.aliases,
            modifiers: value.modifiers,
            symbol: value.symbol,
        }
    }
}
impl<'a> From<DimensionDefinition<'a>> for UnitAliasData<'a> {
    fn from(value: DimensionDefinition<'a>) -> Self {
        UnitAliasData {
            name: value.name.to_string(),
            lineno: value.lineno,
            aliases: value.aliases,
            modifiers: value.modifiers,
            symbol: value.symbol,
        }
    }
}

/// Generate an alias unit definition from self.
trait ToUnitDefAlias<'a> {
    fn to_alias(&self, new_name: String, from: String) -> UnitDefinition<'a>;
}

impl<'input, 'output> ToUnitDefAlias<'input> for UnitAliasData<'output>
where
    'output: 'input,
{
    fn to_alias(&self, new_name: String, from: String) -> UnitDefinition<'input> {
        UnitDefinition {
            name: new_name.clone(),
            symbol: None,
            // Create an expression like `km = kilometer`.
            expression: ParseTree::new(Operator::AssignAlias.into(), new_name.into(), from.into()),
            modifiers: self.modifiers.clone(),
            // Clear the aliases to avoid redefining them
            aliases: vec![],
            lineno: self.lineno,
        }
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
    #[case(
        "hour = 60 = h = hr",
        Some(HashMap::from([
            ("hour".to_string(), BaseUnit::new("hour".to_string(), 60.0, 0)),
            ("hours".to_string(), BaseUnit::new("hour".to_string(), 60.0, 0)),
            // Symbol should not have plural form
            ("h".to_string(), BaseUnit::new("hour".to_string(), 60.0, 0)),
            ("hr".to_string(), BaseUnit::new("hour".to_string(), 60.0, 0)),
            ("hrs".to_string(), BaseUnit::new("hour".to_string(), 60.0, 0)),
        ]))
        ; "Aliases are plural but not symbols"
    )]
    #[case(
        "milli- = 1e-3\nsecond = [time] = s = sec",
        Some(HashMap::from([
            ("second".to_string(), BaseUnit::new("second".to_string(), 1.0, 1)),
            ("seconds".to_string(), BaseUnit::new("second".to_string(), 1.0, 1)),
            ("millisecond".to_string(), BaseUnit::new("millisecond".to_string(), 1e-3, 1)),
            ("milliseconds".to_string(), BaseUnit::new("millisecond".to_string(), 1e-3, 1)),
            ("s".to_string(), BaseUnit::new("second".to_string(), 1.0, 1)),
            ("millis".to_string(), BaseUnit::new("millisecond".to_string(), 1e-3, 1)),
            ("sec".to_string(), BaseUnit::new("second".to_string(), 1.0, 1)),
            ("secs".to_string(), BaseUnit::new("second".to_string(), 1.0, 1)),
            ("millisec".to_string(), BaseUnit::new("millisecond".to_string(), 1e-3, 1)),
            ("millisecs".to_string(), BaseUnit::new("millisecond".to_string(), 1e-3, 1)),
        ]))
        ; "Prefixes and suffixes apply to dimension definitions"
    )]
    #[case(
        "kilo- = 1000\nmeter = 1 = m = metre",
        Some(HashMap::from([
            ("meter".to_string(), BaseUnit::new("meter".to_string(), 1.0, 0)),
            ("meters".to_string(), BaseUnit::new("meter".to_string(), 1.0, 0)),
            ("kilometer".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, 0)),
            ("kilometers".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, 0)),
            ("m".to_string(), BaseUnit::new("meter".to_string(), 1.0, 0)),
            ("kilom".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, 0)),
            ("metre".to_string(), BaseUnit::new("meter".to_string(), 1.0, 0)),
            ("metres".to_string(), BaseUnit::new("meter".to_string(), 1.0, 0)),
            ("kilometre".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, 0)),
            ("kilometres".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, 0)),
        ]))
        ; "Prefixes and suffixes apply to unit definitions"
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

    /// Symbols are correctly stored for all unit definitions.
    #[test]
    fn test_symbols() -> SmootResult<()> {
        let registry = Registry::new_from_str(
            "meter = [length] = m = metre\n\
            second = [time] = s\n\
            speed_of_light = 299792458 m/s = c = c_0\n\
            angstrom = 1e-10 * meter = Å = ångström",
        )?;
        assert_eq!(registry.symbols, {
            let mut h = HashMap::default();
            h.insert("meter".to_string(), "m".to_string());
            h.insert("second".to_string(), "s".to_string());
            h.insert("speed_of_light".to_string(), "c".to_string());
            h.insert("angstrom".to_string(), "Å".to_string());
            h
        });
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
