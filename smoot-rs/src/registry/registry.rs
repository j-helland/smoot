use std::collections::{HashMap, hash_map};
use std::io::Write;
use std::path::Path;

use bitcode::{Decode, Encode};
use linked_hash_map::LinkedHashMap;
use rustc_hash::FxBuildHasher;
use xxhash_rust::xxh3::xxh3_64;

use crate::base_unit::{BaseUnit, DIMENSIONLESS, Dimension};
use crate::converter::Converter;
use crate::error::{SmootError, SmootResult};
use crate::utils::ApproxEq;

use super::registry_parser::{
    DimensionDefinition, NodeData, Operator, ParseTree, PrefixDefinition, UnitDefinition,
    registry_parser,
};

/// Insert a value into a map, returning an error on conflict without clobbering the value.
/// This differs from the builtin hashmap insertion, which clobbers existing values.
macro_rules! try_insert {
    ($entry_type:ident, $lineno:expr, $map:expr, $key:expr, $val:expr) => {
        match $map.entry($key) {
            $entry_type::Entry::Vacant(entry) => Ok(entry.insert($val)),
            $entry_type::Entry::Occupied(entry) => Err(SmootError::ExpressionError(format!(
                "line:{} Attempted to overwrite value {:?}\nwith {:?}:{:?}",
                $lineno,
                entry.get(),
                entry.key(),
                $val
            ))),
        }
    };
}

/// Insert a value into a map, silently ignoring conflicts (no clobbering of existing values).
macro_rules! insert_ignore_conflict {
    ($entry_type:ident, $map:expr, $key:expr, $val:expr) => {
        if let $entry_type::Entry::Vacant(entry) = $map.entry($key) {
            entry.insert($val);
        }
    };
}

/// Registry holds all `BaseUnit` definitions.
///
/// The registry is intended to be cached to disk in a format that is faster to load than
/// re-parsing the unit definitions file.
#[derive(Encode, Decode)]
pub struct Registry {
    pub(crate) units: HashMap<String, BaseUnit, FxBuildHasher>,

    /// A root unit is the canonical unit for a particular dimension (e.g. [length] -> meter).
    // Don't bother with sharing memory with units because there's relatively few dimensions,
    // making this cheap.
    root_units: HashMap<Dimension, BaseUnit, FxBuildHasher>,
    dimensions: HashMap<Dimension, String, FxBuildHasher>,
    prefix_definitions: HashMap<String, PrefixDefinition, FxBuildHasher>,
    symbols: HashMap<String, String, FxBuildHasher>,
    checksum: u64,
}
impl Registry {
    pub(crate) const DELTA_PREFIX: &'static str = "delta_";
    const DELTA_SYMBOL: &'static str = "Δ";
    const OFFSET_IDENTIFIER: &'static str = "offset";

    #[allow(clippy::new_without_default)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            units: HashMap::default(),
            root_units: HashMap::default(),
            dimensions: HashMap::default(),
            prefix_definitions: HashMap::default(),
            symbols: HashMap::default(),
            checksum: 0,
        }
    }

    /// Load a registry from a string containing definitions.
    ///
    /// Errors
    /// ------
    /// `SmootError::ExpressionError`
    ///     If an error is encountered while parsing the given unit definitions.
    pub fn new_from_str(data: &str) -> SmootResult<Self> {
        let mut new = Self::new();
        new.parse_definitions(data)?;
        Ok(new)
    }

    /// Errors
    /// ------
    /// `SmootError::FileError`
    ///     If an error was encountered while loading the unit definitions file. This includes file system errors returned from the OS.
    pub fn new_from_file(path: &Path) -> SmootResult<Self> {
        let mut new = Self::new();
        new.load_from_file(path)?;
        Ok(new)
    }

    /// Errors
    /// ------
    /// `SmootError::CacheError`
    ///     If an error was encountered while loading or saving the registry cache file.
    /// `SmootError::FileError`
    ///     If an error was encountered while loading the unit definitions file during fallback if no cache file exists.
    pub fn new_from_cache_or_file(cache_path: &Path, file_path: &Path) -> SmootResult<Self> {
        Self::load_cache_or(cache_path, file_path)
    }

    /// Extend this registry with additional definitions.
    ///
    /// Errors
    /// ------
    /// `SmootError::ExpressionError`
    ///     If an error is encountered while parsing the given unit definitions.
    pub fn extend(&mut self, data: &str) -> SmootResult<()> {
        self.parse_definitions(data)
    }

    /// Forcibly delete the local cache file if it exists.
    pub fn clear_cache(path: &Path) {
        std::fs::remove_file(path).unwrap_or(());
    }

    #[must_use]
    pub fn get_unit(&self, key: &str) -> Option<&BaseUnit> {
        self.units.get(key)
    }

    /// Errors
    /// ------
    /// `SmootError::NoSuchElement`
    ///     If there is no registered root unit for the dimension. This should not happen unless something has gone very wrong when parsing the unit definitions file.
    pub fn get_root_unit(&self, dimension: Dimension) -> SmootResult<&BaseUnit> {
        self.root_units.get(&dimension.abs()).ok_or_else(|| {
            SmootError::NoSuchElement(format!("Missing root unit for dimension {dimension}"))
        })
    }

    /// Errors
    /// ------
    /// `SmootError::NoSuchElement`
    ///     If there is no registered dimension string for this dimension. This should not happen unless something has gone very wrong when parsing the unit definitions file.
    pub fn get_dimension_string(&self, dimension: Dimension) -> SmootResult<&String> {
        self.dimensions
            .get(&dimension.abs())
            .ok_or_else(|| SmootError::NoSuchElement(format!("{dimension}")))
    }

    #[must_use]
    pub fn get_unit_symbol(&self, name: &String) -> Option<&String> {
        self.symbols.get(name)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.units.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    #[must_use]
    pub fn all_keys(&self) -> Vec<String> {
        self.units.keys().cloned().collect()
    }

    fn are_modifiers_offset(modifiers: &[(&str, f64)]) -> bool {
        modifiers
            .iter()
            .any(|(name, _)| *name == Registry::OFFSET_IDENTIFIER)
    }

    /// Load a units file and cache it.
    fn load_and_cache(path: &Path, cache_path: &Path) -> SmootResult<Self> {
        let mut new = Self::new();
        new.load_from_file(path)?;

        std::fs::File::create(cache_path)
            .and_then(|mut f| f.write(&bitcode::encode(&new)))
            .map_err(|e| {
                SmootError::CacheError(format!(
                    "Failed to write cache file {}: {e}",
                    cache_path.display()
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
                    "Failed to read cache file {}: {e}",
                    cache_path.display()
                ))
            })
            .and_then(|data| {
                bitcode::decode::<Registry>(&data)
                    .map_err(|e| SmootError::CacheError(format!("Failed to decode cache: {e}")))
            });
        if registry.is_err() {
            // No cache, create a new one.
            return Self::load_and_cache(path, cache_path);
        }

        let checksum = std::fs::read(path)
            .map_err(|e| {
                SmootError::FileError(format!(
                    "Failed to load unit definitions file {}: {e}",
                    path.display()
                ))
            })
            .map(|data| Self::compute_checksum(&data))?;

        // If the checksum doesn't match, there was a change to the units. Load from scratch and create a new cache.
        if checksum == registry.as_ref().unwrap().checksum {
            registry
        } else {
            Self::load_and_cache(path, cache_path)
        }
    }

    /// Parse a unit definitions file, populating this registry.
    fn load_from_file(&mut self, path: &Path) -> SmootResult<()> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| SmootError::FileError(format!("{}: {e}", path.display())))?;
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
            .map(str::trim)
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
        let initial_unit_defs = unit_defs.keys().cloned().collect::<Vec<_>>();
        for name in &initial_unit_defs {
            let def = unit_defs.get(name).unwrap().clone();

            // Plural form of base unit e.g. 'seconds'
            try_insert!(
                linked_hash_map,
                def.lineno,
                unit_defs,
                def.name.clone() + "s",
                def.clone()
            )?;

            // Add symbol e.g. 's' for 'second'
            let data: UnitAliasData = def.into();
            if let Some(symbol) = &data.symbol {
                let new_alias = data.to_alias(symbol.to_string(), data.name.clone());
                try_insert!(
                    linked_hash_map,
                    data.lineno,
                    &mut unit_defs,
                    symbol.to_string(),
                    new_alias
                )?;
            }
        }

        // Handle prefixes afterwards to give precedence to regular definitions (and their aliases).
        // For example, we prefer "pascal = Pa" over "Pa = petayear".
        for name in initial_unit_defs {
            let def = unit_defs.get(&name).unwrap().clone();
            if Registry::are_modifiers_offset(&def.modifiers) {
                // Can't prefix offset units like `degC` because prefixes are technically multipliers and offset
                // units are non-multiplicative.
                continue;
            }
            let data: UnitAliasData = def.into();
            self.create_unit_defs_from_prefixes(&mut unit_defs, &name, &data);
        }

        // Add prefixes and suffixes for dimension definitions
        // Dimension definitions like `meter = [length] = m = metre`
        for data in dim_defs.iter().cloned().map(UnitAliasData::from) {
            self.create_unit_defs_from_prefixes(&mut unit_defs, &data.name, &data);
        }

        // Avoid excessive re-hashing of tables from resizing.
        self.dimensions.reserve(dim_defs.len());
        self.units.reserve(unit_defs.len() + dim_defs.len());

        //==================================================
        // Define dimensions
        //==================================================
        // These are the base cases for recursive unit definitions like `smoot = 1.7018 * meter`.
        self.define_root_units(&dim_defs)?;

        //==================================================
        // Evaluate unit definitions
        //==================================================
        // Wire together the remaining abstract definitions into concrete units.
        // For example, `smoot = 1.7018 * meter` will be recursively linked to `meter = [length]`,
        // which was defined as a root unit previously.
        self.define_units(&unit_defs)
    }

    /// Insert a unit definition.
    fn insert_def(&mut self, name: String, aliases: &[String], mut unit: BaseUnit) -> &BaseUnit {
        // Make sure all created base units are fully simplified.
        // Shrink capacity for more efficient storage when the registry is cached to disk.
        unit.simplify();
        unit.dimensionality.shrink_to_fit();

        for alias in aliases {
            let _ = self.units.entry(alias.into()).or_insert(unit.clone());
        }

        // If there is already a unit matching this name, it should be identical.
        debug_assert!(
            self.units
                .get(&name)
                .cloned()
                .map(|mut u| {
                    // Ignore the name in equality -- this could be different because of an alias unit.
                    u.name.clone_from(&unit.name);
                    u
                })
                .unwrap_or(unit.clone())
                == unit,
            "{:?}\n!=\n{unit:?}",
            self.units.get(&name).unwrap(),
        );

        self.units.entry(name).or_insert(unit)
    }

    /// Insert only if the entry doesn't exist
    fn insert_root_unit(&mut self, unit: BaseUnit) {
        debug_assert!(unit.dimensionality.len() == 1, "{unit:#?}");
        self.root_units
            .entry(unit.dimensionality[0].abs())
            .or_insert(unit);
    }

    /// Take all parsed unit definitions and create real units from them.
    fn define_units(
        &mut self,
        unit_defs: &LinkedHashMap<String, UnitDefinition>,
    ) -> Result<(), SmootError> {
        for (name, def) in unit_defs {
            if self.units.contains_key(name) {
                continue;
            }

            let rtree = def.expression.right.as_ref().unwrap();
            let mut unit = self.eval_expression_tree(def.lineno, rtree, unit_defs)?;

            match def.expression.val {
                // Regular unit definitions like `byte = 8 * bit` should assign the lhs name `byte`.
                NodeData::Op(Operator::Assign) => {
                    // Make sure the official name is correct
                    unit.name.clone_from(&def.name);
                }
                // Generated alias expressions like `@alias km = kilometer` should use the rhs name `kilometer` for consistency.
                NodeData::Op(Operator::AssignAlias) => (),
                _ => {
                    return Err(SmootError::ExpressionError(format!(
                        "line:{} Invalid unit expression {:#?}",
                        def.lineno, def.expression,
                    )));
                }
            }

            // Assign modifiers.
            for &(mod_name, mod_val) in &def.modifiers {
                match mod_name {
                    Registry::OFFSET_IDENTIFIER => {
                        unit.offset = mod_val;
                        unit.converter = Converter::Offset;
                    }
                    _ => {
                        return Err(SmootError::ExpressionError(format!(
                            "line:{} Invalid unit modifier '{mod_name}'",
                            def.lineno
                        )));
                    }
                }
            }

            // Delta units for offsets are defined internally. They might inherit Some(offset) from the root BaseUnit (e.g. "kelvin; offset = 0"),
            // which needs to be forcibly unset here to make conversions and operations work correctly.
            if name.starts_with(Registry::DELTA_PREFIX) {
                unit.offset = f64::NAN;
                unit.converter = Converter::Multiplicative;
            }

            let _ = self.insert_def(name.clone(), &def.aliases, unit);
            // Symbol
            if let Some(symbol) = &def.symbol {
                insert_ignore_conflict!(
                    hash_map,
                    self.symbols,
                    def.name.clone(),
                    symbol.to_string()
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
        unit_defs: &LinkedHashMap<String, UnitDefinition>,
    ) -> Result<BaseUnit, SmootError> {
        if self.units.contains_key(symbol) {
            return Ok(self.units.get(symbol).unwrap().clone());
        }

        let def = unit_defs.get(symbol).ok_or_else(|| {
            SmootError::ExpressionError(format!("line:{lineno} Unknown unit {symbol}"))
        })?;
        let rtree = def.expression.right.as_ref().ok_or_else(|| {
            SmootError::ExpressionError(format!(
                "line:{lineno} Unit expression missing right expression tree"
            ))
        })?;

        // Make sure the official name is correct
        let mut unit = self.eval_expression_tree(def.lineno, rtree, unit_defs)?;
        unit.name.clone_from(&def.name);

        // Insert name and aliases
        Ok(self
            .insert_def(def.name.clone(), &def.aliases, unit)
            .clone())
    }

    /// Evaluate a parsed unit expression into a unit definition.
    fn eval_expression_tree(
        &mut self,
        lineno: usize,
        tree: &ParseTree,
        unit_defs: &LinkedHashMap<String, UnitDefinition>,
    ) -> SmootResult<BaseUnit> {
        if tree.left.is_none() && tree.right.is_none() {
            // Leaf
            let unit = match &tree.val {
                NodeData::Integer(int) => BaseUnit::new_constant(f64::from(*int)),
                NodeData::Decimal(float) => BaseUnit::new_constant(*float),
                NodeData::Symbol(symbol) => self.get_or_create_unit(lineno, symbol, unit_defs)?,
                NodeData::Op(op) => {
                    return Err(SmootError::ParseTreeError(format!(
                        "line:{lineno} Invalid operator {op:?}"
                    )));
                }
                NodeData::Dimension(dim) => {
                    return Err(SmootError::ParseTreeError(format!(
                        "line:{lineno} Unexpected dimension {:?}",
                        dim.to_owned()
                    )));
                }
            };
            return Ok(unit);
        }
        if tree.left.is_none() {
            return Err(SmootError::ParseTreeError(format!(
                "line:{lineno} Intermediate node {:?} has None left child",
                tree.val
            )));
        }

        // There must always be a left unit.
        let mut left_unit =
            self.eval_expression_tree(lineno, tree.left.as_ref().unwrap().as_ref(), unit_defs)?;

        // Right unit might be None for unary operators e.g. Sqrt.
        let right_unit = tree
            .right
            .as_ref()
            .map(|right| self.eval_expression_tree(lineno, right, unit_defs));

        // Propagate metadata
        if let Some(Ok(right)) = &right_unit {
            left_unit.converter = right.converter;
            left_unit.offset = right.offset;
        }

        if let NodeData::Op(op) = &tree.val {
            return match op {
                Operator::Div => {
                    if let Some(Ok(right_unit)) = right_unit {
                        left_unit /= right_unit;
                        Ok(left_unit)
                    } else {
                        Err(SmootError::ParseTreeError(format!(
                            "line:{} {} / None expected a right operand",
                            lineno, left_unit.name,
                        )))
                    }
                }
                Operator::Mul => {
                    if let Some(Ok(right_unit)) = right_unit {
                        left_unit *= right_unit;
                        Ok(left_unit)
                    } else {
                        Err(SmootError::ParseTreeError(format!(
                            "line:{} {} * None expected a right operand",
                            lineno, left_unit.name,
                        )))
                    }
                }
                Operator::Sqrt => {
                    left_unit.ipow_root(2)?;
                    Ok(left_unit)
                }
                Operator::Pow => {
                    if let Some(right_unit) = right_unit {
                        let right_unit = right_unit?;
                        if !right_unit.is_constant() {
                            return Err(SmootError::ParseTreeError(format!(
                                "line:{} Invalid operator {} ** {}",
                                lineno, left_unit.name, right_unit.name
                            )));
                        }

                        #[allow(clippy::cast_possible_truncation)]
                        let power = right_unit.multiplier.round() as i32;
                        left_unit.ipowi(power);
                        left_unit.multiplier = left_unit.multiplier.powi(power);
                        left_unit.power = 1;

                        Ok(left_unit)
                    } else {
                        Err(SmootError::ParseTreeError(format!(
                            "line:{} {} ** None expected a right operand",
                            lineno, left_unit.name,
                        )))
                    }
                }
                _ => Err(SmootError::ParseTreeError(format!(
                    "line:{lineno} Invalid operator {op:?}"
                ))),
            };
        }
        unreachable!();
    }

    fn is_comment(line: &str) -> bool {
        line.is_empty() || registry_parser::comment(line).is_ok()
    }

    fn add_prefix(prefix: String, name: &str) -> String {
        prefix + name
    }

    fn make_delta_unit_def<'a>(def: &UnitDefinition<'a>) -> UnitDefinition<'a> {
        let new_name = Registry::add_prefix(Registry::DELTA_PREFIX.to_string(), &def.name);

        let mut aliases = Vec::with_capacity(def.aliases.len() * 2); // delta_ and Δ
        for alias in &def.aliases {
            aliases.push(Registry::add_prefix(
                Registry::DELTA_PREFIX.to_string(),
                alias,
            ));
            aliases.push(Registry::add_prefix(
                Registry::DELTA_SYMBOL.to_string(),
                alias,
            ));
        }

        // Take the expression right of assignment e.g. `y` from `x = y`.
        let right_expr = *def.expression.right.as_ref().unwrap().clone();

        UnitDefinition {
            name: new_name.clone(),
            symbol: def
                .symbol
                .as_ref()
                .map(|s| Registry::add_prefix(Registry::DELTA_SYMBOL.to_string(), s)),
            expression: ParseTree::new(Operator::Assign.into(), new_name.into(), right_expr),
            modifiers: vec![],
            aliases,
            lineno: def.lineno,
        }
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
        LinkedHashMap<String, UnitDefinition<'input>>,
        Vec<DimensionDefinition<'input>>,
    )>
    where
        L: Iterator<Item = (usize, &'input str)>,
    {
        let mut unit_defs: LinkedHashMap<String, UnitDefinition> = LinkedHashMap::new();
        // Dimensions like `[length]` define the category that a unit belongs to.
        let mut dim_defs: Vec<DimensionDefinition> = Vec::new();

        for (lineno, line) in lines {
            if let Ok(dim_def) = registry_parser::dimension_definition(line, lineno) {
                dim_defs.push(dim_def);
            } else if let Ok(unit_def) = registry_parser::unit_with_aliases(line, lineno) {
                if !unit_def.modifiers.is_empty() {
                    let delta_unit_def = Registry::make_delta_unit_def(&unit_def);
                    try_insert!(
                        linked_hash_map,
                        lineno,
                        unit_defs,
                        delta_unit_def.name.clone(),
                        delta_unit_def
                    )?;
                }
                try_insert!(
                    linked_hash_map,
                    lineno,
                    unit_defs,
                    unit_def.name.clone(),
                    unit_def
                )?;
            } else if let Ok(_expr) = registry_parser::derived_dimension(line) {
                // We don't use derived dimensions in Smoot.
            } else if let Ok(prefix_def) = registry_parser::prefix_definition(line, lineno) {
                try_insert!(
                    hash_map,
                    lineno,
                    self.prefix_definitions,
                    prefix_def.name.clone(),
                    prefix_def
                )?;
            } else {
                return Err(SmootError::ParseTreeError(format!(
                    "line:{lineno} Unhandled line '{line}'"
                )));
            }
        }

        Ok((unit_defs, dim_defs))
    }

    /// Take a unit definition and add all prefixes and suffixes as new unit definitions.
    ///
    /// If we have an alias like `kilo- = 1000`, then the newly generated prefixed unit
    /// definitions contain `ParseTree`s for expressions like `kilometer = 1000 * meter`.
    ///
    /// Examples
    /// --------
    /// If we have a suffix `kilo-` and a unit `meter`, then we will generate
    /// `kilometer` and `kilometers`. Note that `kilometers` will be an alias of `kilometer`.
    fn create_unit_defs_from_prefixes<'a>(
        &self,
        unit_defs: &mut LinkedHashMap<String, UnitDefinition<'a>>,
        name: &str,
        data: &UnitAliasData<'a>,
    ) {
        // For now, hard-code plural suffix 's'.
        // TODO(jwh): This should probably be defined somewhere else, although it's unlikely that
        //            we'll have any more suffixes.
        for &suffix in &["", "s"] {
            // Add aliases e.g. 'sec' and 'secs' for 'second'
            for alias in &data.aliases {
                let new_alias = data.to_alias(alias.to_string(), data.name.clone());
                insert_ignore_conflict!(
                    linked_hash_map,
                    unit_defs,
                    alias.to_string() + suffix,
                    new_alias
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
                            name.to_owned().into(),
                        ),
                    ),
                    modifiers: data.modifiers.clone(),
                    // Clear the aliases to avoid redefining them
                    aliases: vec![],
                    lineno: data.lineno,
                };

                let prefix_name = Self::add_prefix(prefix_def.name.clone(), name);

                // Add prefixed symbol e.g. 'millis' and 'ms' for 'second'.
                // No plural forms for symbols e.g. 'ms' [length] would conflict with 'ms' [time].
                if let Some(symbol) = &data.symbol {
                    for prefix_alias in &prefix_def.aliases {
                        let alias_name = Self::add_prefix(prefix_alias.clone(), symbol);
                        let new_alias = data.to_alias(alias_name.clone(), prefix_name.clone());
                        insert_ignore_conflict!(linked_hash_map, unit_defs, alias_name, new_alias);
                    }

                    let alias_name = Self::add_prefix(prefix_def.name.clone(), symbol);
                    let new_alias = data.to_alias(alias_name.clone(), prefix_name.clone());
                    insert_ignore_conflict!(linked_hash_map, unit_defs, alias_name, new_alias);
                }

                // Add all combinations of prefix aliases and unit aliases e.g. 'msec' and 'msecs'.
                for alias in &data.aliases {
                    for prefix_alias in &prefix_def.aliases {
                        let alias_name = Self::add_prefix(prefix_alias.clone(), alias);
                        let new_alias = data.to_alias(alias_name.clone(), prefix_name.clone());
                        insert_ignore_conflict!(
                            linked_hash_map,
                            unit_defs,
                            alias_name + suffix,
                            new_alias
                        );
                    }

                    let alias_name = Self::add_prefix(prefix_def.name.clone(), alias);
                    let new_alias = data.to_alias(alias_name.clone(), prefix_name.clone());
                    insert_ignore_conflict!(
                        linked_hash_map,
                        unit_defs,
                        alias_name + suffix,
                        new_alias
                    );
                }

                // Add all prefix aliases to the unit name e.g. 'msecond'.
                for prefix_alias in &prefix_def.aliases {
                    let alias_name = Self::add_prefix(prefix_alias.clone(), &data.name);
                    let new_alias = data.to_alias(alias_name.clone(), prefix_name.clone());
                    insert_ignore_conflict!(
                        linked_hash_map,
                        unit_defs,
                        alias_name + suffix,
                        new_alias
                    );
                }

                // Add prefixed unit e.g. 'millisecond'.
                let new_def = f_make_new_def(prefix_name.clone());
                insert_ignore_conflict!(linked_hash_map, unit_defs, prefix_name + suffix, new_def);
            }
        }
    }

    /// Create root units from dimension definitions like `meter = [length]`.
    fn define_root_units(&mut self, dim_defs: &Vec<DimensionDefinition>) -> SmootResult<()> {
        let mut next_dimension: Dimension = DIMENSIONLESS;
        for def in dim_defs {
            let dims = if def.is_dimensionless() {
                vec![]
            } else {
                next_dimension += 1;
                vec![next_dimension]
            };

            let unit = if def.modifiers.is_empty() {
                BaseUnit::new(def.name.to_string(), 1.0, dims)
            } else {
                // Verify modifiers
                for &(mod_name, value) in &def.modifiers {
                    if mod_name != Registry::OFFSET_IDENTIFIER {
                        return Err(SmootError::ExpressionError(format!(
                            "line:{} Invalid unit modifier '{}'",
                            def.lineno, mod_name,
                        )));
                    }
                    if !value.approx_eq(0.0) {
                        // approx equality is fine, we set `0.0` explicitly below
                        return Err(SmootError::ExpressionError(format!(
                            "line:{} Dimension definition offset must be 0, got '{}'",
                            def.lineno, value,
                        )));
                    }
                }
                BaseUnit::new_offset(def.name.to_string(), 1.0, 0.0, dims)
            };

            if !unit.dimensionality.is_empty() {
                self.dimensions
                    .insert(unit.dimensionality[0].abs(), def.dimension.to_string());
                self.insert_root_unit(unit.clone());
            }

            // Symbol
            if let Some(symbol) = &def.symbol {
                self.insert_def(symbol.to_string(), &[], unit.clone());
                insert_ignore_conflict!(
                    hash_map,
                    self.symbols,
                    def.name.to_string(),
                    symbol.to_string()
                );
            }
            // Plural unit
            // No aliases needed since non-plural covers it
            self.insert_def(def.name.to_string() + "s", &[], unit.clone());
            // Non-plural
            self.insert_def(def.name.to_string(), &def.aliases, unit);
        }
        Ok(())
    }
}

/// Common data shared between various unit-like definitions.
struct UnitAliasData<'a> {
    name: String,
    lineno: usize,
    aliases: Vec<String>,
    modifiers: Vec<(&'a str, f64)>,
    symbol: Option<String>,
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
    fn test_registry_loads_from_cache() -> SmootResult<()> {
        let cache_path = Path::new(TEST_CACHE_PATH);
        Registry::clear_cache(cache_path);
        let _ = Registry::new_from_cache_or_file(cache_path, Path::new(DEFAULT_UNITS_FILE))?;
        // Second load from cached file
        let _ = Registry::new_from_cache_or_file(cache_path, Path::new(DEFAULT_UNITS_FILE))?;
        Ok(())
    }

    #[case(
        "percent = 0.01 = %",
        {
            let mut map = HashMap::default();
            map.extend([
                ("percent".to_string(), BaseUnit::new("percent".to_string(), 0.01, vec![])),
                ("percents".to_string(), BaseUnit::new("percent".to_string(), 0.01, vec![])),
                ("%".to_string(), BaseUnit::new("percent".to_string(), 0.01, vec![])),
            ]);
            Some(map)
        }
        ; "Dimensionless unit with alias parses"
    )]
    #[case(
        "# ignored comment\n\
        percent = 0.01 = %  # ignored comment",
        {
            let mut map = HashMap::default();
            map.extend([
                ("percent".to_string(), BaseUnit::new("percent".to_string(), 0.01, vec![])),
                ("percents".to_string(), BaseUnit::new("percent".to_string(), 0.01, vec![])),
                ("%".to_string(), BaseUnit::new("percent".to_string(), 0.01, vec![])),
            ]);
            Some(map)
        }
        ; "Comments are ignored"
    )]
    #[case(
        "kilo- = 1e3\n\
        meter = [length]",
        {
            let mut map = HashMap::default();
            map.extend([
                ("meter".to_string(), BaseUnit::new("meter".to_string(), 1.0, vec![1])),
                ("meters".to_string(), BaseUnit::new("meter".to_string(), 1.0, vec![1])),
                ("kilometer".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, vec![1])),
                ("kilometers".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, vec![1])),
            ]);
            Some(map)
        }
        ; "Prefixes are applied"
    )]
    #[case(
        "unit2 = 2 * unit1\n\
        unit1 = 1.0",
        {
            let mut map = HashMap::default();
            map.extend([
                ("unit1".to_string(), BaseUnit::new("unit1".to_string(), 1.0, vec![])),
                ("unit1s".to_string(), BaseUnit::new("unit1".to_string(), 1.0, vec![])),
                ("unit2".to_string(), BaseUnit::new("unit2".to_string(), 2.0, vec![])),
                ("unit2s".to_string(), BaseUnit::new("unit2".to_string(), 2.0, vec![])),
            ]);
            Some(map)
        }
        ; "Derived units can be defined in any order"
    )]
    #[case(
        "radian = []\n\
        steradian = radian ** 2",
        {
            let mut map = HashMap::default();
            map.extend([
                ("radian".to_string(), BaseUnit::new("radian".to_string(), 1.0, vec![])),
                ("radians".to_string(), BaseUnit::new("radian".to_string(), 1.0, vec![])),
                ("steradian".to_string(), BaseUnit::new("steradian".to_string(), 1.0, vec![])),
                ("steradians".to_string(), BaseUnit::new("steradian".to_string(), 1.0, vec![])),
            ]);
            Some(map)
        }
        ; "Derived dimensionless units stay unitless"
    )]
    #[case(
        "hour = 60 = h = hr",
        {
            let mut map = HashMap::default();
            map.extend([
                ("hour".to_string(), BaseUnit::new("hour".to_string(), 60.0, vec![])),
                ("hours".to_string(), BaseUnit::new("hour".to_string(), 60.0, vec![])),
                // Symbol should not have plural form
                ("h".to_string(), BaseUnit::new("hour".to_string(), 60.0, vec![])),
                ("hr".to_string(), BaseUnit::new("hour".to_string(), 60.0, vec![])),
                ("hrs".to_string(), BaseUnit::new("hour".to_string(), 60.0, vec![])),
            ]);
            Some(map)
        }
        ; "Aliases are plural but not symbols"
    )]
    #[case(
        "milli- = 1e-3\n\
        second = [time] = s = sec",
        {
            let mut map = HashMap::default();
            map.extend([
                ("second".to_string(), BaseUnit::new("second".to_string(), 1.0, vec![1])),
                ("seconds".to_string(), BaseUnit::new("second".to_string(), 1.0, vec![1])),
                ("millisecond".to_string(), BaseUnit::new("millisecond".to_string(), 1e-3, vec![1])),
                ("milliseconds".to_string(), BaseUnit::new("millisecond".to_string(), 1e-3, vec![1])),
                ("s".to_string(), BaseUnit::new("second".to_string(), 1.0, vec![1])),
                ("millis".to_string(), BaseUnit::new("millisecond".to_string(), 1e-3, vec![1])),
                ("sec".to_string(), BaseUnit::new("second".to_string(), 1.0, vec![1])),
                ("secs".to_string(), BaseUnit::new("second".to_string(), 1.0, vec![1])),
                ("millisec".to_string(), BaseUnit::new("millisecond".to_string(), 1e-3, vec![1])),
                ("millisecs".to_string(), BaseUnit::new("millisecond".to_string(), 1e-3, vec![1])),
            ]);
            Some(map)
        }
        ; "Prefixes and suffixes apply to dimension definitions"
    )]
    #[case(
        "kilo- = 1000\n\
        meter = 1 = m = metre",
        {
            let mut map = HashMap::default();
            map.extend([
                ("meter".to_string(), BaseUnit::new("meter".to_string(), 1.0, vec![])),
                ("meters".to_string(), BaseUnit::new("meter".to_string(), 1.0, vec![])),
                ("kilometer".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, vec![])),
                ("kilometers".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, vec![])),
                ("m".to_string(), BaseUnit::new("meter".to_string(), 1.0, vec![])),
                ("kilom".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, vec![])),
                ("metre".to_string(), BaseUnit::new("meter".to_string(), 1.0, vec![])),
                ("metres".to_string(), BaseUnit::new("meter".to_string(), 1.0, vec![])),
                ("kilometre".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, vec![])),
                ("kilometres".to_string(), BaseUnit::new("kilometer".to_string(), 1000.0, vec![])),
            ]);
            Some(map)
        }
        ; "Prefixes and suffixes apply to unit definitions"
    )]
    #[case(
        "u1 = 1000\n\
        u2 = u1 ** 2",
        {
            let mut map = HashMap::default();
            map.extend([
                ("u1".to_string(), BaseUnit::new("u1".to_string(), 1000.0, vec![])),
                ("u1s".to_string(), BaseUnit::new("u1".to_string(), 1000.0, vec![])),
                ("u2".to_string(), BaseUnit::new("u2".to_string(), 1000.0 * 1000.0, vec![])),
                ("u2s".to_string(), BaseUnit::new("u2".to_string(), 1000.0 * 1000.0, vec![])),
            ]);
            Some(map)
        }
    )]
    #[case(
        "kelvin = [temperature]; offset: 0\n\
        degree_Celsius = kelvin; offset: 273.15 = °C",
        {
            let mut map = HashMap::default();
            map.extend([
                ("kelvin".to_string(), BaseUnit::new_offset("kelvin".to_string(), 1.0, 0.0, vec![1])),
                ("kelvins".to_string(), BaseUnit::new_offset("kelvin".to_string(), 1.0, 0.0, vec![1])),
                ("degree_Celsius".to_string(), BaseUnit::new_offset("degree_Celsius".to_string(), 1.0, 273.15, vec![1])),
                ("degree_Celsiuss".to_string(), BaseUnit::new_offset("degree_Celsius".to_string(), 1.0, 273.15, vec![1])),
                ("°C".to_string(), BaseUnit::new_offset("degree_Celsius".to_string(), 1.0, 273.15, vec![1])),
                ("delta_degree_Celsius".to_string(), BaseUnit::new("delta_degree_Celsius".to_string(), 1.0, vec![1])),
                ("delta_degree_Celsiuss".to_string(), BaseUnit::new("delta_degree_Celsius".to_string(), 1.0, vec![1])),
                ("Δ°C".to_string(), BaseUnit::new("delta_degree_Celsius".to_string(), 1.0, vec![1])),
            ]);
            Some(map)
        }
        ; "Offset units create delta variants"
    )]
    fn test_registry_load_from_str(
        data: &str,
        expected_units: Option<HashMap<String, BaseUnit, FxBuildHasher>>,
    ) -> SmootResult<()> {
        let res = Registry::new_from_str(data);
        if let Some(expected_units) = expected_units {
            let res = res?;
            assert_eq!(
                res.units, expected_units,
                "{:#?} != {:#?}",
                res.units, expected_units
            );
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
