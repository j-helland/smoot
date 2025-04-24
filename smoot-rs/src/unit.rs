use std::{
    collections::HashMap,
    fmt::{self},
    mem::swap,
    ops::{Add, BitOr, Div, DivAssign, Mul, MulAssign, Sub},
};

use bitcode::{Decode, Encode};
use hashable::Hashable;
use itertools::{EitherOrBoth, Itertools};
use num_traits::ToPrimitive;

use crate::{
    base_unit::{BaseUnit, DIMENSIONLESS_TYPE, DimensionType},
    error::{SmootError, SmootResult},
    hash::Hash,
    parser::expression_parser,
    registry::Registry,
    utils::{ApproxEq, float_eq_rel},
};

type UnitDimensionality<N> = Vec<N>;

pub struct Dimensionality(pub HashMap<String, f64>);

#[derive(Encode, Decode, Hashable, Clone, Debug, PartialEq)]
pub struct Unit {
    numerator_units: Vec<BaseUnit>,
    numerator_dimension: DimensionType,
    numerator_dimensionality: UnitDimensionality<f64>,

    denominator_units: Vec<BaseUnit>,
    denominator_dimension: DimensionType,
    denominator_dimensionality: UnitDimensionality<f64>,
}

impl Unit {
    pub fn new(numerator_units: Vec<BaseUnit>, denominator_units: Vec<BaseUnit>) -> Self {
        let mut unit = Self {
            numerator_units,
            numerator_dimension: DIMENSIONLESS_TYPE,
            numerator_dimensionality: UnitDimensionality::new(),
            denominator_units,
            denominator_dimension: DIMENSIONLESS_TYPE,
            denominator_dimensionality: UnitDimensionality::new(),
        };
        unit.update_dimensionality();
        unit
    }

    pub fn new_dimensionless() -> Self {
        Self::new(vec![], vec![])
    }

    pub fn new_constant(multiplier: f64) -> Self {
        Self::new(vec![BaseUnit::new_constant(multiplier)], vec![])
    }

    /// Integer power operation that creates a new unit.
    pub fn powi(&self, n: i32) -> Self {
        if self.denominator_dimension == DIMENSIONLESS_TYPE
            && self.numerator_dimension == DIMENSIONLESS_TYPE
        {
            return Self::new_dimensionless();
        }
        match n {
            0 => Self::new_dimensionless(),
            1 => self.clone(),
            -1 => Self::new(self.denominator_units.clone(), self.numerator_units.clone()),
            _ => self.powf(n.into()),
        }
    }

    /// In-place integer power operation.
    pub fn ipowi(&mut self, n: i32) {
        if self.denominator_dimension == DIMENSIONLESS_TYPE
            && self.numerator_dimension == DIMENSIONLESS_TYPE
        {
            return;
        }
        match n {
            0 => *self = Self::new_dimensionless(),
            1 => {}
            -1 => {
                swap(&mut self.numerator_units, &mut self.denominator_units);
                swap(
                    &mut self.numerator_dimensionality,
                    &mut self.denominator_dimensionality,
                );
                swap(
                    &mut self.numerator_dimension,
                    &mut self.denominator_dimension,
                );
            }
            _ => self.ipowf(n.into()),
        }
    }

    /// Floating power operation that creates a new unit.
    pub fn powf(&self, n: f64) -> Self {
        let nabs = n.abs();
        let mut numerator = self
            .numerator_units
            .clone()
            .into_iter()
            .map(|mut u| {
                u.ipowf(nabs);
                u
            })
            .collect();
        let mut denominator = self
            .denominator_units
            .clone()
            .into_iter()
            .map(|mut u| {
                u.ipowf(nabs);
                u
            })
            .collect();

        // Flip if power sign is negative.
        if n < 0.0 {
            swap(&mut numerator, &mut denominator);
        }

        Self::new(numerator, denominator)
    }

    /// In-place floating power operation.
    pub fn ipowf(&mut self, n: f64) {
        let nabs = n.abs();
        self.numerator_units.iter_mut().for_each(|u| {
            u.ipowf(nabs);
        });
        self.denominator_units.iter_mut().for_each(|u| {
            u.ipowf(nabs);
        });

        // Flip if power sign is negative.
        if n < 0.0 {
            swap(&mut self.numerator_units, &mut self.denominator_units);
        }
    }

    /// Scale this unit by a constant multiplier.
    pub fn scale(&self, n: f64) -> Self {
        let mut new = self.clone();
        new.iscale(n);
        new
    }

    /// In-place scale this unit by a constant multiplier.
    pub fn iscale(&mut self, n: f64) {
        self.numerator_units.push(BaseUnit::new_constant(n));
    }

    /// Return true if this unit is dimensionless (i.e. has no associated base units).
    pub fn is_dimensionless(&self) -> bool {
        self.numerator_dimension == DIMENSIONLESS_TYPE
            && self.denominator_dimension == DIMENSIONLESS_TYPE
    }

    /// Return true if this unit is compatible with the target (e.g. meter and kilometer).
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        // Fast mask comparison
        let is_same_dimension = self.numerator_dimension == other.numerator_dimension
            && self.denominator_dimension == other.denominator_dimension;
        if !is_same_dimension {
            return false;
        }

        let mut result = true;

        // numerator
        let mut indices = self.numerator_dimension;
        let mut idx: u32;
        while result && indices > 0 {
            idx = indices.trailing_zeros();
            indices &= !(1 << idx);
            result &= float_eq_rel(
                self.numerator_dimensionality[idx as usize],
                other.numerator_dimensionality[idx as usize],
                1e-6,
            );
        }

        // denominator
        indices = self.denominator_dimension;
        while result && indices > 0 {
            idx = indices.trailing_zeros();
            indices &= !(1 << idx);
            result &= float_eq_rel(
                self.denominator_dimensionality[idx as usize],
                other.denominator_dimensionality[idx as usize],
                1e-6,
            );
        }

        result
    }

    /// Compute the multiplicative factor necessary to convert this unit into the target unit.
    ///
    /// Return
    /// ------
    /// Err if this unit is incompatible with the target unit (e.g. meter is incompatible with gram).
    pub fn conversion_factor(&self, other: &Self) -> SmootResult<f64> {
        if !self.is_compatible_with(other) {
            return Err(SmootError::IncompatibleUnitTypes(
                self.get_units_string(true)
                    .unwrap_or("dimensionless".into()),
                other
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into()),
            ));
        }

        let mut numerator_conversion_factor = 1.0;
        for (from, to) in self
            .numerator_units
            .iter()
            .zip(other.numerator_units.iter())
        {
            numerator_conversion_factor *= from.conversion_factor(to)?;
        }

        let mut denominator_conversion_factor = 1.0;
        for (from, to) in self
            .denominator_units
            .iter()
            .zip(other.denominator_units.iter())
        {
            denominator_conversion_factor *= from.conversion_factor(to)?;
        }

        Ok(numerator_conversion_factor / denominator_conversion_factor)
    }

    /// Format a float as a string.
    /// If the float is close to an integer, only display the integer part.
    fn format_float(p: f64) -> String {
        if float_eq_rel(p.fract(), 0.0, 1e-8) {
            format!("{}", p.to_i64().unwrap())
        } else {
            format!("{:?}", p)
        }
    }

    pub fn get_dimensionality(&self, registry: &Registry) -> Option<Dimensionality> {
        if self.is_dimensionless() {
            return None;
        }

        // Numerator
        let mut dims = HashMap::with_capacity(
            self.numerator_dimensionality
                .len()
                .max(self.denominator_dimensionality.len()),
        );

        self.numerator_dimensionality
            .iter()
            .enumerate()
            .filter(|(_, dim)| !dim.approx_eq(0.0))
            .for_each(|(idx, &dim)| {
                let dim_str = registry.get_dimension(&(1 << idx));
                dims.insert(dim_str.clone(), dim);
            });

        // Denominator
        self.denominator_dimensionality
            .iter()
            .enumerate()
            .filter(|(_, dim)| !dim.approx_eq(0.0))
            .for_each(|(idx, &dim)| {
                let dim_str = registry.get_dimension(&(1 << idx));
                dims.entry(dim_str.clone())
                    .and_modify(|d| *d -= dim)
                    .or_insert(-dim);
            });

        Some(Dimensionality(dims))
    }

    pub fn get_dimensionality_str(&self, registry: &Registry) -> Option<String> {
        self.get_dimensionality(registry).map(|dims| {
            dims.0
                .iter()
                .sorted_by_key(|(k, _)| k.as_str())
                .map(|(k, v)| format!("{}: {}", k, Self::format_float(*v)))
                .join(", ")
        })
    }

    /// Convert this unit into a displayable string representation.
    pub fn get_units_string(&self, with_scaling_factor: bool) -> Option<String> {
        let nums = self
            .numerator_units
            .iter()
            .sorted_by_key(|u| u.name.as_str())
            .collect_vec();
        let denoms = self
            .denominator_units
            .iter()
            .sorted_by_key(|u| u.name.as_str())
            .collect_vec();

        if nums.is_empty() && denoms.is_empty() {
            return None;
        }

        let mut numerator = nums
            .iter()
            .map(|u| {
                u.power
                    .map(Self::format_float)
                    .map(|s| format!("{} ** {}", u.name, s))
                    .unwrap_or_else(|| u.name.clone())
            })
            .join(" * ");

        if denoms.is_empty() {
            return Some(numerator);
        }

        let mut denominator = denoms
            .iter()
            .map(|u| {
                u.power
                    .map(Self::format_float)
                    .map(|s| format!("{} ** {}", u.name, s))
                    .unwrap_or_else(|| u.name.clone())
            })
            .join(" * ");

        if nums.len() > 1 && !denoms.is_empty() {
            numerator = format!("({})", numerator);
        }
        if denoms.len() > 1 {
            denominator = format!("({})", denominator);
        }

        if numerator.is_empty() {
            if with_scaling_factor {
                Some(format!("1 / {}", denominator))
            } else {
                Some(format!("/ {}", denominator))
            }
        } else {
            Some(format!("{} / {}", numerator, denominator))
        }
    }

    /// Sync the dimensionality of this unit with its numerator and denominator base units.
    ///
    /// This is invoked automatically during unit reduction / simplification.
    fn update_dimensionality(&mut self) {
        // Use bitmask to get the largest dimension index.
        // If we bitwise or all masks together, the index of the MSB indicates the largest size we need.
        let max_dim = DimensionType::BITS
            - Self::get_dimension_mask(&self.numerator_units)
                .bitor(Self::get_dimension_mask(&self.denominator_units))
                .leading_zeros();
        let max_dim = max_dim as usize;

        // Reset
        self.numerator_dimension = 0;
        self.denominator_dimension = 0;

        self.numerator_dimensionality.fill(0.0);
        self.numerator_dimensionality.resize(max_dim, 0.0);
        self.denominator_dimensionality.fill(0.0);
        self.denominator_dimensionality.resize(max_dim, 0.0);

        // Update
        // A negative dimension in the numerator corresponds to a positive dimension in the denominator.
        for u in self.numerator_units.iter() {
            for (i, &dim) in u.dimensionality.iter().enumerate() {
                let b = u.unit_type & (1 << i);
                if dim < 0.0 {
                    self.denominator_dimension |= b;
                    self.denominator_dimensionality[i] += dim.abs();
                } else {
                    self.numerator_dimension |= b;
                    self.numerator_dimensionality[i] += dim;
                }
            }
        }
        for u in self.denominator_units.iter() {
            for (i, &dim) in u.dimensionality.iter().enumerate() {
                let b = u.unit_type & (1 << i);
                if dim < 0.0 {
                    self.numerator_dimension |= b;
                    self.numerator_dimensionality[i] += dim.abs();
                } else {
                    self.denominator_dimension |= b;
                    self.denominator_dimensionality[i] += dim;
                }
            }
        }
    }

    /// Merge compatible units e.g. `meter * km -> meter ** 2`.
    ///
    /// Return
    /// ------
    /// The multiplicative factor computed during reduction, which may not be one
    /// depending on unit conversion factors.
    pub fn reduce(&mut self) -> f64 {
        // Simplify first to cancel matching units across numerator and denominator.
        // After that, units can be combined independently in the numerator and denominator.
        let mut result_conversion_factor = self.simplify(false);

        // Merge units.
        let mut reduce_func = |units: &mut Vec<BaseUnit>, is_denom: bool| {
            let mut units_reduced: Vec<BaseUnit> = Vec::new();
            units.drain(..).for_each(|next| {
                if next.is_constant() {
                    if is_denom {
                        result_conversion_factor /= next.get_multiplier();
                    } else {
                        result_conversion_factor *= next.get_multiplier();
                    }
                    // Dimensionless constants can be immediately removed.
                    return;
                }

                if let Some(last) = units_reduced.last_mut() {
                    if last.unit_type == next.unit_type {
                        // Aggregate conversion factors
                        let factor = last.conversion_factor_unchecked(&next);
                        if is_denom {
                            result_conversion_factor /= factor;
                        } else {
                            result_conversion_factor *= factor;
                        }

                        last.name = next.name;
                        last.multiplier = next.multiplier;
                        let next_power = next.power.unwrap_or(1.0);
                        last.power = last
                            .power
                            .map(|p| p + next_power)
                            .or(Some(1.0 + next_power));
                        last.dimensionality = last
                            .dimensionality
                            .drain(..)
                            .zip_longest(next.dimensionality.iter())
                            .map(|lr| match lr {
                                EitherOrBoth::Both(l, &r) => l + r,
                                EitherOrBoth::Left(l) => l,
                                EitherOrBoth::Right(r) => *r,
                            })
                            .collect();
                        return;
                    }
                }
                units_reduced.push(next);
            });
            units.extend(units_reduced);
        };

        reduce_func(&mut self.numerator_units, false);
        reduce_func(&mut self.denominator_units, true);
        self.update_dimensionality();

        result_conversion_factor
    }

    /// Reduce this unit into its simplest form (e.g. "meter ** 2 / meter -> meter").
    ///
    /// Return
    /// ------
    /// The multiplicative factor resulting from the simplification, which might not be
    /// one depending on unit conversions that occurred.
    pub fn simplify(&mut self, no_reduction: bool) -> f64 {
        let mut result_conversion_factor = 1.0;

        // Sort into unit type groups to find all possible cancellations.
        self.numerator_units.sort_by(|u1, u2| {
            (u1.unit_type, u1.power.map(|p| -p))
                .partial_cmp(&(u2.unit_type, u2.power.map(|p| -p)))
                .unwrap()
        });
        self.denominator_units.sort_by(|u1, u2| {
            (u1.unit_type, u1.power.map(|p| -p))
                .partial_cmp(&(u2.unit_type, u2.power.map(|p| -p)))
                .unwrap()
        });

        // Markers for numerator and denominator units indicating whether a cancellation occurred.
        let mut numerator_retain = vec![true; self.numerator_units.len()];
        let mut denominator_retain = vec![true; self.denominator_units.len()];

        // Find cancellations.
        let mut inum = 0;
        let mut iden = 0;
        while inum < self.numerator_units.len() && iden < self.denominator_units.len() {
            let u1 = &mut self.numerator_units[inum];
            let u2 = &mut self.denominator_units[iden];

            // Skip if mismatched units.
            match u1.unit_type.cmp(&u2.unit_type) {
                std::cmp::Ordering::Less => {
                    inum += 1;
                    continue;
                }
                std::cmp::Ordering::Greater => {
                    iden += 1;
                    continue;
                }
                _ => (),
            }
            if u1.is_multidimensional() || u2.is_multidimensional() {
                // If the base units are composites (i.e. contain multiple dimensions), we cannot simplify
                // them without breaking them down via `ito_root_units`.
                inum += 1;
                iden += 1;
                continue;
            }

            if u1.unit_type == u2.unit_type {
                let factor = u1.conversion_factor_unchecked(u2);
                if no_reduction && !factor.approx_eq(1.0) {
                    // Make sure we don't reduce units with disparate scales in no_reduction mode.
                    inum += 1;
                    iden += 1;
                    continue;
                }

                // Now there must be a cancellation.
                result_conversion_factor *= factor;
            }

            // Unit exponents (e.g. meter ** 2) may result in a cancellation without completely removing
            // the unit from the numerator/denominator.
            let u1_power = u1.power.unwrap_or(1.0);
            let u2_power = u2.power.unwrap_or(1.0);
            if u1_power == u2_power {
                numerator_retain[inum] = false;
                denominator_retain[iden] = false;
            } else if u1_power < u2_power {
                numerator_retain[inum] = false;
                u2.sub_power(u1_power);
            } else if u1_power > u2_power {
                denominator_retain[iden] = false;
                u1.sub_power(u2_power);
            }

            inum += 1;
            iden += 1;
        }

        // Apply cancellations.
        let mut idx = 0;
        self.numerator_units.retain(|_| {
            idx += 1;
            numerator_retain[idx - 1]
        });
        idx = 0;
        self.denominator_units.retain(|_| {
            idx += 1;
            denominator_retain[idx - 1]
        });

        // Unit cancellations require syncing dimensionality.
        self.update_dimensionality();

        result_conversion_factor
    }

    /// Simplify this unit into the smallest number of root units possible.
    ///
    /// Return
    /// ------
    /// The multiplicative factor resulting from converting this unit into
    /// its constituent root units.
    ///
    /// Examples
    /// --------
    /// `newton.ito_root_units()`
    ///
    /// => `1000 * (gram * meter / second ** 2)`
    pub fn ito_root_units(&mut self, registry: &Registry) -> f64 {
        let mut factor = 1.0;

        let mut numerator = Vec::with_capacity(self.numerator_units.len());
        let mut denominator = Vec::with_capacity(self.denominator_units.len());

        for u in self.numerator_units.iter() {
            factor *= u.get_multiplier();
            Self::update_with_root_units(&mut numerator, &mut denominator, u, registry);
        }
        for u in self.denominator_units.iter() {
            factor /= u.get_multiplier();
            // Swap numerator and denominator because this is a division.
            Self::update_with_root_units(&mut denominator, &mut numerator, u, registry);
        }
        self.numerator_units = numerator;
        self.denominator_units = denominator;

        factor *= self.simplify(true);
        factor
    }

    /// Append numerator and denominator with the root units of the given BaseUnit.
    ///
    /// Examples
    /// --------
    /// `newton = kilogram * meter / second ** 2`
    ///
    /// => `numerator = [gram, meter]; denominator = [second ** 2]`
    fn update_with_root_units(
        numerator: &mut Vec<BaseUnit>,
        denominator: &mut Vec<BaseUnit>,
        base: &BaseUnit,
        registry: &Registry,
    ) {
        for (i, &dim) in base.dimensionality.iter().enumerate() {
            if dim.approx_eq(0.0) {
                continue;
            }
            let dim_abs = dim.abs();

            let dim_type = 1 << i;
            let root = registry.get_root_unit(&dim_type);
            let root = if dim_abs.approx_eq(1.0) {
                root.clone()
            } else {
                root.powf(dim_abs)
            };

            if dim.is_sign_negative() {
                denominator.push(root);
            } else {
                numerator.push(root);
            }
        }
    }

    fn get_dimension_mask(units: &[BaseUnit]) -> DimensionType {
        units.iter().fold(0, |d, u| d | u.unit_type)
    }
}

impl Unit {
    /// Parse a unit expression into its resulting unit.
    ///
    /// Return
    /// ------
    /// (f64, Unit) A tuple whose first element is the multiplicative factor computed during parsing.
    ///             e.g. `2 * meter` returns a multiplicative factor of `2`.
    pub fn parse(registry: &Registry, s: &str) -> SmootResult<(f64, Self)> {
        if s == "dimensionless" {
            // Make sure to return an empty unit container.
            return Ok((1.0, Self::new_dimensionless()));
        }
        expression_parser::unit_expression(s, registry)
            .map(|mut u| {
                let factor = u.reduce();
                (factor, u)
            })
            .map_err(|_| SmootError::ExpressionError(format!("Invalid unit expression {}", s)))
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            // Default to displaying unitless units as `dimensionless`.
            self.get_units_string(true)
                .unwrap_or("dimensionless".into())
        )
    }
}

//==================================================
// Arithmetic operators for Unit
//==================================================
impl Mul for Unit {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.numerator_units.extend_from_slice(&rhs.numerator_units);
        self.denominator_units
            .extend_from_slice(&rhs.denominator_units);
        let _ = self.simplify(true);
        self
    }
}
impl MulAssign<&Unit> for Unit {
    fn mul_assign(&mut self, rhs: &Unit) {
        self.numerator_units.extend_from_slice(&rhs.numerator_units);
        self.denominator_units
            .extend_from_slice(&rhs.denominator_units);
        let _ = self.simplify(true);
    }
}

impl Div for Unit {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        self.numerator_units
            .extend_from_slice(&rhs.denominator_units);
        self.denominator_units
            .extend_from_slice(&rhs.numerator_units);
        let _ = self.simplify(true);
        self
    }
}
impl DivAssign<&Unit> for Unit {
    fn div_assign(&mut self, rhs: &Unit) {
        self.numerator_units
            .extend_from_slice(&rhs.denominator_units);
        self.denominator_units
            .extend_from_slice(&rhs.numerator_units);
        let _ = self.simplify(true);
    }
}

impl Add for Unit {
    type Output = SmootResult<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        if !self.is_compatible_with(&rhs) {
            return Err(SmootError::InvalidOperation(
                "+",
                self.get_units_string(true)
                    .unwrap_or("dimensionless".into()),
                rhs.get_units_string(true).unwrap_or("dimensionless".into()),
            ));
        }
        Ok(self)
    }
}

impl Sub for Unit {
    type Output = SmootResult<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        if !self.is_compatible_with(&rhs) {
            return Err(SmootError::InvalidOperation(
                "-",
                self.get_units_string(true)
                    .unwrap_or("dimensionless".into()),
                rhs.get_units_string(true).unwrap_or("dimensionless".into()),
            ));
        }
        Ok(self)
    }
}

//==================================================
// Unit tests
//==================================================
#[cfg(test)]
mod test_unit {
    use crate::{assert_is_close, test_utils::TEST_REGISTRY};

    use super::*;
    use std::{
        f64,
        hash::{DefaultHasher, Hasher},
        sync::LazyLock,
    };
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
    static UNIT_MINUTE: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("minute").expect("No unit 'minute'"));
    static UNIT_HOUR: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("hour").expect("No unit 'hour'"));
    static UNIT_GRAM: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("gram").expect("No unit 'gram'"));
    static UNIT_WATT: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("watt").expect("No unit 'watt'"));
    static UNIT_NEWTON: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("newton").expect("No unit 'newton'"));
    static UNIT_JOULE: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("joule").expect("No unit 'joule'"));
    static UNIT_PERCENT: LazyLock<&BaseUnit> = LazyLock::new(|| {
        TEST_REGISTRY
            .get_unit("percent")
            .expect("No unit 'percent'")
    });
    static UNIT_RADIAN: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("radian").expect("No unit 'radian'"));
    static UNIT_MOLE: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("mole").expect("No unit 'mole'"));
    static UNIT_AVOGADRO_CONSTANT: LazyLock<&BaseUnit> = LazyLock::new(|| {
        TEST_REGISTRY
            .get_unit("avogadro_constant")
            .expect("No unit 'avogadro_constant'")
    });

    #[case(
        Unit::new_dimensionless(),
        Unit::new_dimensionless(),
        Some(1.0)
        ; "Trivial conversion factor"
    )]
    #[case(
        Unit::new(vec![UNIT_SECOND.clone()], vec![]),
        Unit::new(vec![UNIT_MINUTE.clone()], vec![]),
        Some(1.0 / 60.0)
        ; "Basic conversion factor"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone()], vec![UNIT_SECOND.clone()]),
        Unit::new(vec![UNIT_KILOMETER.clone()], vec![UNIT_HOUR.clone()]),
        Some(60.0 * 60.0 / 1000.0)
        ; "Composite conversion factor"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone()], vec![]),
        Unit::new(vec![UNIT_SECOND.clone()], vec![]),
        None
        ; "Incompatible units"
    )]
    fn test_conversion_factor(u1: Unit, u2: Unit, expected: Option<f64>) -> SmootResult<()> {
        if let Some(expected) = expected {
            assert_is_close!(u1.conversion_factor(&u2)?, expected);
            assert_is_close!(u2.conversion_factor(&u1)?, 1.0 / expected);
        } else {
            assert!(u1.conversion_factor(&u2).is_err());
        }
        Ok(())
    }

    #[case(
        Unit::new_dimensionless(),
        None
        ; "Dimensionless"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone()], vec![UNIT_SECOND.clone()]),
        Some("meter / second")
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone(), UNIT_SECOND.clone()], vec![]),
        Some("meter * second")
    )]
    #[case(
        Unit::new(
            vec![UNIT_METER.clone(), UNIT_WATT.clone()],
            vec![UNIT_SECOND.clone(), UNIT_GRAM.clone()],
        ),
        Some("(meter * watt) / (gram * second)")
        ; "Multiple units in the numerator and denominator are parenthesized and sorted"
    )]
    #[case(
        Unit::new(
            vec![UNIT_METER.clone(), UNIT_METER.clone()],
            vec![UNIT_SECOND.clone(), UNIT_SECOND.clone()],
        ),
        Some("(meter * meter) / (second * second)")
        ; "Repeated multiplication is not reduced"
    )]
    #[case(
        Unit::new(
            vec![UNIT_METER.powf(2.0)],
            vec![UNIT_SECOND.powf(2.0)],
        ),
        Some("meter ** 2 / second ** 2")
        ; "Powers"
    )]
    #[case(
        Unit::new(
            vec![UNIT_METER.powf(2.5)],
            vec![],
        ),
        Some("meter ** 2.5")
        ; "Fractional powers"
    )]
    #[case(
        Unit::new(
            vec![],
            vec![UNIT_METER.clone()],
        ),
        Some("1 / meter")
        ; "No numerator"
    )]
    #[case(
        Unit::new(vec![UNIT_PERCENT.clone()], vec![]),
        Some("percent")
        ; "Named dimensionless unit"
    )]
    fn test_get_units_string(u: Unit, expected: Option<&str>) {
        assert_eq!(u.get_units_string(true), expected.map(String::from));
    }

    #[case(
        Unit::new(
            vec![UNIT_METER.clone()],
            vec![],
        ),
        Some("[length]: 1")
        ; "Numerator"
    )]
    #[case(
        Unit::new(
            vec![],
            vec![UNIT_METER.clone()],
        ),
        Some("[length]: -1")
        ; "Denominator"
    )]
    #[case(
        Unit::new(
            vec![UNIT_METER.powf(2.5)],
            vec![],
        ),
        Some("[length]: 2.5")
        ; "Power"
    )]
    #[case(
        Unit::new(
            vec![UNIT_METER.clone(), UNIT_KILOMETER.clone()],
            vec![],
        ),
        Some("[length]: 2")
        ; "Multi-unit"
    )]
    #[case(
        Unit::new(vec![], vec![]),
        None
        ; "Dimensionless"
    )]
    #[case(
        Unit::new(
            vec![UNIT_NEWTON.clone()],
            vec![],
        ),
        Some("[length]: 1, [mass]: 1, [time]: -2")
        ; "Multidimensional base unit"
    )]
    fn test_get_dimensionality_string(u: Unit, expected: Option<&str>) {
        assert_eq!(
            u.get_dimensionality_str(&TEST_REGISTRY),
            expected.map(String::from)
        );
    }

    #[test]
    fn test_mul_numerator() {
        let u1 = Unit::new(vec![UNIT_SECOND.clone()], vec![]);
        let u2 = Unit::new(vec![UNIT_METER.clone()], vec![]);

        let u = u1 * u2;

        assert_eq!(
            u.numerator_units,
            vec![UNIT_METER.clone(), UNIT_SECOND.clone()]
                .into_iter()
                .sorted_by_key(|u| u.unit_type)
                .collect_vec()
        );
        assert!(u.denominator_units.is_empty());
    }

    #[test]
    fn test_mul_denominator() {
        let u1 = Unit::new(vec![], vec![UNIT_SECOND.clone()]);
        let u2 = Unit::new(vec![], vec![UNIT_METER.clone()]);

        let u = u1 * u2;

        assert!(u.numerator_units.is_empty());
        assert_eq!(
            u.denominator_units,
            vec![UNIT_METER.clone(), UNIT_SECOND.clone()]
                .into_iter()
                .sorted_by_key(|u| u.unit_type)
                .collect_vec()
        );
    }

    #[test]
    /// Multiplication of units should trigger simplification.
    fn tet_mul_simplifies_units() {
        let u1 = Unit::new(vec![UNIT_SECOND.clone()], vec![UNIT_METER.clone()]);
        let u2 = Unit::new(vec![UNIT_METER.clone()], vec![UNIT_SECOND.clone()]);

        let u = u1 * u2;

        assert!(u.numerator_units.is_empty());
        assert!(u.denominator_units.is_empty());
    }

    #[test]
    fn test_div_numerator() {
        let u1 = Unit::new(vec![UNIT_METER.clone()], vec![]);
        let u2 = Unit::new(vec![UNIT_SECOND.clone()], vec![]);

        let u = u1 / u2;

        assert_eq!(u.numerator_units, vec![UNIT_METER.clone()]);
        assert_eq!(u.denominator_units, vec![UNIT_SECOND.clone()]);
    }

    #[test]
    fn test_div_denominator() {
        let u1 = Unit::new(vec![], vec![UNIT_METER.clone()]);
        let u2 = Unit::new(vec![], vec![UNIT_SECOND.clone()]);

        let u = u1 / u2;

        assert_eq!(u.numerator_units, vec![UNIT_SECOND.clone()]);
        assert_eq!(u.denominator_units, vec![UNIT_METER.clone()]);
    }

    #[test]
    fn test_div_simplifies_units() {
        let u1 = Unit::new(vec![UNIT_SECOND.clone()], vec![]);
        let u2 = Unit::new(vec![UNIT_SECOND.clone()], vec![]);

        let u = u1 / u2;

        assert!(u.numerator_units.is_empty());
        assert!(u.denominator_units.is_empty());
    }

    #[test]
    /// Simplifying units must compute the correct conversion factor for the cancelled units.
    fn test_simplify_computes_conversion_factor() {
        let mut u = Unit::new(
            vec![UNIT_SECOND.clone(), UNIT_MINUTE.clone()],
            vec![UNIT_HOUR.clone()],
        );

        let conversion_factor = u.simplify(false);

        assert_is_close!(conversion_factor, 1.0 / 60.0 / 60.0);
        assert_eq!(u.numerator_units, vec![UNIT_MINUTE.clone()]);
        assert!(u.denominator_units.is_empty());
    }

    #[test]
    /// Simpliyfing will cancel multiple units within the same unit type.
    fn test_simplify_cancels_multiple_units() {
        let mut u = Unit::new(
            vec![UNIT_SECOND.clone(), UNIT_SECOND.clone()],
            vec![
                UNIT_SECOND.clone(),
                UNIT_SECOND.clone(),
                UNIT_SECOND.clone(),
            ],
        );

        let _ = u.simplify(true);

        assert!(u.numerator_units.is_empty());
        assert_eq!(u.denominator_units, vec![UNIT_SECOND.clone()]);
    }

    #[test]
    /// Simplifying will cancel units across unit types.
    fn test_simplify_cancels_units_of_different_types() {
        let mut u = Unit::new(
            vec![UNIT_SECOND.clone(), UNIT_METER.clone()],
            vec![UNIT_SECOND.clone(), UNIT_METER.clone()],
        );

        let _ = u.simplify(false);

        assert!(u.numerator_units.is_empty());
        assert!(u.denominator_units.is_empty());
    }

    #[test]
    /// Simplifying units correctly updates the dimensionality of the numerator and denominator.
    fn test_simplify_updates_unit_dimensionality() {
        let mut u = Unit::new(
            vec![UNIT_SECOND.clone(), UNIT_SECOND.clone(), UNIT_METER.clone()],
            vec![UNIT_SECOND.clone(), UNIT_METER.clone()],
        );

        let _ = u.simplify(true);

        assert_eq!(u.denominator_dimension, DIMENSIONLESS_TYPE);
        assert!(
            u.denominator_dimensionality
                .iter()
                .all(|d| d.approx_eq(0.0))
        );
        assert_eq!(u.numerator_dimension, UNIT_SECOND.unit_type);
        assert_eq!(u.numerator_dimensionality, UNIT_SECOND.dimensionality);
    }

    #[test]
    fn test_simplify_with_mixed_ordering() {
        let mut u = Unit::new(
            vec![UNIT_SECOND.clone(), UNIT_METER.clone(), UNIT_SECOND.clone()],
            vec![UNIT_METER.clone(), UNIT_SECOND.clone()],
        );

        let _ = u.simplify(true);

        assert_eq!(u.numerator_units, vec![UNIT_SECOND.clone()]);
        assert!(u.denominator_units.is_empty());
    }

    /// Numerator units with exponents are correctly simplified.
    #[test]
    fn test_simplify_adjusts_numerator_powers() {
        let mut u = Unit::new(vec![UNIT_METER.powf(2.0)], vec![UNIT_METER.clone()]);
        let expected = Unit::new(vec![UNIT_METER.clone()], vec![]);

        let _ = u.simplify(true);

        assert_eq!(u, expected);
    }

    /// Denominator units with exponents are correctly simplified.
    #[test]
    fn test_simplify_adjusts_denominator_powers() {
        let mut u = Unit::new(vec![UNIT_METER.clone()], vec![UNIT_METER.powf(2.0)]);
        let expected = Unit::new(vec![], vec![UNIT_METER.clone()]);

        let _ = u.simplify(true);

        assert_eq!(u, expected);
    }

    #[test]
    fn test_simplify_with_fractional_powers() {
        let mut u = Unit::new(vec![UNIT_KILOMETER.powf(0.5)], vec![UNIT_METER.clone()]);

        let conversion_factor = u.simplify(false);

        assert_is_close!(conversion_factor, 1000.0_f64.sqrt());
        assert!(u.numerator_units.is_empty());
        assert_eq!(u.denominator_units, vec![UNIT_METER.powf(0.5)]);
    }

    #[case(
        Unit::new_dimensionless(),
        Unit::new_dimensionless(),
        true
        ; "Trivial"
    )]
    #[case(
        Unit::new_dimensionless(),
        Unit::new(vec![UNIT_PERCENT.clone()], vec![]),
        true
        ; "Dimensionless <> named dimensionless"
    )]
    #[case(
        Unit::new(vec![UNIT_RADIAN.clone()], vec![]),
        Unit::new(vec![UNIT_PERCENT.clone()], vec![]),
        true
        ; "Named dimensionless <> dimensionless"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone()], vec![]),
        Unit::new(vec![UNIT_METER.clone()], vec![]),
        true
        ; "Basic compatibility"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone(), UNIT_SECOND.clone()], vec![UNIT_METER.clone()]),
        Unit::new(vec![UNIT_METER.clone(), UNIT_SECOND.clone()], vec![UNIT_METER.clone()]),
        true
        ; "Complex compatibility"
    )]
    #[case(
        Unit::new(vec![UNIT_SECOND.clone()], vec![]),
        Unit::new(vec![UNIT_METER.clone()], vec![]),
        false
        ; "Basic incompatibility"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone(), UNIT_SECOND.clone()], vec![]),
        Unit::new(vec![UNIT_SECOND.clone(), UNIT_METER.clone()], vec![]),
        true
        ; "Invariant to ordering"
    )]
    #[case(
        Unit::new(vec![], vec![UNIT_METER.clone()]),
        Unit::new(vec![], vec![UNIT_METER.clone()]),
        true
        ; "Basic denominator compatibility"
    )]
    #[case(
        Unit::new(vec![], vec![UNIT_METER.clone()]),
        Unit::new(vec![], vec![UNIT_SECOND.clone()]),
        false
        ; "Basic denominator incompatibility"
    )]
    #[case(
        Unit::new(vec![UNIT_SECOND.clone(), UNIT_SECOND.clone()], vec![]),
        Unit::new(vec![UNIT_SECOND.clone()], vec![]),
        false
        ; "Dimensionality incompatibility"
    )]
    #[case(
        Unit::new(vec![UNIT_NEWTON.clone()], vec![]),
        Unit::new(vec![UNIT_JOULE.clone()], vec![]),
        false
        ; "Composite incompatible units"
    )]
    #[case(
        Unit::new(vec![UNIT_AVOGADRO_CONSTANT.clone()], vec![]),
        Unit::new(vec![], vec![UNIT_MOLE.clone()]),
        true
        ; "Composite unit avogadro_constant is equivalent to 1 / mole"
    )]
    fn test_is_compatible_with(u1: Unit, u2: Unit, expected: bool) {
        assert_eq!(
            u1.is_compatible_with(&u2),
            expected,
            "{:#?}\nnot compatible with\n{:#?}",
            u1,
            u2
        );
    }

    #[test]
    fn test_is_compatible_with_incompatible_units() {
        let u1 = Unit::new(vec![UNIT_METER.clone()], vec![]);
        let u2 = Unit::new_dimensionless();
        assert!(!u1.is_compatible_with(&u2));
    }

    #[test]
    /// Dimensionless units should be self-compatible.
    fn test_is_compatible_with_dimensionless() {
        let u1: Unit = Unit::new_dimensionless();
        // This is equivalent to a dimensionless unit.
        let u2: Unit = Unit::new_dimensionless();
        assert!(u1.is_compatible_with(&u2));
    }

    #[test]
    /// Floating point imprecision should not cause units to be considered incompatible.
    fn test_is_compatible_with_float_imprecision() {
        let u1 = Unit::new(vec![UNIT_METER.clone()], vec![]);
        let mut u2 = u1.clone();
        u2.ipowf(1.0 + f64::EPSILON);
        assert!(u1.is_compatible_with(&u2));
    }

    #[test]
    /// Fractional powers should be checked for compatibility.
    fn test_is_compatible_fractional_powers() {
        let u1 = Unit::new(vec![UNIT_METER.clone()], vec![]);
        let u2 = Unit::new(vec![UNIT_METER.powf(0.5)], vec![]);
        assert!(!u1.is_compatible_with(&u2));
    }

    #[case(
        Unit::new(vec![UNIT_KILOMETER.clone(), UNIT_METER.clone()], vec![]),
        Unit::new(vec![UNIT_METER.powf(2.0)], vec![]),
        1000.0
        ; "Numerator"
    )]
    #[case(
        Unit::new(vec![], vec![UNIT_KILOMETER.clone(), UNIT_METER.clone()]),
        Unit::new(vec![], vec![UNIT_METER.powf(2.0)]),
        1e-3
        ; "Denominator"
    )]
    #[case(
        Unit::new(vec![UNIT_JOULE.clone()], vec![UNIT_NEWTON.clone()]),
        Unit::new(vec![UNIT_JOULE.clone()], vec![UNIT_NEWTON.clone()]),
        1.0
        ; "Incompatible units with matching dimensions do not reduce"
    )]
    #[case(
        Unit::new(vec![UNIT_KILOMETER.powf(0.5), UNIT_METER.clone()], vec![]),
        Unit::new(vec![UNIT_KILOMETER.powf(1.5)], vec![]),
        1e-3_f64.sqrt()
        ; "Units with fractional powers"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.powf(0.5), UNIT_KILOMETER.clone()], vec![]),
        Unit::new(vec![UNIT_METER.powf(1.5)], vec![]),
        1000.0
        ; "Units with fractional powers 2"
    )]
    #[case(
        Unit::new(vec![BaseUnit::new_constant(2.0), UNIT_METER.clone()], vec![]),
        Unit::new(vec![UNIT_METER.clone()], vec![]),
        2.0
        ; "Constants are removed from numerator"
    )]
    #[case(
        Unit::new(vec![], vec![BaseUnit::new_constant(2.0), UNIT_METER.clone()]),
        Unit::new(vec![], vec![UNIT_METER.clone()]),
        1.0 / 2.0
        ; "Constants are removed from denominator"
    )]
    fn test_reduce(mut unit: Unit, expected: Unit, expected_factor: f64) {
        assert_is_close!(unit.reduce(), expected_factor);
        assert_is_close!(unit.reduce(), 1.0); // idempotent
        assert_eq!(unit, expected);
    }

    #[case(
        Unit::new(vec![UNIT_SECOND.clone()], vec![UNIT_METER.clone()]),
        Unit::new(vec![UNIT_SECOND.clone()], vec![UNIT_METER.clone()]),
        1.0
        ; "noop"
    )]
    #[case(
        Unit::new(vec![UNIT_KILOMETER.clone()], vec![]),
        Unit::new(vec![UNIT_METER.clone()], vec![]),
        1e3
        ; "numerator"
    )]
    #[case(
        Unit::new(vec![], vec![UNIT_KILOMETER.clone()]),
        Unit::new(vec![], vec![UNIT_METER.clone()]),
        1e-3
        ; "denominator"
    )]
    #[case(
        Unit::new(vec![UNIT_KILOMETER.clone(), UNIT_MINUTE.clone()], vec![]),
        Unit::new(vec![UNIT_METER.clone(), UNIT_SECOND.clone()], vec![]),
        1e3 * 60.0
        ; "multiple units"
    )]
    #[case(
        Unit::new_dimensionless(),
        Unit::new_dimensionless(),
        1.0
        ; "dimensionless"
    )]
    #[case(
        Unit::new(vec![UNIT_NEWTON.clone()], vec![]),
        Unit::new(vec![UNIT_METER.clone(), UNIT_GRAM.clone()], vec![UNIT_SECOND.powf(2.0)]),
        1000.0
        ; "Multidimensional base unit"
    )]
    #[case(
        Unit::new(vec![UNIT_JOULE.clone()], vec![UNIT_NEWTON.clone()]),
        Unit::new(vec![UNIT_METER.clone()], vec![]),
        1.0
        ; "Simplified"
    )]
    fn test_ito_root_unit(mut unit: Unit, expected: Unit, expected_factor: f64) {
        assert_is_close!(unit.ito_root_units(&TEST_REGISTRY), expected_factor);
        assert_eq!(unit, expected);
    }

    #[test]
    fn test_hash() {
        let u1 = Unit::new(vec![UNIT_METER.clone()], vec![UNIT_SECOND.clone()]);
        assert_eq!(hash(&u1), hash(&u1.clone()));

        let u2 = u1.powi(-1);
        assert_ne!(hash(&u1), hash(&u2));
    }

    fn hash<T: Hash>(val: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        val.hash(&mut hasher);
        hasher.finish()
    }
}
