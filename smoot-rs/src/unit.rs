use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{self},
    mem::swap,
    ops::{Add, Div, DivAssign, Mul, MulAssign, Neg, Sub},
};

use bitcode::{Decode, Encode};
use bitflags::bitflags;
use hashable::Hashable;
use itertools::Itertools;
use rustc_hash::FxBuildHasher;

use crate::{
    base_unit::{BaseUnit, Dimension, is_dim_eq, simplify_dimensionality, sqrt_dimensionality},
    converter::Converter,
    error::{SmootError, SmootResult},
    hash::Hash,
    parser::expression_parser,
    registry::Registry,
    utils::ApproxEq,
};

pub struct Dimensionality(pub HashMap<String, i32, FxBuildHasher>);

bitflags! {
    #[derive(PartialEq)]
    pub struct UnitFormat : u8 {
        const Default = 0;
        const Compact = 1;
        const WithoutSpaces = 1 << 1;
        const WithScalingFactor = 1 << 2;
    }
}

#[derive(Encode, Decode, Hashable, Clone, Debug, PartialEq)]
pub struct Unit {
    /// e.g. `meter` in `meter / second`.
    pub(crate) numerator_units: Vec<BaseUnit>,

    /// Reciprocal base units for this unit e.g. `second` in `meter / second`.
    pub(crate) denominator_units: Vec<BaseUnit>,

    /// The active dimensions for this unit, derived from the numerator and denominator base units.
    dimensionality: Vec<Dimension>,
}

impl Unit {
    pub fn new(numerator_units: Vec<BaseUnit>, denominator_units: Vec<BaseUnit>) -> Self {
        let mut unit = Self {
            numerator_units,
            denominator_units,
            dimensionality: Vec::new(),
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
    pub fn powi(&self, p: i32) -> Self {
        if self.is_dimensionless() {
            return Self::new_dimensionless();
        }
        match p {
            0 => Self::new_dimensionless(),
            1 => self.clone(),
            -1 => Self::new(self.denominator_units.clone(), self.numerator_units.clone()),
            _ => {
                let mut new = self.clone();
                new.ipowi(p);
                new
            }
        }
    }

    /// In-place integer power operation.
    pub fn ipowi(&mut self, p: i32) {
        if self.is_dimensionless() {
            return;
        }
        match p {
            0 => *self = Self::new_dimensionless(),
            1 => {}
            -1 => {
                swap(&mut self.numerator_units, &mut self.denominator_units);
                self.dimensionality.iter_mut().for_each(|d| *d = d.neg());
            }
            _ => {
                self.numerator_units.iter_mut().for_each(|u| u.ipowi(p));
                self.denominator_units.iter_mut().for_each(|u| u.ipowi(p));
                self.update_dimensionality();
            }
        }
    }

    pub fn isqrt(&mut self) -> SmootResult<()> {
        for u in self
            .numerator_units
            .iter_mut()
            .chain(self.denominator_units.iter_mut())
        {
            u.isqrt()?;
        }
        self.dimensionality = sqrt_dimensionality(&self.dimensionality).map_err(|_| {
            SmootError::InvalidOperation(format!(
                "sqrt would result in a non-integral power for {}",
                self,
            ))
        })?;
        Ok(())
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
        self.dimensionality.is_empty()
    }

    /// Return true if this unit is compatible with the target (e.g. meter and kilometer).
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        is_dim_eq(&self.dimensionality, &other.dimensionality)
    }

    /// Return true if values of this unit can be converted into the target unit.
    pub fn are_converters_compatible(&self, other: &Self) -> bool {
        self.numerator_units
            .iter()
            .map(|u| &u.converter)
            .eq(other.numerator_units.iter().map(|u| &u.converter))
            && self
                .denominator_units
                .iter()
                .map(|u| &u.converter)
                .eq(other.denominator_units.iter().map(|u| &u.converter))
    }

    pub fn is_offset(&self) -> bool {
        self.numerator_units
            .iter()
            .chain(self.denominator_units.iter())
            .any(|u| u.converter == Converter::Offset && !u.offset.approx_eq(0.0))
    }

    /// Convert all offset BaseUnits into delta units.
    pub fn into_delta(self) -> Self {
        let numerator = self
            .numerator_units
            .into_iter()
            .map(|u| u.into_delta())
            .collect();
        let denominator = self
            .denominator_units
            .into_iter()
            .map(|u| u.into_delta())
            .collect();
        Self {
            numerator_units: numerator,
            denominator_units: denominator,
            dimensionality: self.dimensionality,
        }
    }

    pub fn get_dimensionality(&self, registry: &Registry) -> Option<Dimensionality> {
        if self.is_dimensionless() {
            return None;
        }

        let mut dims = HashMap::default();
        dims.reserve(self.dimensionality.len());

        self.dimensionality.iter().copied().for_each(|d| {
            let dim_str = registry.get_dimension_string(d);
            dims.entry(dim_str.clone())
                .and_modify(|e| *e += d.signum() as i32)
                .or_insert(d.signum() as i32);
        });

        Some(Dimensionality(dims))
    }

    pub fn get_dimensionality_str(&self, registry: &Registry) -> Option<String> {
        self.get_dimensionality(registry).map(|dims| {
            dims.0
                .iter()
                .sorted_by_key(|(k, _)| k.as_str())
                .map(|(k, v)| format!("{}: {}", k, v))
                .join(", ")
        })
    }

    /// Convert this unit into a displayable string representation.
    ///
    /// Parameters
    /// ----------
    /// with_scaling_factor
    ///     If true, returns a string like `1 / meter` instead of `/ meter`.
    pub fn get_units_string(
        &self,
        registry: Option<&Registry>,
        format: UnitFormat,
    ) -> Option<String> {
        if self.numerator_units.is_empty() && self.denominator_units.is_empty() {
            return None;
        }

        // Return the name of this unit, respecting formatting options.
        let f_get_name = |name: &String| -> String {
            if format == UnitFormat::Default {
                return name.clone();
            }
            if format.intersects(UnitFormat::Compact) {
                registry
                    .and_then(|r| r.get_unit_symbol(name))
                    .unwrap_or(name)
                    .clone()
            } else {
                name.clone()
            }
        };
        let f_fmt_result = |result: String| -> String {
            if format.intersects(UnitFormat::WithoutSpaces) {
                result.replace(" ", "")
            } else {
                result
            }
        };

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

        let mut numerator = nums
            .iter()
            .map(|u| {
                if u.power == 1 {
                    f_get_name(&u.name).clone()
                } else {
                    format!("{} ** {}", f_get_name(&u.name), u.power)
                }
            })
            .join(" * ");

        if denoms.is_empty() {
            return Some(f_fmt_result(numerator));
        }

        let mut denominator = denoms
            .iter()
            .map(|u| {
                if u.power == 1 {
                    f_get_name(&u.name).clone()
                } else {
                    format!("{} ** {}", f_get_name(&u.name), u.power)
                }
            })
            .join(" * ");

        if nums.len() > 1 && !denoms.is_empty() {
            numerator = format!("({})", numerator);
        }
        if denoms.len() > 1 {
            denominator = format!("({})", denominator);
        }

        let result = if numerator.is_empty() {
            if format.intersects(UnitFormat::WithScalingFactor) {
                Some(format!("1 / {}", denominator))
            } else {
                Some(format!("/ {}", denominator))
            }
        } else {
            Some(format!("{} / {}", numerator, denominator))
        };

        result.map(f_fmt_result)
    }

    /// Sync the dimensionality of this unit with its numerator and denominator base units.
    ///
    /// This is invoked automatically during unit reduction / simplification.
    fn update_dimensionality(&mut self) {
        self.dimensionality.clear();

        self.dimensionality.extend(
            self.numerator_units
                .iter()
                .flat_map(|u| u.dimensionality.iter()),
        );
        self.dimensionality.extend(
            self.denominator_units
                .iter()
                .flat_map(|u| u.dimensionality.iter())
                .map(|d| -d),
        );

        self.dimensionality.sort_unstable();
        simplify_dimensionality(&mut self.dimensionality);
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
                    if last.get_dimension_type() == next.get_dimension_type() {
                        // Aggregate conversion factors
                        // To handle cases like m * km^2, we need to convert m -> km. That means we need the conversion
                        // factor in terms of last's exponent, not next's exponent.
                        let factor = last.get_multiplier() / next.multiplier.powi(last.power);
                        if is_denom {
                            result_conversion_factor /= factor;
                        } else {
                            result_conversion_factor *= factor;
                        }

                        last.name = next.name;
                        last.multiplier = next.multiplier;
                        last.power += next.power;
                        last.dimensionality.extend(next.dimensionality);
                        last.dimensionality.sort_unstable();
                        last.simplify();
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
            (u1.get_dimension_type(), u1.power)
                .partial_cmp(&(u2.get_dimension_type(), u2.power))
                .unwrap()
        });
        self.denominator_units.sort_by(|u1, u2| {
            (u1.get_dimension_type(), u1.power)
                .partial_cmp(&(u2.get_dimension_type(), u2.power))
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
            let u1_type = u1.get_dimension_type();
            let u2_type = u2.get_dimension_type();

            // Skip if mismatched units.
            match u1_type.cmp(&u2_type) {
                Ordering::Less => {
                    inum += 1;
                    continue;
                }
                Ordering::Greater => {
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

            if u1_type == u2_type {
                let multiplicative_factor = u1.get_multiplier() / u2.get_multiplier();
                if no_reduction && !multiplicative_factor.approx_eq(1.0) {
                    // Make sure we don't reduce units with disparate scales in no_reduction mode.
                    inum += 1;
                    iden += 1;
                    continue;
                }

                // Now there must be a cancellation.
                result_conversion_factor *= multiplicative_factor;
            }

            // Unit exponents (e.g. meter ** 2) may result in a cancellation without completely removing
            // the unit from the numerator/denominator.
            match u1.power.cmp(&u2.power) {
                Ordering::Equal => {
                    numerator_retain[inum] = false;
                    denominator_retain[iden] = false;
                }
                Ordering::Less => {
                    numerator_retain[inum] = false;
                    u2.div_dimensionality(u1);
                }
                Ordering::Greater => {
                    denominator_retain[iden] = false;
                    u1.div_dimensionality(u2);
                }
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
        if base.dimensionality.is_empty() {
            return;
        }

        let mut f_update = |dim: Dimension, power: i32| {
            let root = registry.get_root_unit(dim).powi(power);
            if dim.is_negative() {
                denominator.push(root);
            } else {
                numerator.push(root);
            }
        };

        let mut last = base.dimensionality[0];
        let mut power = 1;

        for &dim in &base.dimensionality[1..] {
            if dim == last {
                power += 1;
                continue;
            }
            f_update(last, power);
            power = 1;
            last = dim;
        }
        // Handle the final segment
        f_update(last, power);
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
            self.get_units_string(None, UnitFormat::Default | UnitFormat::WithScalingFactor)
                .unwrap_or("dimensionless".to_string())
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
            return Err(SmootError::InvalidOperation(format!(
                "Invalid Unit operation '{}' + '{}'",
                self, rhs,
            )));
        }
        Ok(self)
    }
}

impl Sub for Unit {
    type Output = SmootResult<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        if !self.is_compatible_with(&rhs) {
            return Err(SmootError::InvalidOperation(format!(
                "Invalid Unit operation '{}' - '{}'",
                self, rhs,
            )));
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
    static UNIT_DEGC: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("degC").expect("No unit 'degC'"));
    static UNIT_KELVIN: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("kelvin").expect("No unit 'kelvin'"));
    static UNIT_DELTA_DEGC: LazyLock<&BaseUnit> = LazyLock::new(|| {
        TEST_REGISTRY
            .get_unit("delta_degC")
            .expect("No unit 'delta_degC'")
    });

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
            vec![UNIT_METER.powi(2)],
            vec![UNIT_SECOND.powi(2)],
        ),
        Some("meter ** 2 / second ** 2")
        ; "Powers"
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
        assert_eq!(
            u.get_units_string(None, UnitFormat::Default | UnitFormat::WithScalingFactor),
            expected.map(String::from)
        );
    }

    #[case(
        Unit::new(vec![UNIT_METER.powi(2)], vec![]),
        Some("m ** 2"),
        UnitFormat::Compact
        ; "Compact"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone()], vec![UNIT_SECOND.clone()]),
        Some("m / s"),
        UnitFormat::Compact
        ; "Numerator and denominator"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.powi(2)], vec![]),
        Some("m**2"),
        UnitFormat::Compact | UnitFormat::WithoutSpaces
        ; "Compact without spaces"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone()], vec![UNIT_SECOND.clone()]),
        Some("m/s"),
        UnitFormat::Compact | UnitFormat::WithoutSpaces
        ; "Compact without spaces numerator and denominator"
    )]
    fn test_get_compact_units_string(u: Unit, expected: Option<&str>, format: UnitFormat) {
        assert_eq!(
            u.get_units_string(Some(&TEST_REGISTRY), format),
            expected.map(String::from)
        )
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

    #[case(
        Unit::new_dimensionless(),
        true
        ; "Trivially dimensionless"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone()], vec![]),
        false
        ; "Trivially not dimensionless"
    )]
    fn test_is_dimensionless(u: Unit, expected: bool) {
        assert_eq!(u.is_dimensionless(), expected);
    }

    #[case(
        Unit::new(vec![UNIT_DEGC.clone()], vec![]),
        true
    )]
    #[case(
        Unit::new(vec![UNIT_KELVIN.clone()], vec![]),
        false
        ; "Root unit is not offset"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone()], vec![]),
        false
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone(), UNIT_DEGC.clone()], vec![]),
        true
        ; "Composite unit with offset base unit is offset"
    )]
    #[case(
        Unit::new(vec![], vec![UNIT_DEGC.clone()]),
        true
        ; "Offset denominator"
    )]
    #[case(
        Unit::new(vec![UNIT_DELTA_DEGC.clone()], vec![]),
        false
        ; "Delta units are not offset"
    )]
    fn test_is_offset(u: Unit, expected: bool) {
        assert_eq!(u.is_offset(), expected);
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
                .sorted_by_key(|u| u.get_dimension_type())
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
                .sorted_by_key(|u| u.get_dimension_type())
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

        assert_eq!(u.dimensionality, UNIT_SECOND.dimensionality);
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
        let mut u = Unit::new(vec![UNIT_METER.powi(2)], vec![UNIT_METER.clone()]);
        let expected = Unit::new(vec![UNIT_METER.clone()], vec![]);

        let _ = u.simplify(true);

        assert_eq!(u, expected, "{:#?} != {:#?}", u, expected);
    }

    /// Denominator units with exponents are correctly simplified.
    #[test]
    fn test_simplify_adjusts_denominator_powers() {
        let mut u = Unit::new(vec![UNIT_METER.clone()], vec![UNIT_METER.powi(2)]);
        let expected = Unit::new(vec![], vec![UNIT_METER.clone()]);

        let _ = u.simplify(true);

        assert_eq!(u, expected, "{:#?} != {:#?}", u, expected);
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

    #[case(
        Unit::new(vec![UNIT_KILOMETER.clone(), UNIT_METER.clone()], vec![]),
        Unit::new(vec![UNIT_METER.powi(2)], vec![]),
        1000.0
        ; "Numerator"
    )]
    #[case(
        Unit::new(vec![], vec![UNIT_KILOMETER.clone(), UNIT_METER.clone()]),
        Unit::new(vec![], vec![UNIT_METER.powi(2)]),
        1e-3
        ; "Denominator"
    )]
    #[case(
        Unit::new(vec![UNIT_METER.clone(), UNIT_KILOMETER.clone()], vec![]),
        Unit::new(vec![UNIT_KILOMETER.powi(2)], vec![]),
        1e-3
        ; "Numerator reversed"
    )]
    #[case(
        Unit::new(vec![UNIT_JOULE.clone()], vec![UNIT_NEWTON.clone()]),
        Unit::new(vec![UNIT_JOULE.clone()], vec![UNIT_NEWTON.clone()]),
        1.0
        ; "Incompatible units with matching dimensions do not reduce"
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
        assert_eq!(unit, expected, "{:#?} != {:#?}", unit, expected);
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
        Unit::new(vec![UNIT_METER.clone(), UNIT_GRAM.clone()], vec![UNIT_SECOND.powi(2)]),
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
        assert_eq!(unit, expected, "{:#?} != {:#?}", unit, expected);
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
