use std::{
    fmt::{self},
    mem::swap,
    ops::{Add, Div, DivAssign, Mul, MulAssign, Sub},
};

use bitcode::{Decode, Encode};
use itertools::{EitherOrBoth, Itertools};
use num_traits::ToPrimitive;

use crate::{
    base_unit::{BaseUnit, DimensionType, DIMENSIONLESS},
    error::{SmootError, SmootResult},
    parser::expression_parser,
    registry::Registry,
    types::Number,
    utils::float_eq_rel,
};

type UnitDimensionality<N> = Vec<N>;

#[derive(Encode, Decode, Clone, Debug, PartialEq)]
pub struct Unit<N: Number> {
    pub numerator_units: Vec<BaseUnit<N>>,
    pub numerator_dimension: DimensionType,
    pub numerator_dimensionality: UnitDimensionality<f64>,

    pub denominator_units: Vec<BaseUnit<N>>,
    pub denominator_dimension: DimensionType,
    pub denominator_dimensionality: UnitDimensionality<f64>,
}

impl<N: Number> Unit<N> {
    pub fn new(numerator_units: Vec<BaseUnit<N>>, denominator_units: Vec<BaseUnit<N>>) -> Self {
        let mut unit = Self {
            numerator_units,
            numerator_dimension: DIMENSIONLESS,
            numerator_dimensionality: UnitDimensionality::new(),
            denominator_units,
            denominator_dimension: DIMENSIONLESS,
            denominator_dimensionality: UnitDimensionality::new(),
        };
        unit.update_dimensionality();
        unit
    }

    /// Integer power operation that creates a new unit.
    pub fn powi(&self, n: i32) -> Self {
        if self.denominator_dimension == DIMENSIONLESS && self.numerator_dimension == DIMENSIONLESS
        {
            return Self::new(vec![], vec![]);
        }
        match n {
            0 => Self::new(vec![], vec![]),
            1 => self.clone(),
            -1 => Self::new(self.denominator_units.clone(), self.numerator_units.clone()),
            _ => self.powf(n.into()),
        }
    }

    /// In-place integer power operation.
    pub fn ipowi(&mut self, n: i32) {
        if self.denominator_dimension == DIMENSIONLESS && self.numerator_dimension == DIMENSIONLESS
        {
            return;
        }
        match n {
            0 => *self = Self::new(vec![], vec![]),
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
                u.mul_dimensionality(nabs);
                u.power = u.power.or(Some(1.0)).map(|p| p * nabs);
                u
            })
            .collect();
        let mut denominator = self
            .denominator_units
            .clone()
            .into_iter()
            .map(|mut u| {
                u.mul_dimensionality(nabs);
                u.power = u.power.or(Some(1.0)).map(|p| p * nabs);
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
            u.mul_dimensionality(nabs);
            u.power = u.power.or(Some(1.0)).map(|p| p * nabs);
        });
        self.denominator_units.iter_mut().for_each(|u| {
            u.mul_dimensionality(nabs);
            u.power = u.power.or(Some(1.0)).map(|p| p * nabs);
        });

        // Flip if power sign is negative.
        if n < 0.0 {
            swap(&mut self.numerator_units, &mut self.denominator_units);
        }
    }

    pub fn is_dimensionless(&self) -> bool {
        self.numerator_dimension == DIMENSIONLESS && self.denominator_dimension == DIMENSIONLESS
    }

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

    pub fn conversion_factor(&self, other: &Self) -> SmootResult<N> {
        if !self.is_compatible_with(other) {
            return Err(SmootError::IncompatibleUnitTypes(
                self.get_units_string().unwrap_or("dimensionless".into()),
                other.get_units_string().unwrap_or("dimensionless".into()),
            ));
        }

        let mut numerator_conversion_factor = N::one();
        for (from, to) in self
            .numerator_units
            .iter()
            .zip(other.numerator_units.iter())
        {
            numerator_conversion_factor *= from.conversion_factor(to)?;
        }

        let mut denominator_conversion_factor = N::one();
        for (from, to) in self
            .denominator_units
            .iter()
            .zip(other.denominator_units.iter())
        {
            denominator_conversion_factor *= from.conversion_factor(to)?;
        }

        Ok(numerator_conversion_factor / denominator_conversion_factor)
    }

    /// Format a power as a string.
    /// If the power is close to an integer, only display the integer part.
    fn format_power(p: f64) -> String {
        if float_eq_rel(p.fract(), 0.0, 1e-8) {
            format!("{}", p.to_i64().unwrap())
        } else {
            format!("{:?}", p)
        }
    }

    pub fn get_units_string(&self) -> Option<String> {
        let nums = self
            .numerator_units
            .iter()
            .filter(|u| u.unit_type != DIMENSIONLESS)
            .sorted_by_key(|u| u.name.as_str())
            .collect_vec();
        let denoms = self
            .denominator_units
            .iter()
            .filter(|u| u.unit_type != DIMENSIONLESS)
            .sorted_by_key(|u| u.name.as_str())
            .collect_vec();

        if nums.is_empty() && denoms.is_empty() {
            return None;
        }

        let mut numerator = nums
            .iter()
            .map(|u| {
                u.power
                    .map(Self::format_power)
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
                    .map(Self::format_power)
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

        Some(format!("{} / {}", numerator, denominator))
    }

    fn update_dimensionality(&mut self) {
        self.numerator_dimension = Self::get_dimension_mask(&self.numerator_units);
        Self::get_dimensionality(&mut self.numerator_dimensionality, &self.numerator_units);

        self.denominator_dimension = Self::get_dimension_mask(&self.denominator_units);
        Self::get_dimensionality(
            &mut self.denominator_dimensionality,
            &self.denominator_units,
        );
    }

    pub fn reduce(&mut self) -> N {
        let mut result_conversion_factor = self.simplify(false);

        let mut reduce_ = |units: &mut Vec<BaseUnit<N>>| {
            let mut units_reduced: Vec<BaseUnit<N>> = Vec::new();
            units.drain(..).for_each(|u| {
                if let Some(last) = units_reduced.last_mut() {
                    if let Ok(factor) = last.conversion_factor(&u) {
                        // Aggregate conversion factors
                        result_conversion_factor *= factor;

                        last.name = u.name;
                        last.multiplier = u.multiplier;
                        // Powers must update.
                        last.power = last.power.map(|p| p + u.power.unwrap_or(1.0)).or(Some(2.0));
                        last.dimensionality = last
                            .dimensionality
                            .drain(..)
                            .zip_longest(u.dimensionality.iter())
                            .map(|lr| match lr {
                                EitherOrBoth::Both(l, &r) => l + r,
                                EitherOrBoth::Left(l) => l,
                                EitherOrBoth::Right(r) => *r,
                            })
                            .collect();
                        return;
                    }
                }
                units_reduced.push(u);
            });
            units.extend(units_reduced);
        };

        reduce_(&mut self.numerator_units);
        reduce_(&mut self.denominator_units);
        self.update_dimensionality();

        result_conversion_factor
    }

    pub fn simplify(&mut self, no_reduction: bool) -> N {
        let mut result_conversion_factor = N::one();

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

        let mut numerator_retain = vec![true; self.numerator_units.len()];
        let mut denominator_retain = vec![true; self.denominator_units.len()];

        let mut inum = 0;
        let mut iden = 0;
        while inum < self.numerator_units.len() && iden < self.denominator_units.len() {
            let u1 = &mut self.numerator_units[inum];
            let u2 = &mut self.denominator_units[iden];

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

            if let Ok(factor) = u1.conversion_factor(u2) {
                if no_reduction && !factor.approx_eq(N::one()) {
                    // Make sure we don't reduce units with disparate scales.
                    inum += 1;
                    iden += 1;
                    continue;
                }
                result_conversion_factor *= factor;
            }

            let u1_power = u1.power.unwrap_or(1.0);
            let u2_power = u2.power.unwrap_or(1.0);
            if u1_power == u2_power {
                numerator_retain[inum] = false;
                denominator_retain[iden] = false;
            } else if u1_power < u2_power {
                numerator_retain[inum] = false;
                u2.ipowf(u2_power - u1_power);
            } else if u1_power > u2_power {
                denominator_retain[iden] = false;
                u1.ipowf(u1_power - u2_power);
            }

            inum += 1;
            iden += 1;
        }

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

        // Make sure that dimensionality is synced.
        self.update_dimensionality();

        result_conversion_factor
    }

    fn get_dimension_mask(units: &[BaseUnit<N>]) -> DimensionType {
        units.iter().fold(0, |d, u| d | u.unit_type)
    }

    fn get_dimensionality(dimensionality: &mut UnitDimensionality<f64>, units: &[BaseUnit<N>]) {
        dimensionality.fill(0.0);
        dimensionality.resize(
            units
                .iter()
                .map(|u| u.dimensionality.len())
                .max()
                .unwrap_or(0),
            0.0,
        );
        units
            .iter()
            .flat_map(|u| u.dimensionality.iter().enumerate())
            .for_each(|(i, d)| {
                dimensionality[i] += *d;
            });
    }
}

impl Unit<f64> {
    pub fn parse(registry: &Registry, s: &str) -> SmootResult<Self> {
        expression_parser::unit_expression(s, registry)
            .map(|mut u| {
                u.reduce();
                u
            })
            .map_err(|_| SmootError::InvalidUnitExpression(0, s.into()))
    }
}

impl<N: Number> fmt::Display for Unit<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.get_units_string().unwrap_or("dimensionless".into())
        )
    }
}

impl<N: Number> Mul for Unit<N> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.numerator_units.extend_from_slice(&rhs.numerator_units);
        self.denominator_units
            .extend_from_slice(&rhs.denominator_units);
        let _ = self.simplify(true);
        self
    }
}

impl<N: Number> MulAssign<&Unit<N>> for Unit<N> {
    fn mul_assign(&mut self, rhs: &Unit<N>) {
        self.numerator_units.extend_from_slice(&rhs.numerator_units);
        self.denominator_units
            .extend_from_slice(&rhs.denominator_units);
        let _ = self.simplify(true);
    }
}

impl<N: Number> Div for Unit<N> {
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

impl<N: Number> DivAssign<&Unit<N>> for Unit<N> {
    fn div_assign(&mut self, rhs: &Unit<N>) {
        self.numerator_units
            .extend_from_slice(&rhs.denominator_units);
        self.denominator_units
            .extend_from_slice(&rhs.numerator_units);
        let _ = self.simplify(true);
    }
}

impl<N: Number> Add for Unit<N> {
    type Output = SmootResult<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        if !self.is_compatible_with(&rhs) {
            return Err(SmootError::InvalidOperation(
                "+",
                self.get_units_string().unwrap_or("dimensionless".into()),
                rhs.get_units_string().unwrap_or("dimensionless".into()),
            ));
        }
        Ok(self)
    }
}

impl<N: Number> Sub for Unit<N> {
    type Output = SmootResult<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        if !self.is_compatible_with(&rhs) {
            return Err(SmootError::InvalidOperation(
                "-",
                self.get_units_string().unwrap_or("dimensionless".into()),
                rhs.get_units_string().unwrap_or("dimensionless".into()),
            ));
        }
        Ok(self)
    }
}

impl From<Unit<f64>> for Unit<i64> {
    fn from(value: Unit<f64>) -> Self {
        Self {
            numerator_units: value
                .numerator_units
                .iter()
                .cloned()
                .map(|u| u.into())
                .collect(),
            numerator_dimension: value.numerator_dimension,
            numerator_dimensionality: value.numerator_dimensionality,
            denominator_units: value
                .denominator_units
                .iter()
                .cloned()
                .map(|u| u.into())
                .collect(),
            denominator_dimension: value.denominator_dimension,
            denominator_dimensionality: value.denominator_dimensionality,
        }
    }
}

#[cfg(test)]
mod test_unit {
    use super::*;
    use std::{
        f64,
        sync::{Arc, LazyLock},
    };
    use test_case::case;

    use crate::{assert_is_close, registry::REGISTRY};

    static UNIT_DIMENSIONLESS: LazyLock<&Arc<BaseUnit<f64>>> = LazyLock::new(|| {
        REGISTRY
            .get_unit("dimensionless")
            .expect("No unit 'dimensionless'")
    });
    static UNIT_METER: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("meter").expect("No unit 'meter'"));
    static UNIT_KILOMETER: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("kilometer").expect("No unit 'kilometer'"));
    static UNIT_SECOND: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("second").expect("No unit 'second'"));
    static UNIT_MINUTE: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("minute").expect("No unit 'minute'"));
    static UNIT_HOUR: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("hour").expect("No unit 'hour'"));
    static UNIT_GRAM: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("gram").expect("No unit 'gram'"));
    static UNIT_WATT: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("watt").expect("No unit 'watt'"));

    #[case(
        Unit::new(vec![], vec![]),
        Unit::new(vec![], vec![]),
        Some(1.0)
        ; "Trivial conversion factor"
    )]
    #[case(
        Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]),
        Unit::new(vec![BaseUnit::clone(&UNIT_MINUTE)], vec![]),
        Some(1.0 / 60.0)
        ; "Basic conversion factor"
    )]
    #[case(
        Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![BaseUnit::clone(&UNIT_SECOND)]),
        Unit::new(vec![BaseUnit::clone(&UNIT_KILOMETER)], vec![BaseUnit::clone(&UNIT_HOUR)]),
        Some(60.0 * 60.0 / 1000.0)
        ; "Composite conversion factor"
    )]
    #[case(
        Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]),
        Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]),
        None
        ; "Incompatible units"
    )]
    fn test_conversion_factor(
        u1: Unit<f64>,
        u2: Unit<f64>,
        expected: Option<f64>,
    ) -> SmootResult<()> {
        if let Some(expected) = expected {
            assert_is_close!(u1.conversion_factor(&u2)?, expected);
            assert_is_close!(u2.conversion_factor(&u1)?, 1.0 / expected);
        } else {
            assert!(u1.conversion_factor(&u2).is_err());
        }
        Ok(())
    }

    #[case(
        Unit::new(vec![], vec![]),
        None
        ; "Dimensionless"
    )]
    #[case(
        Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![BaseUnit::clone(&UNIT_SECOND)]),
        Some("meter / second")
    )]
    #[case(
        Unit::new(vec![BaseUnit::clone(&UNIT_METER), BaseUnit::clone(&UNIT_SECOND)], vec![]),
        Some("meter * second")
    )]
    #[case(
        Unit::new(
            vec![BaseUnit::clone(&UNIT_METER), BaseUnit::clone(&UNIT_WATT)],
            vec![BaseUnit::clone(&UNIT_SECOND), BaseUnit::clone(&UNIT_GRAM)],
        ),
        Some("(meter * watt) / (gram * second)")
        ; "Multiple units in the numerator and denominator are parenthesized and sorted"
    )]
    #[case(
        Unit::new(
            vec![BaseUnit::clone(&UNIT_METER), BaseUnit::clone(&UNIT_METER)],
            vec![BaseUnit::clone(&UNIT_SECOND), BaseUnit::clone(&UNIT_SECOND)],
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
    fn test_get_units_string(u: Unit<f64>, expected: Option<&str>) {
        assert_eq!(u.get_units_string(), expected.map(String::from));
    }

    #[test]
    fn test_mul_numerator() {
        let u1 = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);
        let u2 = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);

        let u = u1 * u2;

        assert_eq!(
            u.numerator_units,
            vec![BaseUnit::clone(&UNIT_METER), BaseUnit::clone(&UNIT_SECOND)]
                .into_iter()
                .sorted_by_key(|u| u.unit_type)
                .collect_vec()
        );
        assert!(u.denominator_units.is_empty());
    }

    #[test]
    fn test_mul_denominator() {
        let u1 = Unit::new(vec![], vec![BaseUnit::clone(&UNIT_SECOND)]);
        let u2 = Unit::new(vec![], vec![BaseUnit::clone(&UNIT_METER)]);

        let u = u1 * u2;

        assert!(u.numerator_units.is_empty());
        assert_eq!(
            u.denominator_units,
            vec![BaseUnit::clone(&UNIT_METER), BaseUnit::clone(&UNIT_SECOND)]
                .into_iter()
                .sorted_by_key(|u| u.unit_type)
                .collect_vec()
        );
    }

    #[test]
    /// Multiplication of units should trigger simplification.
    fn tet_mul_simplifies_units() {
        let u1 = Unit::new(
            vec![BaseUnit::clone(&UNIT_SECOND)],
            vec![BaseUnit::clone(&UNIT_METER)],
        );
        let u2 = Unit::new(
            vec![BaseUnit::clone(&UNIT_METER)],
            vec![BaseUnit::clone(&UNIT_SECOND)],
        );

        let u = u1 * u2;

        assert!(u.numerator_units.is_empty());
        assert!(u.denominator_units.is_empty());
    }

    #[test]
    fn test_div_numerator() {
        let u1 = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let u2 = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);

        let u = u1 / u2;

        assert_eq!(u.numerator_units, vec![BaseUnit::clone(&UNIT_METER)]);
        assert_eq!(u.denominator_units, vec![BaseUnit::clone(&UNIT_SECOND)]);
    }

    #[test]
    fn test_div_denominator() {
        let u1 = Unit::new(vec![], vec![BaseUnit::clone(&UNIT_METER)]);
        let u2 = Unit::new(vec![], vec![BaseUnit::clone(&UNIT_SECOND)]);

        let u = u1 / u2;

        assert_eq!(u.numerator_units, vec![BaseUnit::clone(&UNIT_SECOND)]);
        assert_eq!(u.denominator_units, vec![BaseUnit::clone(&UNIT_METER)]);
    }

    #[test]
    fn test_div_simplifies_units() {
        let u1 = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);
        let u2 = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);

        let u = u1 / u2;

        assert!(u.numerator_units.is_empty());
        assert!(u.denominator_units.is_empty());
    }

    #[test]
    /// Simplifying units must compute the correct conversion factor for the cancelled units.
    fn test_simplify_computes_conversion_factor() {
        let mut u = Unit::new(
            vec![BaseUnit::clone(&UNIT_SECOND), BaseUnit::clone(&UNIT_MINUTE)],
            vec![BaseUnit::clone(&UNIT_HOUR)],
        );

        let conversion_factor = u.simplify(false);

        assert_is_close!(conversion_factor, 1.0 / 60.0 / 60.0);
        assert_eq!(u.numerator_units, vec![BaseUnit::clone(&UNIT_MINUTE)]);
        assert!(u.denominator_units.is_empty());
    }

    #[test]
    /// Simpliyfing will cancel multiple units within the same unit type.
    fn test_simplify_cancels_multiple_units() {
        let mut u = Unit::new(
            vec![BaseUnit::clone(&UNIT_SECOND), BaseUnit::clone(&UNIT_SECOND)],
            vec![
                BaseUnit::clone(&UNIT_SECOND),
                BaseUnit::clone(&UNIT_SECOND),
                BaseUnit::clone(&UNIT_SECOND),
            ],
        );

        let _ = u.simplify(true);

        assert!(u.numerator_units.is_empty());
        assert_eq!(u.denominator_units, vec![BaseUnit::clone(&UNIT_SECOND)]);
    }

    #[test]
    /// Simplifying will cancel units across unit types.
    fn test_simplify_cancels_units_of_different_types() {
        let mut u = Unit::new(
            vec![BaseUnit::clone(&UNIT_SECOND), BaseUnit::clone(&UNIT_METER)],
            vec![BaseUnit::clone(&UNIT_SECOND), BaseUnit::clone(&UNIT_METER)],
        );

        let _ = u.simplify(false);

        assert!(u.numerator_units.is_empty());
        assert!(u.denominator_units.is_empty());
    }

    #[test]
    /// Simplifying units correctly updates the dimensionality of the numerator and denominator.
    fn test_simplify_updates_unit_dimensionality() {
        let mut u = Unit::new(
            vec![
                BaseUnit::clone(&UNIT_SECOND),
                BaseUnit::clone(&UNIT_SECOND),
                BaseUnit::clone(&UNIT_METER),
            ],
            vec![BaseUnit::clone(&UNIT_SECOND), BaseUnit::clone(&UNIT_METER)],
        );

        let _ = u.simplify(true);

        assert_eq!(u.denominator_dimension, DIMENSIONLESS);
        assert!(u.denominator_dimensionality.is_empty());
        assert_eq!(u.numerator_dimension, UNIT_SECOND.unit_type);
        assert_eq!(u.numerator_dimensionality, UNIT_SECOND.dimensionality);
    }

    #[test]
    fn test_simplify_with_mixed_ordering() {
        let mut u = Unit::new(
            vec![
                BaseUnit::clone(&UNIT_SECOND),
                BaseUnit::clone(&UNIT_METER),
                BaseUnit::clone(&UNIT_SECOND),
            ],
            vec![BaseUnit::clone(&UNIT_METER), BaseUnit::clone(&UNIT_SECOND)],
        );

        let _ = u.simplify(true);

        assert_eq!(u.numerator_units, vec![BaseUnit::clone(&UNIT_SECOND)]);
        assert!(u.denominator_units.is_empty());
    }

    #[test]
    fn test_simplify_with_fractional_powers() {
        let mut u = Unit::new(
            vec![UNIT_KILOMETER.powf(0.5)],
            vec![BaseUnit::clone(&UNIT_METER)],
        );

        let conversion_factor = u.simplify(false);

        println!("{:#?}", u);

        assert_is_close!(conversion_factor, 1000.0_f64.sqrt());
        assert!(u.numerator_units.is_empty());
        assert_eq!(u.denominator_units, vec![UNIT_METER.powf(0.5)]);
    }

    // #[test]
    #[case(
        Unit::new(vec![], vec![]),
        Unit::new(vec![], vec![]),
        true
        ; "Trivial"
    )]
    #[case(
        Unit::new(vec![BaseUnit::clone(&UNIT_DIMENSIONLESS)], vec![]),
        Unit::new(vec![BaseUnit::clone(&UNIT_DIMENSIONLESS)], vec![]),
        true
        ; "Basic compatibility"
    )]
    #[case(
        Unit::new(vec![BaseUnit::clone(&UNIT_DIMENSIONLESS), BaseUnit::clone(&UNIT_SECOND)], vec![BaseUnit::clone(&UNIT_METER)]),
        Unit::new(vec![BaseUnit::clone(&UNIT_DIMENSIONLESS), BaseUnit::clone(&UNIT_SECOND)], vec![BaseUnit::clone(&UNIT_METER)]),
        true
        ; "Complex compatibility"
    )]
    #[case(
        Unit::new(vec![BaseUnit::clone(&UNIT_DIMENSIONLESS)], vec![]),
        Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]),
        false
        ; "Basic incompatibility"
    )]
    #[case(
        Unit::new(vec![BaseUnit::clone(&UNIT_METER), BaseUnit::clone(&UNIT_SECOND)], vec![]),
        Unit::new(vec![BaseUnit::clone(&UNIT_SECOND), BaseUnit::clone(&UNIT_METER)], vec![]),
        true
        ; "Invariant to ordering"
    )]
    #[case(
        Unit::new(vec![], vec![BaseUnit::clone(&UNIT_METER)]),
        Unit::new(vec![], vec![BaseUnit::clone(&UNIT_METER)]),
        true
        ; "Basic denominator compatibility"
    )]
    #[case(
        Unit::new(vec![], vec![BaseUnit::clone(&UNIT_METER)]),
        Unit::new(vec![], vec![BaseUnit::clone(&UNIT_SECOND)]),
        false
        ; "Basic denominator incompatibility"
    )]
    #[case(
        Unit::new(vec![BaseUnit::clone(&UNIT_SECOND), BaseUnit::clone(&UNIT_SECOND)], vec![]),
        Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]),
        false
        ; "Dimensionality incompatibility"
    )]
    fn test_is_compatible_with(u1: Unit<f64>, u2: Unit<f64>, expected: bool) {
        assert_eq!(u1.is_compatible_with(&u2), expected);
    }

    #[test]
    fn test_is_compatible_with_incompatible_units() {
        let u1 = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let u2 = Unit::new(vec![], vec![]);
        assert!(!u1.is_compatible_with(&u2));
    }

    #[test]
    /// Dimensionless units should be self-compatible.
    fn test_is_compatible_with_dimensionless() {
        let u1 = Unit::new(vec![BaseUnit::clone(&UNIT_DIMENSIONLESS)], vec![]);
        // This is equivalent to a dimensionless unit.
        let u2 = Unit::new(vec![], vec![]);
        assert!(u1.is_compatible_with(&u2));
    }

    #[test]
    /// Floating point imprecision should not cause units to be considered incompatible.
    fn test_is_compatible_with_float_imprecision() {
        let u1 = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let mut u2 = u1.clone();
        u2.ipowf(1.0 + f64::EPSILON);
        assert!(u1.is_compatible_with(&u2));
    }

    #[test]
    /// Fractional powers should be checked for compatibility.
    fn test_is_compatible_fractional_powers() {
        let u1 = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let u2 = Unit::new(vec![UNIT_METER.powf(0.5)], vec![]);
        assert!(!u1.is_compatible_with(&u2));
    }
}
