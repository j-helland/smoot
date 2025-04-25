use std::ops::{Div, DivAssign, Mul, MulAssign};

use crate::hash::Hash;
use bitcode::{Decode, Encode};
use hashable::Hashable;

use crate::{
    error::{SmootError, SmootResult},
    utils::ApproxEq,
};

pub type DimensionType = u64;
pub const DIMENSIONLESS_TYPE: DimensionType = 0;

#[derive(Encode, Decode, Hashable, Clone, Debug, PartialEq)]
pub struct BaseUnit {
    pub name: String,
    pub multiplier: f64,
    pub power: Option<f64>,
    pub unit_type: DimensionType,
    pub dimensionality: Vec<f64>,
}

/// Compute the dimensionality vector for a given dimension type.
/// e.g. 0x1 -> [1.0]
fn get_dimensionality(mut unit_type: DimensionType) -> Vec<f64> {
    let size = (DimensionType::BITS - unit_type.leading_zeros()) as usize;
    let mut dimensionality = vec![0.0; size];
    while unit_type > 0 {
        let idx = unit_type.trailing_zeros() as usize;
        dimensionality[idx] = 1.0;
        unit_type &= !(1 << idx);
    }
    dimensionality
}

impl BaseUnit {
    pub fn new(name: String, multiplier: f64, unit_type: DimensionType) -> Self {
        Self {
            name,
            multiplier,
            power: None,
            unit_type,
            dimensionality: get_dimensionality(unit_type),
        }
    }

    /// Create a new, dimensionless base unit. These are used to store constants like pi.
    pub fn new_constant(multiplier: f64) -> Self {
        Self {
            name: String::new(),
            multiplier,
            power: None,
            unit_type: DIMENSIONLESS_TYPE,
            dimensionality: vec![],
        }
    }

    /// In-place multiplication of this unit's dimensionality vector by a number.
    pub fn mul_dimensionality(&mut self, n: f64) {
        self.dimensionality
            .iter_mut()
            // Only apply operations to dimensions that are present
            .filter(|d| !d.approx_eq(0.0))
            .for_each(|d| *d *= n);
    }

    pub fn sub_dimensionality(&mut self, n: f64) {
        self.dimensionality
            .iter_mut()
            // Only apply operations to dimensions that are present
            .filter(|d| !d.approx_eq(0.0))
            .for_each(|d| *d -= n);
    }

    /// Return true if this unit is a composite of multiple dimensions.
    pub fn is_multidimensional(&self) -> bool {
        (self.unit_type >> self.unit_type.trailing_zeros()) > 1
    }

    /// Return true if this unit is a constant value e.g. `1`.
    /// Constant values can result from parsing unit expressions like `2 / meter`.
    pub fn is_constant(&self) -> bool {
        self.name.is_empty()
    }

    /// Get the multiplicative factor associated with this base unit.
    pub fn get_multiplier(&self) -> f64 {
        self.power
            .map(|p| self.multiplier.powf(p))
            .unwrap_or(self.multiplier)
    }

    /// Get the multiplicative factor needed to convert this unit into a target unit.
    ///
    /// Return
    /// ------
    /// Err if the units are incompatible (e.g. meter and gram).
    pub fn conversion_factor(&self, target: &Self) -> SmootResult<f64> {
        // Fast check with unit types, slower vector equality check for more detail.
        if self.unit_type != target.unit_type
            || !self.dimensionality.approx_eq(&target.dimensionality)
        {
            return Err(SmootError::IncompatibleUnitTypes(
                self.name.clone(),
                target.name.clone(),
            ));
        }

        // convert to the base unit, then to the target unit
        Ok(self.conversion_factor_unchecked(target))
    }

    /// Return the multiplicative factor needed to convert this unit into the target unit,
    /// assuming that both units have compatible dimensionality.
    #[inline(always)]
    pub fn conversion_factor_unchecked(&self, target: &Self) -> f64 {
        self.get_multiplier() / target.get_multiplier()
    }

    /// In-place power with a floating point exponent.
    pub fn ipowf(&mut self, n: f64) {
        self.power = self.power.or(Some(1.0)).map(|p| p * n);
        self.mul_dimensionality(n);
    }

    /// Return a new BaseUnit raised to a floating point power.
    pub fn powf(&self, n: f64) -> Self {
        let mut new = self.clone();
        new.ipowf(n);
        new
    }

    pub fn sub_power(&mut self, p: f64) {
        let new_power = self.power.unwrap_or(1.0) - p;
        if new_power.approx_eq(1.0) {
            self.power = None;
        } else {
            self.power = Some(new_power);
        }
        self.sub_dimensionality(p);
    }
}

//==================================================
// Arithmetic operators
//==================================================
impl Mul for BaseUnit {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut new = self.clone();
        new *= rhs;
        new
    }
}

impl MulAssign for BaseUnit {
    fn mul_assign(&mut self, rhs: Self) {
        self.multiplier *= rhs.get_multiplier();
        self.unit_type |= rhs.unit_type;

        self.dimensionality.extend(
            (0..rhs
                .dimensionality
                .len()
                .saturating_sub(self.dimensionality.len()))
                .map(|_| 0.0),
        );
        for i in 0..self.dimensionality.len().min(rhs.dimensionality.len()) {
            if !self.dimensionality[i].approx_eq(0.0) || !rhs.dimensionality[i].approx_eq(0.0) {
                self.dimensionality[i] += rhs.dimensionality[i];
            }
        }
    }
}

impl Div for BaseUnit {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut new = self.clone();
        new /= rhs;
        new
    }
}

impl DivAssign for BaseUnit {
    fn div_assign(&mut self, rhs: Self) {
        self.multiplier /= rhs.get_multiplier();
        self.unit_type |= rhs.unit_type;

        self.dimensionality.extend(
            (0..rhs
                .dimensionality
                .len()
                .saturating_sub(self.dimensionality.len()))
                .map(|_| 0.0),
        );
        for i in 0..self.dimensionality.len().min(rhs.dimensionality.len()) {
            if !self.dimensionality[i].approx_eq(0.0) || !rhs.dimensionality[i].approx_eq(0.0) {
                self.dimensionality[i] -= rhs.dimensionality[i];
            }
        }
    }
}

//==================================================
// Unit tests
//==================================================
#[cfg(test)]
mod test_base_unit {
    use std::hash::{DefaultHasher, Hasher};

    use test_case::case;

    use crate::assert_is_close;

    use super::*;

    #[case(
        BaseUnit::new("left".into(), 2.0, 1),
        BaseUnit::new("right".into(), 3.0, 1 << 1),
        BaseUnit::new("left".into(), 6.0, 1 | (1 << 1))
        ; "Multipliers multiply and unit types combine"
    )]
    fn test_mul(left: BaseUnit, right: BaseUnit, expected: BaseUnit) {
        assert_eq!(left * right, expected);
    }

    #[case(
        BaseUnit::new("test".into(), 1.0, 1),
        false
        ; "single dimension 1"
    )]
    #[case(
        BaseUnit::new("test".into(), 1.0, 1 << 8),
        false
        ; "single dimension 2"
    )]
    #[case(
        BaseUnit::new("test".into(), 1.0, (1 << 3) | (1 << 8)),
        true
        ; "multidimensional"
    )]
    fn test_is_multidimensional(unit: BaseUnit, expected: bool) {
        assert_eq!(unit.is_multidimensional(), expected);
    }

    #[test]
    /// The conversion factor between compatible units is computed correctly.
    fn test_conversion_factor() -> SmootResult<()> {
        // Given two units with the same type
        let u1 = BaseUnit::new("u1".into(), 1.0, 0);
        let u2 = BaseUnit::new("u2".into(), 2.0, 0);

        // Then a self conversion factor is 1.0
        assert_eq!(u1.conversion_factor(&u1)?, 1.0);

        // The conversion factor and reciprocal match.
        assert_is_close!(u1.conversion_factor(&u2)?, 0.5);
        assert_is_close!(u2.conversion_factor(&u1)?, 2.0);

        Ok(())
    }

    #[test]
    /// Trying to convert between incompatible units is an error.
    fn test_conversion_factor_incompatible_types() {
        // Given two units with disparate types
        let u1 = BaseUnit::new("u1".into(), 1.0, 0);
        let u2 = BaseUnit::new("u2".into(), 2.0, 1);

        // Then the result is an error
        let result = u1.conversion_factor(&u2);
        assert!(result.is_err());

        // commutative
        let result = u2.conversion_factor(&u1);
        assert!(result.is_err());
    }

    #[test]
    fn test_conversion_factor_unequal_dimensionality() {
        let u1 = BaseUnit {
            name: "u1".into(),
            multiplier: 1.0,
            power: None,
            unit_type: 0,
            dimensionality: vec![2.0, -2.0, 1.0],
        };
        let u2 = BaseUnit {
            name: "u2".into(),
            multiplier: 1.0,
            power: None,
            unit_type: 0,
            dimensionality: vec![1.0, -2.0, 1.0],
        };

        let result = u1.conversion_factor(&u2);
        assert!(result.is_err());

        // commutative
        let result = u2.conversion_factor(&u1);
        assert!(result.is_err());
    }

    #[test]
    fn test_hash() {
        let u1 = BaseUnit::new("u1".into(), 1.0, 0);
        assert_eq!(hash(&u1), hash(&u1.clone()));

        let u2 = BaseUnit::new("u2".into(), 2.0, 1);
        assert_ne!(hash(&u1), hash(&u2));
    }

    fn hash<T: Hash>(val: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        val.hash(&mut hasher);
        hasher.finish()
    }
}
