use std::{
    cmp::Ordering,
    ops::{Div, DivAssign, Mul, MulAssign, Neg},
};

use crate::hash::Hash;
use bitcode::{Decode, Encode};
use hashable::Hashable;

use crate::error::{SmootError, SmootResult};

pub type DimensionType = u64;
pub const DIMENSIONLESS_TYPE: DimensionType = 0;

pub type Dimension = i8;

#[derive(Encode, Decode, Hashable, Clone, Debug, PartialEq)]
pub struct BaseUnit {
    pub name: String,
    pub multiplier: f64,
    pub unit_type: DimensionType,
    pub dimensionality: Vec<Dimension>,
    pub power: i32,
}

impl BaseUnit {
    pub fn new(name: String, multiplier: f64, unit_type: DimensionType) -> Self {
        Self {
            name,
            multiplier,
            unit_type,
            dimensionality: get_dimensionality(unit_type),
            power: 1,
        }
    }

    /// Create a new, dimensionless base unit. These are used to store constants like pi.
    pub fn new_constant(multiplier: f64) -> Self {
        Self {
            name: String::new(),
            multiplier,
            unit_type: DIMENSIONLESS_TYPE,
            dimensionality: vec![],
            power: 1,
        }
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
        self.multiplier.powi(self.power)
    }

    /// Get the multiplicative factor needed to convert this unit into a target unit.
    ///
    /// Return
    /// ------
    /// Err if the units are incompatible (e.g. meter and gram).
    pub fn conversion_factor(&self, target: &Self) -> SmootResult<f64> {
        // Fast check with unit types, slower vector equality check for more detail.
        if self.unit_type != target.unit_type || self.dimensionality != target.dimensionality {
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

    pub fn ipowi(&mut self, p: i32) {
        self.power *= p;
        if p == 0 {
            self.dimensionality.clear();
            self.unit_type = DIMENSIONLESS_TYPE;
            return;
        }
        if p == 1 {
            return;
        }

        let ncopies = p.unsigned_abs().saturating_sub(1) as usize;
        let copies = self.dimensionality.repeat(ncopies);
        self.dimensionality.extend(copies);
        if p.is_negative() {
            self.dimensionality.iter_mut().for_each(|d| *d = d.neg());
        }
        self.dimensionality.sort();
    }

    pub fn powi(&self, p: i32) -> Self {
        let mut new = self.clone();
        new.ipowi(p);
        new
    }

    pub fn isqrt(&mut self) -> SmootResult<()> {
        if self.dimensionality.is_empty() {
            return Ok(());
        }
        if self.dimensionality.len() % 2 != 0 {
            return Err(SmootError::InvalidOperation(format!(
                "Invalid operation on BaseUnit: sqrt would result in a non-integral dimensionality for {}",
                self.name,
            )));
        }

        let mut result = Vec::with_capacity(self.dimensionality.len() / 2);
        let mut last = self.dimensionality[0];
        let mut count = 1;

        for &num in &self.dimensionality[1..] {
            if num == last {
                count += 1;
                continue;
            }

            let new_count = count / 2;
            result.extend(std::iter::repeat_n(last, new_count));

            last = num;
            count = 1;
        }

        // Handle the last group.
        let new_count = count / 2;
        result.extend(std::iter::repeat_n(last, new_count));

        // There should be no affect on self.unit_type since powers can only multiply/divide existing dimensionality numbers.
        self.dimensionality = result;
        self.power /= 2;
        Ok(())
    }

    pub fn sqrt(&self) -> SmootResult<Self> {
        let mut new = self.clone();
        new.isqrt()?;
        Ok(new)
    }

    pub fn simplify(&mut self) {
        simplify_dimensionality(&mut self.dimensionality);
        self.unit_type = dimensionality_to_type(&self.dimensionality);
    }

    pub fn div_dimensionality(&mut self, other: &Self) {
        self.power -= other.power;
        if self.power == 0 {
            self.dimensionality.clear();
            self.unit_type = DIMENSIONLESS_TYPE;
            return;
        }

        self.dimensionality
            .extend(other.dimensionality.iter().map(|d| -d));
        self.dimensionality.sort();
        self.simplify();
    }
}

/// Compute the dimensionality vector for a given dimension type.
fn get_dimensionality(mut unit_type: DimensionType) -> Vec<Dimension> {
    let size = (DimensionType::BITS - unit_type.leading_zeros()) as usize;
    let mut dims = Vec::with_capacity(size);
    let mut idx = 0;
    while unit_type > 0 {
        if unit_type & 0x1 == 0x1 {
            dims.push(idx + 1); // `0` is not allowed, we need sign bit to determine numerator or denominator
            idx += 1;
            unit_type >>= 1;
        } else {
            // Jump to next highest set bit
            let offset = unit_type.trailing_zeros() as Dimension;
            idx += offset;
            unit_type >>= offset;
        }
    }
    // dims is sorted by construction
    dims
}

/// Cancel any opposite dimensions in-place.
/// e.g. `[-2, -1, -1, 1, 2, 2] -> [-1, 2]`.
///
/// Capacity is retained.
pub(crate) fn simplify_dimensionality(dimensionality: &mut Vec<Dimension>) {
    let len = dimensionality.len();
    if len < 2 {
        return;
    }

    let mut should_keep = vec![true; len];

    let mut left = 0;
    let mut right = len - 1;
    while left < right {
        let sum = dimensionality[left] as i16 + dimensionality[right] as i16;
        match sum.cmp(&0) {
            Ordering::Equal => {
                should_keep[left] = false;
                should_keep[right] = false;
                left += 1;
                right -= 1;
            }
            // Negative is too large, move right
            Ordering::Less => left += 1,
            // Positive is too large, move left
            Ordering::Greater => right -= 1,
        }
    }

    // Drop cancelled elements
    let mut idx = 0;
    dimensionality.retain(|_| {
        idx += 1;
        should_keep[idx - 1]
    });
}

/// Convert a dimensionality vector into a dimension bitmask
/// e.g. `[1, 3, 4] -> 0b1101` (note little endian).
fn dimensionality_to_type(dimensionality: &[Dimension]) -> DimensionType {
    if dimensionality.is_empty() {
        return DIMENSIONLESS_TYPE;
    }

    let mut last = dimensionality[0];
    let mut result = 1 << (last.abs() - 1); // `0` is not a possible value
    for &val in &dimensionality[1..] {
        if val == last {
            continue;
        }
        result |= 1 << (val.abs() - 1); // `0` is not a possible value
        last = val;
    }

    result
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

        self.dimensionality.extend(rhs.dimensionality);
        self.dimensionality.sort();
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

        self.dimensionality
            .extend(rhs.dimensionality.into_iter().map(|d| -d));
        self.dimensionality.sort();
        self.simplify();
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

    #[case(0, vec![]; "Dimensionless")]
    #[case(1, vec![1]; "One dimension")]
    #[case(1 << 1, vec![2]; "One dimension in higher bit")]
    #[case(1 | (1 << 1), vec![1, 2]; "Multiple dimensions")]
    #[case(1 | (1 << 4), vec![1, 5]; "Multiple dimensions with gap")]
    fn test_get_dimensionality(unit_type: DimensionType, expected: Vec<Dimension>) {
        assert_eq!(get_dimensionality(unit_type), expected);
    }

    #[case(vec![], DIMENSIONLESS_TYPE; "Dimensionless")]
    #[case(vec![1], 1; "Single dimension")]
    #[case(vec![1, 1], 1; "Multiple matching dimensions")]
    #[case(vec![-1, 1], 1; "Multiple matching dimensions with sign difference")]
    #[case(vec![-2, -1, 1, 2, 3], 1 | (1 << 1) | (1 << 2); "Multiple dimensions")]
    fn test_dimensionality_to_type(dims: Vec<Dimension>, expected: DimensionType) {
        assert_eq!(dimensionality_to_type(&dims), expected);
    }

    #[case(vec![], vec![]; "Dimensionless")]
    #[case(vec![1], vec![1]; "Single dimension")]
    #[case(vec![1, 1], vec![1, 1]; "Unchanged")]
    #[case(vec![-1, -1], vec![-1, -1]; "Unchanged negative")]
    #[case(vec![-1, 1], vec![]; "Cancellation")]
    #[case(vec![-3, -2, -1, 1, 2, 3], vec![]; "Cancellation many")]
    #[case(vec![-1, -1, 1], vec![-1]; "Cancellation with remaining negative")]
    #[case(vec![-1, 1, 1], vec![1]; "Cancellation with remaining positive")]
    fn test_simplify_dimensionality(mut dimensionality: Vec<Dimension>, expected: Vec<Dimension>) {
        simplify_dimensionality(&mut dimensionality);
        assert_eq!(dimensionality, expected);
    }

    #[test]
    fn test_simplify() {
        let mut actual = BaseUnit {
            name: String::new(),
            multiplier: 1.0,
            unit_type: 0,
            dimensionality: vec![-3, -2, -1, 1, 2, 2, 3, 3],
            power: 1,
        };
        let expected = BaseUnit {
            name: String::new(),
            multiplier: 1.0,
            unit_type: (1 << 1) | (1 << 2),
            dimensionality: vec![2, 3],
            power: 1,
        };
        actual.simplify();
        assert_eq!(actual, expected);
    }

    #[case(
        BaseUnit::new("left".into(), 2.0, 1),
        BaseUnit::new("right".into(), 3.0, 1 << 1),
        BaseUnit::new("left".into(), 6.0, 1 | (1 << 1))
        ; "Multipliers multiply and unit types combine"
    )]
    fn test_mul(left: BaseUnit, right: BaseUnit, expected: BaseUnit) {
        println!("{:#?}", right);
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
            unit_type: 0,
            dimensionality: vec![0, 0, -1, -1, 2],
            power: 1,
        };
        let u2 = BaseUnit {
            name: "u2".into(),
            multiplier: 1.0,
            unit_type: 0,
            dimensionality: vec![0, -1, -1, 2],
            power: 1,
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
