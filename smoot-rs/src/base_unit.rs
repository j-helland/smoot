use std::{
    cmp::Ordering,
    ops::{AddAssign, Div, DivAssign, Mul, MulAssign, Neg, SubAssign},
};

use crate::{converter::Converter, hash::Hash, registry::Registry, utils::ApproxEq};
use bitcode::{Decode, Encode};
use hashable::Hashable;
use num_traits::{FromPrimitive, PrimInt};
use wide::i8x16;

use crate::error::{SmootError, SmootResult};

/// Dimension is essentially an enum value representing a physical dimension like `[length]` or `[time]`.
/// These values are determined dynamically at runtime, but we assume that there are no more than 127 of them.
///
/// Negative values are used to represent reciprocal dimensions e.g. `1 / [time]`. Note that `0` is a reserved
/// value that should never appear in dimensionality vectors for a unit.
pub type Dimension = i8;
pub const DIMENSIONLESS: Dimension = 0;

/// A bitmask indicating which Dimensions are active for a unit.
/// For example, a unit with dimensionality `[1, 1, 2, 5]` has a dimension type of `0b10011`.
pub type DimensionType = u64;
pub const DIMENSIONLESS_TYPE: DimensionType = 0;

#[derive(Encode, Decode, Hashable, Clone, Debug)]
pub struct BaseUnit {
    /// A human-readable identifier for this unit.
    pub name: String,

    /// The scalar used to compute conversion factors to and from this unit. For example, `kilometer`
    /// has a multiplier of `1000`, while `meter` has a multiplier of `1`.
    pub multiplier: f64,

    /// Offset for non-mulitiplicative units like temperatures (e.g. degC, degF).
    /// NOTE: `NaN` values are used as sentinels to indicate that this unit does not have an offset.
    pub offset: f64,

    /// The active physical dimensions for this unit.
    pub dimensionality: Vec<Dimension>,

    /// Holds user-applied exponents to a unit. Also useful for disply purposes e.g. creating a string `meter ** 2`.
    /// A `BaseUnit` will never be parsed from unit definitions with a non-unit power.
    pub power: i32,

    /// Specifies the conversion implementation to use when converting to/from this unit.
    /// For example, offset units like temperatures need different conversion logic than multiplicative units like lengths.
    pub converter: Converter,
}

impl PartialEq for BaseUnit {
    fn eq(&self, other: &Self) -> bool {
        // Treat all NaN values as equal for the sake of comparison.
        // NaN is simply used as a sentinel value for offsets.
        let is_offset_eq =
            (self.offset.is_nan() && other.offset.is_nan()) || self.offset.approx_eq(other.offset);

        is_offset_eq
            && self.name == other.name
            && self.multiplier.approx_eq(other.multiplier)
            && self.dimensionality == other.dimensionality
            && self.power == other.power
            && self.converter == other.converter
    }
}

impl BaseUnit {
    pub fn new(name: String, multiplier: f64, mut dimensionality: Vec<Dimension>) -> Self {
        debug_assert!(dimensionality.iter().all(|&d| d != 0));
        dimensionality.sort_unstable();
        Self {
            name,
            multiplier,
            offset: f64::NAN,
            dimensionality,
            power: 1,
            converter: Converter::Multiplicative,
        }
    }

    pub fn new_offset(
        name: String,
        multiplier: f64,
        offset: f64,
        mut dimensionality: Vec<Dimension>,
    ) -> Self {
        debug_assert!(dimensionality.iter().all(|&d| d != 0));
        dimensionality.sort_unstable();
        Self {
            name,
            multiplier,
            offset,
            dimensionality,
            power: 1,
            converter: Converter::Offset,
        }
    }

    /// Create a new, dimensionless base unit. These are used to store constants like pi.
    pub fn new_constant(multiplier: f64) -> Self {
        Self {
            name: String::new(),
            multiplier,
            offset: f64::NAN,
            dimensionality: vec![],
            power: 1,
            converter: Converter::Multiplicative,
        }
    }

    /// Return true if this unit is a composite of multiple dimensions.
    pub fn is_multidimensional(&self) -> bool {
        self.dimensionality
            .iter()
            .map(|dim| dim.abs() - 1)
            .fold(0, |acc, dim| acc | (1 << dim))
            .count_ones()
            > 1
    }

    /// Return a mask whose set bits correspond to the dimensions of this unit.
    ///
    /// Examples
    /// --------
    /// `dimensionality: [1, 1, 3, 4] -> 0b1101`
    pub fn get_dimension_type(&self) -> DimensionType {
        dimensionality_to_type(&self.dimensionality)
    }

    /// Return true if this unit is a constant value e.g. `1`.
    /// Constant values can result from parsing unit expressions like `2 / meter`.
    pub fn is_constant(&self) -> bool {
        self.name.is_empty()
    }

    /// Return true if this unit has an offset i.e. is non-multiplicative.
    pub fn is_offset(&self) -> bool {
        !self.offset.is_nan()
    }

    /// Get the multiplicative factor associated with this base unit.
    pub fn get_multiplier(&self) -> f64 {
        self.multiplier.powi(self.power)
    }

    /// Return true if this unit can be used in expressions with another unit.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        is_dim_eq(&self.dimensionality, &other.dimensionality) && self.converter == other.converter
    }

    /// Convert this unit into its delta variant if thi unit is an offset unit. Otherwise, do nothing.
    pub fn into_delta(self) -> Self {
        if self.is_offset() {
            Self::new(
                Registry::DELTA_PREFIX.to_string() + self.name.as_str(),
                self.multiplier,
                self.dimensionality.clone(),
            )
        } else {
            self
        }
    }

    pub fn convert_from<S, N>(&self, value: &mut S)
    where
        N: FromPrimitive,
        S: MulAssign<N> + AddAssign<N>,
    {
        self.converter.convert_from(value, self);
    }

    pub fn convert_to<S, N>(&self, value: &mut S)
    where
        N: FromPrimitive,
        S: DivAssign<N> + SubAssign<N>,
    {
        self.converter.convert_to(value, self);
    }

    /// Take the in-place p-th power of this unit.
    pub fn ipowi(&mut self, p: i32) {
        if p == 0 {
            self.power = 0;
            self.dimensionality.clear();
            return;
        }
        if p == 1 {
            return;
        }

        self.power *= p;
        let ncopies = p.unsigned_abs().saturating_sub(1) as usize;
        let copies = self.dimensionality.repeat(ncopies);
        self.dimensionality.extend(copies);
        if p.is_negative() {
            self.dimensionality.iter_mut().for_each(|d| *d = d.neg());
        }
        self.dimensionality.sort_unstable();
    }

    /// Return the p-th power of this unit.
    pub fn powi(&self, p: i32) -> Self {
        let mut new = self.clone();
        new.ipowi(p);
        new
    }

    /// In-place take the p-th root of this unit.
    ///
    /// Errors
    /// ------
    /// `SmootError::InvalidOperation`
    ///     If the root would result in a non-integral power for this unit.
    pub fn ipow_root(&mut self, p: i32) -> SmootResult<()> {
        self.dimensionality = pow_root_dimensionality(p.unsigned_abs(), &self.dimensionality)
            .map_err(|_| {
                SmootError::InvalidOperation(format!(
                    "Power '1 / {p}' would result in a non-integral power for '{} ** {}'",
                    self.name, self.power
                ))
            })?;
        self.power /= p;
        Ok(())
    }

    /// Return the p-th root of this unit.
    ///
    /// Errors
    /// ------
    /// `SmootError::InvalidOperation`
    ///     If the root would result in a non-integral power for this unit.
    pub fn pow_root(&self, p: i32) -> SmootResult<Self> {
        let mut new = self.clone();
        new.ipow_root(p)?;
        Ok(new)
    }

    pub fn simplify(&mut self) {
        simplify_dimensionality(&mut self.dimensionality);
    }

    pub fn div_dimensionality(&mut self, other: &Self) {
        self.power -= other.power;
        if self.power == 0 {
            self.dimensionality.clear();
            return;
        }

        self.dimensionality
            .extend(other.dimensionality.iter().map(|d| -d));
        self.dimensionality.sort_unstable();
        self.simplify();
    }
}

pub(crate) fn pow_root_dimensionality(
    p: u32,
    dimensionality: &[Dimension],
) -> SmootResult<Vec<Dimension>> {
    if dimensionality.is_empty() {
        return Ok(vec![]);
    }

    let p = p as usize;
    if p == 0 || dimensionality.len() % p != 0 {
        return Err(SmootError::SmootError);
    }

    let mut result = Vec::with_capacity(dimensionality.len() / p);
    let mut last = dimensionality[0];
    let mut count = 1;

    for &num in &dimensionality[1..] {
        if num == last {
            count += 1;
            continue;
        }

        let new_count = count / p;
        result.extend(std::iter::repeat_n(last, new_count));

        last = num;
        count = 1;
    }

    let new_count = count / p;
    result.extend(std::iter::repeat_n(last, new_count));

    Ok(result)
}

/// Return true if two dimensionality vectors are equivalent.
/// Uses SIMD where possible to parallelize comparisons.
pub(crate) fn is_dim_eq(dim1: &[Dimension], dim2: &[Dimension]) -> bool {
    const CHUNK_SIZE: usize = size_of::<i8x16>();

    if dim1.len() != dim2.len() {
        return false;
    }

    let mut chunks1 = dim1.chunks_exact(CHUNK_SIZE);
    let mut chunks2 = dim2.chunks_exact(CHUNK_SIZE);

    while let (Some(c1), Some(c2)) = (chunks1.next(), chunks2.next()) {
        let c1_simd = i8x16::from_slice_unaligned(c1);
        let c2_simd = i8x16::from_slice_unaligned(c2);
        if c1_simd != c2_simd {
            return false;
        }
    }
    chunks1.remainder() == chunks2.remainder()
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
        let sum = i16::from(dimensionality[left]) + i16::from(dimensionality[right]);
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
        self.dimensionality.extend(rhs.dimensionality);
        self.dimensionality.sort_unstable();
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
        self.dimensionality
            .extend(rhs.dimensionality.into_iter().map(|d| -d));
        self.dimensionality.sort_unstable();
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

    use super::*;

    #[case(vec![], 0, Some(vec![]))]
    #[case(vec![], 1, Some(vec![]))]
    #[case(vec![], 2, Some(vec![]))]
    #[case(vec![1], 1, Some(vec![1]))]
    #[case(vec![1], 2, None)]
    #[case(vec![1, 1], 2, Some(vec![1]))]
    #[case(vec![1, 1, 2, 2, 2, 2], 2, Some(vec![1, 2, 2]))]
    #[case(vec![1, 1, 2, 2, 2], 2, None)]
    #[case(vec![1, 1, 1], 3, Some(vec![1]))]
    #[case(vec![1, 1, 1, 2, 2, 2], 3, Some(vec![1, 2]))]
    #[case(vec![1, 1, 1, 2, 2], 3, None)]
    #[case(vec![-1, -1], 2, Some(vec![-1]); "Negative dimensions")]
    fn test_pow_root_dimensionality(
        dims: Vec<Dimension>,
        p: u32,
        expected: Option<Vec<Dimension>>,
    ) -> SmootResult<()> {
        let actual = pow_root_dimensionality(p, &dims);
        if let Some(expected) = expected {
            let actual = actual?;
            assert_eq!(actual, expected, "{:?} != {:?}", actual, expected);
        } else {
            assert!(actual.is_err());
        }
        Ok(())
    }

    #[case(vec![], vec![], true; "Dimensionless")]
    #[case(vec![1], vec![], false; "Dimensionless mismatch")]
    #[case(vec![1, 1], vec![1], false; "Repeated dimension mismatch")]
    #[case(vec![1, 2, 3], vec![1, 2, 3], true; "Multidimensional")]
    #[case(
        (0..32).collect(),
        (0..32).collect(),
        true
        ; "Multidimensional with chunks"
    )]
    fn test_is_dim_eq(dims1: Vec<Dimension>, dims2: Vec<Dimension>, expected: bool) {
        assert_eq!(is_dim_eq(&dims1, &dims2), expected);
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
            offset: f64::NAN,
            dimensionality: vec![-3, -2, -1, 1, 2, 2, 3, 3],
            power: 1,
            converter: Converter::Multiplicative,
        };
        let expected = BaseUnit {
            name: String::new(),
            multiplier: 1.0,
            offset: f64::NAN,
            dimensionality: vec![2, 3],
            power: 1,
            converter: Converter::Multiplicative,
        };
        actual.simplify();
        assert_eq!(actual, expected);
    }

    #[case(
        BaseUnit::new("left".into(), 2.0, vec![1]),
        BaseUnit::new("right".into(), 3.0, vec![2]),
        BaseUnit::new("left".into(), 6.0, vec![1, 2])
        ; "Multipliers multiply and unit types combine"
    )]
    fn test_mul(left: BaseUnit, right: BaseUnit, expected: BaseUnit) {
        assert_eq!(left * right, expected);
    }

    #[case(
        BaseUnit::new("test".into(), 1.0, vec![1]),
        false
        ; "single dimension 1"
    )]
    #[case(
        BaseUnit::new("test".into(), 1.0, vec![8]),
        false
        ; "single dimension 2"
    )]
    #[case(
        BaseUnit::new("test".into(), 1.0, vec![3, 8]),
        true
        ; "multidimensional"
    )]
    fn test_is_multidimensional(unit: BaseUnit, expected: bool) {
        assert_eq!(unit.is_multidimensional(), expected);
    }

    #[test]
    fn test_hash() {
        let u1 = BaseUnit::new("u1".into(), 1.0, vec![]);
        assert_eq!(hash(&u1), hash(&u1.clone()));

        let u2 = BaseUnit::new("u2".into(), 2.0, vec![1]);
        assert_ne!(hash(&u1), hash(&u2));
    }

    fn hash<T: Hash>(val: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        val.hash(&mut hasher);
        hasher.finish()
    }
}
