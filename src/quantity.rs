use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use bitcode::{Decode, Encode};
use numpy::ndarray::ArrayD;

use crate::{
    error::{SmootError, SmootResult},
    parser::expression_parser,
    registry::Registry,
    types::Number,
    unit::Unit,
    utils::{Powf, Powi},
};

pub trait Storage<N: Number>: Mul<N, Output = Self> + MulAssign<N> + Clone + Sized {}
impl<N: Number> Storage<N> for N {}
impl<N: Number> Storage<N> for ArrayD<N> {}

#[derive(Encode, Decode, Clone, Debug, PartialEq)]
pub struct Quantity<N: Number, S: Storage<N>> {
    pub magnitude: S,
    pub unit: Unit<f64>,

    _marker: PhantomData<N>,
}

impl<N: Number, S: Storage<N>> Quantity<N, S> {
    pub fn new(magnitude: S, unit: Unit<f64>) -> Self {
        Self {
            magnitude,
            unit,
            _marker: PhantomData,
        }
    }

    pub fn new_dimensionless(magnitude: S) -> Self {
        Self {
            magnitude,
            unit: Unit::new(vec![], vec![]),
            _marker: PhantomData,
        }
    }

    pub fn is_dimensionless(&self) -> bool {
        self.unit.is_dimensionless()
    }

    pub fn m_as(&self, unit: &Unit<f64>) -> SmootResult<S> {
        if self.unit.eq(unit) {
            Ok(self.magnitude.clone())
        } else {
            let factor = self.unit.conversion_factor(unit)?;
            Ok(self.magnitude.clone() * N::from_f64(factor).unwrap())
        }
    }

    pub fn to(&self, unit: &Unit<f64>) -> SmootResult<Quantity<N, S>> {
        let mut q = self.clone();
        q.ito(unit)?;
        Ok(q)
    }

    pub fn ito(&mut self, unit: &Unit<f64>) -> SmootResult<()> {
        if self.unit.eq(unit) {
            return Ok(());
        }

        if !self.unit.is_compatible_with(unit) {
            return Err(SmootError::IncompatibleUnitTypes(
                self.unit
                    .get_units_string()
                    .unwrap_or("dimensionless".into()),
                unit.get_units_string().unwrap_or("dimensionless".into()),
            ));
        }
        self.convert_to(unit)
    }

    pub fn ito_reduced_units(&mut self) {
        let factor = self.unit.reduce();
        self.magnitude *= N::from_f64(factor).unwrap();
    }

    fn convert_to(&mut self, unit: &Unit<f64>) -> SmootResult<()> {
        let factor = self.unit.conversion_factor(unit)?;
        self.magnitude *= N::from_f64(factor).unwrap();
        self.unit = unit.clone();
        Ok(())
    }
}

impl Quantity<f64, f64> {
    pub fn parse(registry: &Registry, s: &str) -> SmootResult<Self> {
        let s = s.replace(" ", "");
        expression_parser::expression(&s, registry)
            .map(|mut q| {
                q.ito_reduced_units();
                q
            })
            .map_err(|_| SmootError::InvalidQuantityExpression(0, s))
    }
}

impl<N, S> Powf for Quantity<N, S>
where
    N: Number,
    S: Storage<N> + Powf,
{
    fn powf(self, p: f64) -> Self {
        Quantity::new(self.magnitude.powf(p), self.unit.powf(p))
    }
}
impl<N, S> Powi for Quantity<N, S>
where
    N: Number,
    S: Storage<N> + Powi,
{
    fn powi(self, p: i32) -> Self {
        Quantity::new(self.magnitude.powi(p), self.unit.powi(p))
    }
}

impl<N: Number, S: Storage<N> + PartialOrd> PartialOrd for Quantity<N, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Convert to same units before comparison.
        other
            .m_as(&self.unit)
            .ok()
            .and_then(|o| self.magnitude.partial_cmp(&o))
    }
}

/// Multiply quantities
impl<N: Number, S: Storage<N>> MulAssign for Quantity<N, S>
where
    S: MulAssign,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.unit.mul_assign(&rhs.unit);
        let factor = self.unit.simplify(true);
        self.magnitude *= rhs.magnitude * N::from_f64(factor).unwrap();
    }
}
impl<N: Number, S: Storage<N>> MulAssign<&Quantity<N, S>> for Quantity<N, S>
where
    S: MulAssign,
{
    fn mul_assign(&mut self, rhs: &Self) {
        self.unit.mul_assign(&rhs.unit);
        let factor = self.unit.simplify(true);
        self.magnitude *= rhs.magnitude.clone() * N::from_f64(factor).unwrap();
    }
}
impl<N: Number, S: Storage<N>> Mul<&Quantity<N, S>> for &Quantity<N, S>
where
    S: MulAssign,
{
    type Output = Quantity<N, S>;

    fn mul(self, rhs: &Quantity<N, S>) -> Self::Output {
        let mut new = self.clone();
        new *= rhs;
        new
    }
}
/// Scalar multiplication
impl<N: Number, S: Storage<N>> MulAssign<N> for Quantity<N, S> {
    fn mul_assign(&mut self, rhs: N) {
        self.magnitude *= rhs;
    }
}
impl<N: Number, S: Storage<N>> Mul<N> for &Quantity<N, S> {
    type Output = Quantity<N, S>;

    fn mul(self, rhs: N) -> Self::Output {
        let mut new = self.clone();
        new *= rhs;
        new
    }
}
impl<N: Number, S: Storage<N>> Mul<N> for Quantity<N, S> {
    type Output = Quantity<N, S>;

    fn mul(self, rhs: N) -> Self::Output {
        let mut new = self.clone();
        new *= rhs;
        new
    }
}
/// Array multiplication
impl<N: Number> Mul<ArrayD<N>> for Quantity<N, ArrayD<N>> {
    type Output = Self;

    fn mul(mut self, rhs: ArrayD<N>) -> Self {
        self.magnitude = self.magnitude * rhs;
        self
    }
}
impl<N: Number> Mul<Quantity<N, ArrayD<N>>> for Quantity<N, ArrayD<N>> {
    type Output = Self;

    fn mul(mut self, rhs: Quantity<N, ArrayD<N>>) -> Self::Output {
        self.unit.mul_assign(&rhs.unit);
        let factor = self.unit.simplify(true);
        let factor = N::from_f64(factor).unwrap();
        self.magnitude = self.magnitude * rhs.magnitude * factor;
        self
    }
}

/// Add quantities
impl<N: Number, S: Storage<N>> AddAssign for Quantity<N, S>
where
    S: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        let conversion_factor = self
            .unit
            .conversion_factor(&rhs.unit)
            .expect("Incompatible units used in add_assign");
        let conversion_factor = N::from_f64(conversion_factor).unwrap();
        self.magnitude += rhs.magnitude * conversion_factor;
    }
}
impl<N: Number, S: Storage<N>> AddAssign<&Quantity<N, S>> for Quantity<N, S>
where
    S: AddAssign,
{
    fn add_assign(&mut self, rhs: &Self) {
        let conversion_factor = self
            .unit
            .conversion_factor(&rhs.unit)
            .expect("Incompatible units used in add_assign");
        let conversion_factor = N::from_f64(conversion_factor).unwrap();
        self.magnitude += rhs.magnitude.clone() * conversion_factor;
    }
}
impl<N: Number, S: Storage<N>> Add<&Quantity<N, S>> for &Quantity<N, S>
where
    S: AddAssign,
{
    type Output = SmootResult<Quantity<N, S>>;

    fn add(self, rhs: &Quantity<N, S>) -> Self::Output {
        let mut new = self.clone();
        let conversion_factor = new.unit.conversion_factor(&rhs.unit)?;
        let conversion_factor = N::from_f64(conversion_factor).unwrap();
        new.magnitude += rhs.magnitude.clone() * conversion_factor;
        Ok(new)
    }
}
/// Add scalar
impl<N: Number, S: Storage<N>> Add<N> for Quantity<N, S>
where
    S: AddAssign<N>,
{
    type Output = SmootResult<Self>;

    fn add(mut self, rhs: N) -> Self::Output {
        if !self.unit.is_dimensionless() {
            return Err(SmootError::InvalidOperation(
                "+",
                self.unit
                    .get_units_string()
                    .unwrap_or("dimensionless".into()),
                "dimensionaless".into(),
            ));
        }
        self.magnitude += rhs;
        Ok(self)
    }
}
/// Add array
impl<N: Number> Add<ArrayD<N>> for Quantity<N, ArrayD<N>> {
    type Output = SmootResult<Self>;

    fn add(mut self, rhs: ArrayD<N>) -> Self::Output {
        if !self.unit.is_dimensionless() {
            return Err(SmootError::InvalidOperation(
                "+",
                self.unit
                    .get_units_string()
                    .unwrap_or("dimensionless".into()),
                "dimensionaless".into(),
            ));
        }
        self.magnitude = self.magnitude + rhs;
        Ok(self)
    }
}
impl<N: Number> Add<Quantity<N, ArrayD<N>>> for Quantity<N, ArrayD<N>> {
    type Output = SmootResult<Self>;

    fn add(mut self, rhs: Quantity<N, ArrayD<N>>) -> Self::Output {
        if !self.unit.is_compatible_with(&rhs.unit) {
            return Err(SmootError::InvalidOperation(
                "+",
                self.unit
                    .get_units_string()
                    .unwrap_or("dimensionless".into()),
                rhs.unit
                    .get_units_string()
                    .unwrap_or("dimensionless".into()),
            ));
        }
        self.magnitude = self.magnitude + rhs.magnitude;
        Ok(self)
    }
}

/// Negate quantities
impl<N: Number, S: Storage<N>> Neg for Quantity<N, S>
where
    S: Neg<Output = S>,
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.magnitude = self.magnitude.neg();
        self
    }
}

/// Subtract quantities
impl<N: Number, S: Storage<N>> SubAssign for Quantity<N, S>
where
    S: SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        let conversion_factor = self
            .unit
            .conversion_factor(&rhs.unit)
            .expect("Incompatible units used in sub_assign");
        let conversion_factor = N::from_f64(conversion_factor).unwrap();
        self.magnitude -= rhs.magnitude * conversion_factor;
    }
}
impl<N: Number, S: Storage<N>> SubAssign<&Quantity<N, S>> for Quantity<N, S>
where
    S: SubAssign,
{
    fn sub_assign(&mut self, rhs: &Self) {
        let conversion_factor = self
            .unit
            .conversion_factor(&rhs.unit)
            .expect("Incompatible units used in sub_assign");
        let conversion_factor = N::from_f64(conversion_factor).unwrap();
        self.magnitude -= rhs.magnitude.clone() * conversion_factor;
    }
}
impl<N: Number, S: Storage<N>> Sub<&Quantity<N, S>> for &Quantity<N, S>
where
    S: SubAssign,
{
    type Output = SmootResult<Quantity<N, S>>;

    fn sub(self, rhs: &Quantity<N, S>) -> Self::Output {
        let conversion_factor = self.unit.conversion_factor(&rhs.unit)?;
        let conversion_factor = N::from_f64(conversion_factor).unwrap();
        let mut new = self.clone();
        new.magnitude -= rhs.magnitude.clone() * conversion_factor;
        Ok(new)
    }
}
/// Subtract scalar
impl<N: Number, S: Storage<N>> Sub<N> for Quantity<N, S>
where
    S: SubAssign<N>,
{
    type Output = SmootResult<Self>;

    fn sub(mut self, rhs: N) -> Self::Output {
        if !self.unit.is_dimensionless() {
            return Err(SmootError::InvalidOperation(
                "-",
                self.unit
                    .get_units_string()
                    .unwrap_or("dimensionless".into()),
                "dimensionaless".into(),
            ));
        }
        self.magnitude -= rhs;
        Ok(self)
    }
}
/// Subtract array
impl<N: Number> Sub<ArrayD<N>> for Quantity<N, ArrayD<N>> {
    type Output = SmootResult<Self>;

    fn sub(mut self, rhs: ArrayD<N>) -> Self::Output {
        if !self.unit.is_dimensionless() {
            return Err(SmootError::InvalidOperation(
                "-",
                self.unit
                    .get_units_string()
                    .unwrap_or("dimensionless".into()),
                "dimensionaless".into(),
            ));
        }
        self.magnitude = self.magnitude - rhs;
        Ok(self)
    }
}
impl<N: Number> Sub<Quantity<N, ArrayD<N>>> for Quantity<N, ArrayD<N>> {
    type Output = SmootResult<Self>;

    fn sub(mut self, rhs: Quantity<N, ArrayD<N>>) -> Self::Output {
        if !self.unit.is_compatible_with(&rhs.unit) {
            return Err(SmootError::InvalidOperation(
                "-",
                self.unit
                    .get_units_string()
                    .unwrap_or("dimensionless".into()),
                rhs.unit
                    .get_units_string()
                    .unwrap_or("dimensionless".into()),
            ));
        }
        self.magnitude = self.magnitude - rhs.magnitude;
        Ok(self)
    }
}

/// Divide quantities
impl<N: Number, S: Storage<N>> Div<&Quantity<N, S>> for &Quantity<N, S>
where
    S: DivAssign,
{
    type Output = Quantity<N, S>;

    fn div(self, rhs: &Quantity<N, S>) -> Self::Output {
        let mut new = self.clone();
        new /= rhs;
        new
    }
}
impl<N: Number, S: Storage<N>> DivAssign for Quantity<N, S>
where
    S: DivAssign,
{
    fn div_assign(&mut self, rhs: Self) {
        self.magnitude /= rhs.magnitude;
        self.unit.div_assign(&rhs.unit);
        let factor = self.unit.simplify(true);
        let factor = N::from_f64(factor).unwrap();
        self.magnitude *= factor;
    }
}
impl<N: Number, S: Storage<N>> DivAssign<&Quantity<N, S>> for Quantity<N, S>
where
    S: DivAssign,
{
    fn div_assign(&mut self, rhs: &Self) {
        self.magnitude /= rhs.magnitude.clone();
        self.unit.div_assign(&rhs.unit);
        let factor = self.unit.simplify(true);
        let factor = N::from_f64(factor).unwrap();
        self.magnitude *= factor;
    }
}

/// Divide scalar
impl<N: Number, S: Storage<N>> Div<N> for Quantity<N, S>
where
    S: DivAssign<N>,
{
    type Output = Self;

    fn div(mut self, rhs: N) -> Self::Output {
        self.magnitude /= rhs;
        self
    }
}
/// Divide array
impl<N: Number> Div<ArrayD<N>> for Quantity<N, ArrayD<N>> {
    type Output = Self;

    fn div(mut self, rhs: ArrayD<N>) -> Self::Output {
        self.magnitude = self.magnitude / rhs;
        self
    }
}
impl<N: Number> Div<Quantity<N, ArrayD<N>>> for Quantity<N, ArrayD<N>> {
    type Output = Self;

    fn div(mut self, rhs: Quantity<N, ArrayD<N>>) -> Self::Output {
        self.magnitude = self.magnitude / rhs.magnitude;
        self
    }
}

impl From<Quantity<f64, f64>> for Quantity<i64, i64> {
    fn from(value: Quantity<f64, f64>) -> Self {
        Self {
            magnitude: value.magnitude as i64,
            unit: value.unit,
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod test_quantity {
    use std::sync::{Arc, LazyLock};

    use numpy::ndarray::Array;

    use crate::{assert_is_close, base_unit::BaseUnit, registry::REGISTRY};

    use super::*;

    static UNIT_METER: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("meter").expect("No unit 'meter'"));
    static UNIT_KILOMETER: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("kilometer").expect("No unit 'kilometer'"));
    static UNIT_SECOND: LazyLock<&Arc<BaseUnit<f64>>> =
        LazyLock::new(|| REGISTRY.get_unit("second").expect("No unit 'second`'"));

    #[test]
    fn test_quantity_ito_fractional_power() -> SmootResult<()> {
        let km_sqrt = Unit::new(vec![BaseUnit::clone(&UNIT_KILOMETER)], vec![]).powf(0.5);
        let m_sqrt = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]).powf(0.5);

        let mut q = Quantity::new(1.0, km_sqrt);

        q.ito(&m_sqrt)?;

        assert_is_close!(q.magnitude, UNIT_KILOMETER.multiplier.sqrt());
        assert_eq!(q.unit, m_sqrt);

        Ok(())
    }

    #[test]
    fn test_quantity_ito() -> SmootResult<()> {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let kilometer = Unit::new(vec![BaseUnit::clone(&UNIT_KILOMETER)], vec![]);

        let mut q = Quantity::new(1.0, meter);

        q.ito(&kilometer)?;

        assert_is_close!(q.magnitude, 1.0 / 1000.0);
        assert_eq!(q.unit, kilometer);

        Ok(())
    }

    #[test]
    fn test_quantity_ito_incompatible_units() {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let second = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);
        let mut q = Quantity::new(1.0, meter);

        assert!(q.ito(&second).is_err());
    }

    #[test]
    fn test_quantity_to() -> SmootResult<()> {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let kilometer = Unit::new(vec![BaseUnit::clone(&UNIT_KILOMETER)], vec![]);

        let q = Quantity::new(1.0, meter);

        let q_converted = q.to(&kilometer)?;

        assert_is_close!(q_converted.magnitude, 1.0 / 1000.0);
        assert_eq!(q_converted.unit, kilometer);

        Ok(())
    }

    #[test]
    fn test_quantity_to_incompatible_units() {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let second = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);
        let q = Quantity::new(1.0, meter);

        assert!(q.to(&second).is_err());
    }

    #[test]
    fn test_quantity_m_as() -> SmootResult<()> {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let kilometer = Unit::new(vec![BaseUnit::clone(&UNIT_KILOMETER)], vec![]);
        let q = Quantity::new(1.0, meter);

        let magnitude = q.m_as(&kilometer)?;

        assert_is_close!(magnitude, 1.0 / 1000.0);

        Ok(())
    }

    #[test]
    fn test_quantity_m_as_incompatible_units() {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let second = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);
        let q = Quantity::new(1.0, meter);

        assert!(q.m_as(&second).is_err());
    }

    #[test]
    fn test_quantity_quantity_mul() {
        let q1 = Quantity::new_dimensionless(1.0);
        let q2 = Quantity::new_dimensionless(2.0);
        let q = &q1 * &q2;
        assert_is_close!(q.magnitude, 2.0);
    }

    #[test]
    fn test_quantity_scalar_mul() {
        let q = Quantity::new_dimensionless(1.0);
        let q_scaled = q * 2.0;
        assert_is_close!(q_scaled.magnitude, 2.0);
    }

    #[test]
    /// Can element-wise multiple array quantity by scalar
    fn test_quantity_array_scalar_mul() {
        let arr = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let q = Quantity::new_dimensionless(arr);

        let q_scaled = q * 2.0;

        q_scaled.magnitude.for_each(|x| {
            assert_is_close!(*x, 2.0);
        });
    }

    #[test]
    /// Can element-wise multiple array quantity by naked array
    fn test_quantity_array_array_mul() {
        let arr1 = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let arr2 = Array::from_shape_vec(vec![2], vec![2.0; 2]).unwrap();
        let q = Quantity::new_dimensionless(arr1);

        let q_scaled = q * arr2;

        q_scaled.magnitude.for_each(|x| {
            assert_is_close!(*x, 2.0);
        });
    }

    #[test]
    /// Can element-wise multiply array quantities
    fn test_quantity_array_quantity_array_mul() {
        let arr1 = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let arr2 = Array::from_shape_vec(vec![2], vec![2.0; 2]).unwrap();
        let q1 = Quantity::new_dimensionless(arr1);
        let q2 = Quantity::new_dimensionless(arr2);

        let q_scaled = q1 * q2;

        q_scaled.magnitude.for_each(|x| {
            assert_is_close!(*x, 2.0);
        });
    }

    #[test]
    /// Can add quantities with compatible units
    fn test_quantity_add() -> SmootResult<()> {
        let q1 = Quantity::new_dimensionless(1.0);
        let q2 = Quantity::new_dimensionless(2.0);

        let q = (&q1 + &q2)?;
        assert_is_close!(q.magnitude, 3.0);

        Ok(())
    }

    #[test]
    /// Cannot add quantities with incompatible units
    fn test_quantity_add_incompatible_units() {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let second = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);
        let q1 = Quantity::new(1.0, meter);
        let q2 = Quantity::new(2.0, second);

        let result = &q1 + &q2;
        assert!(result.is_err());
    }

    #[test]
    /// Can add dimensionless scalar to a dimensionless quantity
    fn test_quantity_add_scalar() -> SmootResult<()> {
        let q = Quantity::new_dimensionless(1.0);

        let q_scaled = (q + 2.0)?;
        assert!(q_scaled.unit.is_dimensionless());
        assert_is_close!(q_scaled.magnitude, 3.0);

        Ok(())
    }

    #[test]
    /// Cannot add dimensionless scalar to a non-dimensionless quantity
    fn test_quantity_add_scalar_incompatible_units() {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let q = Quantity::new(1.0, meter);
        let result = q + 2.0;
        assert!(result.is_err());
    }

    #[test]
    /// Can add scalar to an array quantity
    fn test_quantity_array_scalar_add() -> SmootResult<()> {
        let arr = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let q = Quantity::new_dimensionless(arr);

        let q_scaled = (q + 2.0)?;
        q_scaled.magnitude.for_each(|x| {
            assert_is_close!(*x, 3.0);
        });

        Ok(())
    }

    #[test]
    /// Can add an array to an array quantity
    fn test_quantity_array_array_add() -> SmootResult<()> {
        let arr1 = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let arr2 = Array::from_shape_vec(vec![2], vec![2.0; 2]).unwrap();
        let q1 = Quantity::new_dimensionless(arr1);

        let q_scaled = (q1 + arr2)?;

        q_scaled.magnitude.for_each(|x| {
            assert_is_close!(*x, 3.0);
        });

        Ok(())
    }

    #[test]
    /// Can add an array quantity to an array quantity
    fn test_quantity_array_quantity_array_add() -> SmootResult<()> {
        let arr1 = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let arr2 = Array::from_shape_vec(vec![2], vec![2.0; 2]).unwrap();
        let q1 = Quantity::new_dimensionless(arr1);
        let q2 = Quantity::new_dimensionless(arr2);

        let q_scaled = (q1 + q2)?;

        q_scaled.magnitude.for_each(|x| {
            assert_is_close!(*x, 3.0);
        });

        Ok(())
    }

    #[test]
    /// Can negate a quantity
    fn test_quantity_neg() {
        let q = Quantity::new_dimensionless(1.0);
        let neg_q = -q;
        assert_eq!(neg_q.magnitude, -1.0);
    }

    #[test]
    /// Can negate an array quantity
    fn test_quantity_neg_array() {
        let arr = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let q = Quantity::new_dimensionless(arr);

        let neg_q = -q;

        neg_q.magnitude.for_each(|x| {
            assert_eq!(*x, -1.0);
        });
    }

    #[test]
    /// Can subtract quantities with compatible units
    fn test_quantity_sub() -> SmootResult<()> {
        let q1 = Quantity::new_dimensionless(1.0);
        let q2 = Quantity::new_dimensionless(2.0);

        let q = (&q1 - &q2)?;
        assert_is_close!(q.magnitude, -1.0);

        Ok(())
    }

    #[test]
    /// Cannot subtract quantities with incompatible units
    fn test_quantity_sub_incompatible_units() {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let second = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);
        let q1 = Quantity::new(1.0, meter);
        let q2 = Quantity::new(2.0, second);

        let result = &q1 - &q2;
        assert!(result.is_err());
    }

    #[test]
    /// Can subtract dimensionless scalar from a dimensionless quantity
    fn test_quantity_sub_scalar() -> SmootResult<()> {
        let q = Quantity::new_dimensionless(1.0);

        let q_scaled = (q - 2.0)?;
        assert!(q_scaled.unit.is_dimensionless());
        assert_is_close!(q_scaled.magnitude, -1.0);

        Ok(())
    }

    #[test]
    /// Can subtract an array from an array quantity
    fn test_quantity_array_array_sub() -> SmootResult<()> {
        let arr1 = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let arr2 = Array::from_shape_vec(vec![2], vec![2.0; 2]).unwrap();
        let q1 = Quantity::new_dimensionless(arr1);

        let q = (q1 - arr2)?;

        q.magnitude.for_each(|x| {
            assert_is_close!(*x, -1.0);
        });

        Ok(())
    }

    #[test]
    /// Can subtract an array quantity from an array quantity
    fn test_quantity_array_quantity_array_sub() -> SmootResult<()> {
        let arr1 = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let arr2 = Array::from_shape_vec(vec![2], vec![2.0; 2]).unwrap();
        let q1 = Quantity::new_dimensionless(arr1);
        let q2 = Quantity::new_dimensionless(arr2);

        let q = (q1 - q2)?;

        q.magnitude.for_each(|x| {
            assert_is_close!(*x, -1.0);
        });

        Ok(())
    }

    #[test]
    /// Can divide quantity by quantity
    fn test_quantity_div_quantity() {
        let q1 = Quantity::new_dimensionless(1.0);
        let q2 = Quantity::new_dimensionless(2.0);

        let q = &q1 / &q2;

        assert_is_close!(q.magnitude, 0.5);
    }

    #[test]
    /// Can divide quantity by scalar
    fn test_quantity_scalar_div() {
        let q = Quantity::new_dimensionless(1.0);
        let q_scaled = q / 2.0;
        assert_is_close!(q_scaled.magnitude, 0.5);
    }

    #[test]
    /// Can divide array quantity by scalar
    fn test_quantity_array_quantity_scalar_div() {
        let arr = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let q = Quantity::new_dimensionless(arr);

        let q_scaled = q / 2.0;

        q_scaled.magnitude.for_each(|x| {
            assert_is_close!(*x, 0.5);
        });
    }

    #[test]
    /// Can divide array quantity by array
    fn test_quantity_array_quantity_array_div() {
        let arr1 = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let arr2 = Array::from_shape_vec(vec![2], vec![2.0; 2]).unwrap();
        let q = Quantity::new_dimensionless(arr1);

        let q_scaled = q / arr2;

        q_scaled.magnitude.for_each(|x| {
            assert_is_close!(*x, 0.5);
        });
    }

    #[test]
    /// Can divide array quantity by array quantity
    fn test_quantity_array_quantity_array_quantity_div() {
        let arr1 = Array::from_shape_vec(vec![2], vec![1.0; 2]).unwrap();
        let arr2 = Array::from_shape_vec(vec![2], vec![2.0; 2]).unwrap();
        let q1 = Quantity::new_dimensionless(arr1);
        let q2 = Quantity::new_dimensionless(arr2);

        let q = q1 / q2;

        q.magnitude.for_each(|x| {
            assert_is_close!(*x, 0.5);
        });
    }
}
