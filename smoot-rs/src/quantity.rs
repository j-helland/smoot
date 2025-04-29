use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

use bitcode::{Decode, Encode};
use hashable::Hashable;
use ndarray::{Array, ArrayD, Ix1, Ix2, Zip};

use crate::{
    error::{SmootError, SmootResult},
    hash::Hash,
    parser::expression_parser,
    registry::Registry,
    types::Number,
    unit::{Dimensionality, Unit},
    utils::{ConvertMagnitude, LogExp, Powi, Sqrt, Trigonometry},
};

pub trait Storage<N: Number>:
    Mul<N, Output = Self>
    + MulAssign<N>
    + Sqrt
    + Trigonometry<Output = Self>
    + LogExp<Output = Self>
    + Clone
    + Sized
{
}
impl<N: Number> Storage<N> for N {}
impl<N: Number> Storage<N> for ArrayD<N> {}

#[derive(Encode, Decode, Hashable, Clone, Debug, PartialEq)]
pub struct Quantity<N: Number, S: Storage<N>> {
    /// e.g. `1` in `1 meter`.
    pub magnitude: S,

    /// e.g. `meter` in `1 meter`.
    pub unit: Unit,

    _marker: PhantomData<N>,
}

impl<N: Number, S: Storage<N>> Quantity<N, S>
where
    S: ConvertMagnitude,
{
    pub fn new(magnitude: S, unit: Unit) -> Self {
        Self {
            magnitude,
            unit,
            _marker: PhantomData, // Only needed to support the `N` generic.
        }
    }

    /// Create a new, dimensionless quantity.
    pub fn new_dimensionless(magnitude: S) -> Self {
        Self::new(magnitude, Unit::new_dimensionless())
    }

    /// Return true if this quantity has no associated units.
    pub fn is_dimensionless(&self) -> bool {
        self.unit.is_dimensionless()
    }

    pub fn get_dimensionality(&self, registry: &Registry) -> Option<Dimensionality> {
        self.unit.get_dimensionality(registry)
    }

    /// Return the underlying value of this quantity, converted to target units.
    pub fn m_as(&self, unit: &Unit, factor: Option<f64>) -> SmootResult<S> {
        let factor = factor.unwrap_or(1.0);
        if self.unit.eq(unit) {
            Ok(self.magnitude.convert(factor))
        } else {
            Ok(self
                .unit
                .conversion_factor(unit)
                .map(|f| self.magnitude.convert(f * factor))?)
        }
    }

    /// Return a new quantity whose value is converted to the target units.
    pub fn to(&self, unit: &Unit, factor: Option<f64>) -> SmootResult<Quantity<N, S>> {
        let mut q = self.clone();
        q.ito(unit, factor)?;
        Ok(q)
    }

    /// In-place value conversion to the target units.
    ///
    /// Parameters
    /// ----------
    /// unit : The target unit to convert to.
    /// factor : Optional additional multiplicative factor to apply.
    pub fn ito(&mut self, unit: &Unit, factor: Option<f64>) -> SmootResult<()> {
        if self.unit.eq(unit) {
            return Ok(());
        }

        if !self.unit.is_compatible_with(unit) {
            return Err(SmootError::IncompatibleUnitTypes(
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into()),
                unit.get_units_string(true)
                    .unwrap_or("dimensionless".into()),
            ));
        }
        self.convert_to(unit, factor)
    }

    /// In-place simplification of the units associated with this quantity (e.g. "meter * km -> meter ** 2").
    /// The underlying value may change depending on any unit conversions that occur during reduction.
    pub fn ito_reduced_units(&mut self) {
        let factor = self.unit.reduce();
        self.magnitude.iconvert(factor);
    }

    /// In-place conversion of this quantity into root units.
    ///
    /// Examples
    /// --------
    /// 1 kilometer / hour -> 3.6 meter / second
    pub fn ito_root_units(&mut self, registry: &Registry) {
        let factor = self.unit.ito_root_units(registry);
        self.magnitude.iconvert(factor);
    }

    pub fn isqrt(&mut self) -> SmootResult<()> {
        self.magnitude = self.magnitude.sqrt();
        self.unit.isqrt()
    }

    fn convert_to(&mut self, unit: &Unit, factor: Option<f64>) -> SmootResult<()> {
        let factor = self.unit.conversion_factor(unit)? * factor.unwrap_or(1.0);
        self.magnitude.iconvert(factor);
        self.unit = unit.clone();
        Ok(())
    }
}

/// Parsing
impl Quantity<f64, f64> {
    /// Parse an expression into a quantity (e.g. "1 meter / second").
    pub fn parse(registry: &Registry, s: &str) -> SmootResult<Self> {
        expression_parser::expression(s, registry)
            .map(|mut q| {
                q.ito_reduced_units();
                q
            })
            .map_err(|_| SmootError::ExpressionError(format!("Invalid quantity expression {}", s)))
    }
}

impl<N: Number, S: Storage<N>> Trigonometry for Quantity<N, S>
where
    S: ConvertMagnitude,
{
    type Output = SmootResult<Quantity<N, S>>;

    fn sin(&self) -> Self::Output {
        if !self.is_dimensionless() {
            return Err(SmootError::InvalidOperation(format!(
                "sin expected dimensionless quantity but got {}",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into())
            )));
        }
        Ok(Quantity::new(self.magnitude.sin(), self.unit.clone()))
    }

    fn cos(&self) -> Self::Output {
        if !self.is_dimensionless() {
            return Err(SmootError::InvalidOperation(format!(
                "cos expected dimensionless quantity but got {}",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into())
            )));
        }
        Ok(Quantity::new(self.magnitude.cos(), self.unit.clone()))
    }

    fn tan(&self) -> Self::Output {
        if !self.is_dimensionless() {
            return Err(SmootError::InvalidOperation(format!(
                "tan expected dimensionless quantity but got {}",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into())
            )));
        }
        Ok(Quantity::new(self.magnitude.tan(), self.unit.clone()))
    }

    fn arcsin(&self) -> Self::Output {
        if !self.is_dimensionless() {
            return Err(SmootError::InvalidOperation(format!(
                "arcsin expected dimensionless quantity but got {}",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into())
            )));
        }
        Ok(Quantity::new(self.magnitude.arcsin(), self.unit.clone()))
    }

    fn arccos(&self) -> Self::Output {
        if !self.is_dimensionless() {
            return Err(SmootError::InvalidOperation(format!(
                "arccos expected dimensionless quantity but got {}",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into())
            )));
        }
        Ok(Quantity::new(self.magnitude.arccos(), self.unit.clone()))
    }

    fn arctan(&self) -> Self::Output {
        if !self.is_dimensionless() {
            return Err(SmootError::InvalidOperation(format!(
                "arctan expected dimensionless quantity but got {}",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into())
            )));
        }
        Ok(Quantity::new(self.magnitude.arctan(), self.unit.clone()))
    }
}

impl<N: Number, S: Storage<N>> LogExp for Quantity<N, S>
where
    S: ConvertMagnitude,
{
    type Output = SmootResult<Quantity<N, S>>;

    fn ln(&self) -> Self::Output {
        if !self.is_dimensionless() {
            return Err(SmootError::InvalidOperation(format!(
                "ln expected dimensionless quantity but got {}",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into())
            )));
        }
        Ok(Quantity::new(self.magnitude.ln(), self.unit.clone()))
    }

    fn log10(&self) -> Self::Output {
        if !self.is_dimensionless() {
            return Err(SmootError::InvalidOperation(format!(
                "log10 expected dimensionless quantity but got {}",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into())
            )));
        }
        Ok(Quantity::new(self.magnitude.log10(), self.unit.clone()))
    }

    fn log2(&self) -> Self::Output {
        if !self.is_dimensionless() {
            return Err(SmootError::InvalidOperation(format!(
                "log2 expected dimensionless quantity but got {}",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into())
            )));
        }
        Ok(Quantity::new(self.magnitude.log2(), self.unit.clone()))
    }

    fn exp(&self) -> Self::Output {
        if !self.is_dimensionless() {
            return Err(SmootError::InvalidOperation(format!(
                "exp expected dimensionless quantity but got {}",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into())
            )));
        }
        Ok(Quantity::new(self.magnitude.exp(), self.unit.clone()))
    }
}

/// Scalar operators
impl<N: Number> Quantity<N, N>
where
    N: ConvertMagnitude,
{
    /// Approximate equality between quantities, accounting for floating point imprecision.
    pub fn approx_eq(&self, other: &Self) -> bool {
        // Early outs to avoid copies for speed
        if !self.unit.is_compatible_with(&other.unit) {
            return false;
        }
        if self.unit == other.unit {
            return self.magnitude.approx_eq(other.magnitude);
        }

        let mut q1 = self.clone();
        q1.ito_reduced_units();
        let mut q2 = other.clone();
        q2.ito_reduced_units();

        let factor = q2.unit.conversion_factor(&q1.unit).unwrap();
        q2.magnitude.iconvert(factor);
        q1.magnitude.approx_eq(q2.magnitude)
    }
}

/// Array operators
impl<N: Number> Quantity<N, ArrayD<N>>
where
    N: ConvertMagnitude,
    ArrayD<N>: ConvertMagnitude,
{
    /// Approximate equality between quantities, accounting for floating point imprecision.
    pub fn approx_eq(&self, other: &Self) -> SmootResult<ArrayD<bool>> {
        self.require_same_shape(other)?;

        Ok(other.unit.conversion_factor(&self.unit).ok().map_or_else(
            // If units cannot be converted, all entries false.
            || {
                Array::from_shape_vec(self.magnitude.shape(), vec![false; self.magnitude.len()])
                    .unwrap()
            },
            |factor| {
                Zip::from(&self.magnitude)
                    .and(&other.magnitude)
                    .map_collect(|&a, &b| a.approx_eq(b.convert(factor)))
            },
        ))
    }

    /// Matrix multiplication. Works like numpy.dot, but only supports 1D and 2D arrays.
    pub fn dot(self, other: &Self) -> SmootResult<Self> {
        // Because we work with dynamically dimensioned arrays, we need to explicitly handle each combination of dimensions
        // explicitly. This is because numpy arrays passed from numpy are fully dynamic, whereas ndarray encodes dimensionality
        // into the rust type system.
        let shape1 = self.magnitude.shape();
        let shape2 = other.magnitude.shape();
        let mut magnitude = if shape1.len() == 1 && shape2.len() == 1 {
            // For vector-vector multiplication, we compute a scalar result. We have to wrap it in a dynamic array
            // for consistency.
            let m1 = self.magnitude.into_dimensionality::<Ix1>().unwrap();
            let m2 = other
                .magnitude
                .clone()
                .into_dimensionality::<Ix1>()
                .unwrap();
            let magnitude = m1.dot(&m2);
            Array::from_shape_vec(vec![1], vec![magnitude]).unwrap()
        } else if shape1.len() == 2 && shape2.len() == 1 {
            let m1 = self.magnitude.into_dimensionality::<Ix2>().unwrap();
            let m2 = other
                .magnitude
                .clone()
                .into_dimensionality::<Ix1>()
                .unwrap();
            let magnitude = m1.dot(&m2);
            magnitude.into_dyn()
        } else if shape1.len() == 2 && shape2.len() == 2 {
            let m1 = self.magnitude.into_dimensionality::<Ix2>().unwrap();
            let m2 = other
                .magnitude
                .clone()
                .into_dimensionality::<Ix2>()
                .unwrap();
            let magnitude = m1.dot(&m2);
            magnitude.into_dyn()
        } else {
            return Err(SmootError::InvalidArrayDimensionality(format!(
                "Matrix multiplication not supported between dimensions {:?} and {:?}",
                shape1, shape2
            )));
        };

        // Rescale the result to simplified units.
        let mut unit = self.unit * other.unit.clone();
        let factor = unit.simplify(false);
        magnitude.iconvert(factor);

        Ok(Self::new(magnitude, unit))
    }

    fn require_same_shape(&self, other: &Self) -> SmootResult<()> {
        if self.magnitude.dim() != other.magnitude.dim() {
            Err(SmootError::MismatchedArrayShape(format!(
                "{:?} != {:?}",
                self.magnitude.dim(),
                other.magnitude.dim()
            )))
        } else {
            Ok(())
        }
    }
}

impl<N: Number, S: Storage<N>> Powi for Quantity<N, S>
where
    S: Powi + ConvertMagnitude,
{
    fn powi(self, p: i32) -> Self {
        Quantity::new(self.magnitude.powi(p), self.unit.powi(p))
    }
}

impl<N: Number, S: Storage<N> + PartialOrd> PartialOrd for Quantity<N, S>
where
    S: ConvertMagnitude,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Convert to same units before comparison.
        other
            .m_as(&self.unit, None)
            .ok()
            .and_then(|o| self.magnitude.partial_cmp(&o))
    }
}

// Modulo quantities
impl<N: Number, S: Storage<N>> Rem for Quantity<N, S>
where
    S: RemAssign + ConvertMagnitude,
{
    type Output = SmootResult<Self>;

    fn rem(mut self, rhs: Self) -> Self::Output {
        self.magnitude %= self
            .unit
            .conversion_factor(&rhs.unit)
            .map(|f| rhs.magnitude.convert(f))?;
        Ok(self)
    }
}
impl<N: Number, S: Storage<N>> Rem<&Quantity<N, S>> for Quantity<N, S>
where
    S: RemAssign + ConvertMagnitude,
{
    type Output = SmootResult<Self>;

    fn rem(mut self, rhs: &Self) -> Self::Output {
        self.magnitude %= rhs
            .unit
            .conversion_factor(&self.unit)
            .map(|f| rhs.magnitude.convert(f))?;
        Ok(self)
    }
}
// Array modulo
impl<N: Number> Rem<ArrayD<N>> for Quantity<N, ArrayD<N>>
where
    N: Rem<Output = N>,
    ArrayD<N>: ConvertMagnitude,
{
    type Output = Quantity<N, ArrayD<N>>;

    fn rem(self, rhs: ArrayD<N>) -> Self::Output {
        let magnitude = Zip::from(&self.magnitude)
            .and(&rhs)
            .map_collect(|&a, &b| a % b);
        Quantity::new(magnitude, self.unit.clone())
    }
}
impl<N: Number> Rem<&Quantity<N, ArrayD<N>>> for &Quantity<N, ArrayD<N>>
where
    N: Rem<Output = N> + ConvertMagnitude,
    ArrayD<N>: ConvertMagnitude,
{
    type Output = SmootResult<Quantity<N, ArrayD<N>>>;

    fn rem(self, rhs: &Quantity<N, ArrayD<N>>) -> Self::Output {
        let factor = rhs.unit.conversion_factor(&self.unit)?;
        let magnitude = Zip::from(&self.magnitude)
            .and(&rhs.magnitude)
            .map_collect(|&a, &b| a % b.convert(factor));
        Ok(Quantity::new(magnitude, self.unit.clone()))
    }
}

//==================================================
// Arithmetic operators
//==================================================
/// Multiply quantities
impl<N: Number, S: Storage<N>> MulAssign for Quantity<N, S>
where
    S: MulAssign + ConvertMagnitude,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.unit *= &rhs.unit;
        let factor = self.unit.simplify(false);
        self.magnitude *= rhs.magnitude.convert(factor);
    }
}
impl<N: Number, S: Storage<N>> MulAssign<&Quantity<N, S>> for Quantity<N, S>
where
    S: MulAssign + ConvertMagnitude,
{
    fn mul_assign(&mut self, rhs: &Self) {
        self.unit *= &rhs.unit;
        let factor = self.unit.simplify(false);
        self.magnitude *= rhs.magnitude.convert(factor);
    }
}
impl<N: Number, S: Storage<N>> Mul for Quantity<N, S>
where
    S: MulAssign + ConvertMagnitude,
{
    type Output = Quantity<N, S>;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}
impl<N: Number, S: Storage<N>> Mul<&Quantity<N, S>> for &Quantity<N, S>
where
    S: MulAssign + ConvertMagnitude,
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
impl<N: Number> Mul<&Quantity<N, ArrayD<N>>> for Quantity<N, ArrayD<N>>
where
    ArrayD<N>: ConvertMagnitude,
{
    type Output = Self;

    fn mul(mut self, rhs: &Quantity<N, ArrayD<N>>) -> Self::Output {
        self.unit.mul_assign(&rhs.unit);
        let factor = self.unit.simplify(false);
        self.magnitude = self.magnitude * rhs.magnitude.convert(factor);
        self
    }
}

/// Add quantities
impl<N: Number, S: Storage<N>> AddAssign for Quantity<N, S>
where
    S: AddAssign + ConvertMagnitude,
{
    fn add_assign(&mut self, rhs: Self) {
        let factor = rhs
            .unit
            .conversion_factor(&self.unit)
            .expect("Incompatible units used in add_assign");
        self.magnitude += rhs.magnitude.convert(factor);
    }
}
impl<N: Number, S: Storage<N>> AddAssign<&Quantity<N, S>> for Quantity<N, S>
where
    S: AddAssign + ConvertMagnitude,
{
    fn add_assign(&mut self, rhs: &Self) {
        let factor = rhs
            .unit
            .conversion_factor(&self.unit)
            .expect("Incompatible units used in add_assign");
        self.magnitude += rhs.magnitude.convert(factor);
    }
}
impl<N: Number, S: Storage<N>> Add for Quantity<N, S>
where
    S: AddAssign + ConvertMagnitude,
{
    type Output = SmootResult<Quantity<N, S>>;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.magnitude += rhs
            .unit
            .conversion_factor(&self.unit)
            .map(|f| rhs.magnitude.convert(f))?;
        Ok(self)
    }
}
impl<N: Number, S: Storage<N>> Add<&Quantity<N, S>> for &Quantity<N, S>
where
    S: AddAssign + ConvertMagnitude,
{
    type Output = SmootResult<Quantity<N, S>>;

    fn add(self, rhs: &Quantity<N, S>) -> Self::Output {
        let mut new = self.clone();
        new.magnitude += rhs
            .unit
            .conversion_factor(&self.unit)
            .map(|f| rhs.magnitude.convert(f))?;
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
            return Err(SmootError::InvalidOperation(format!(
                "Invalid Quantity operation '{}' + '{}'",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into()),
                "dimensionaless",
            )));
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
            return Err(SmootError::InvalidOperation(format!(
                "Invalid Quantity operation '{}' + '{}'",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into()),
                "dimensionaless",
            )));
        }
        self.magnitude = self.magnitude + rhs;
        Ok(self)
    }
}
impl<N: Number> Add<&Quantity<N, ArrayD<N>>> for Quantity<N, ArrayD<N>>
where
    ArrayD<N>: ConvertMagnitude,
{
    type Output = SmootResult<Quantity<N, ArrayD<N>>>;

    fn add(mut self, rhs: &Quantity<N, ArrayD<N>>) -> Self::Output {
        let factor = rhs.unit.conversion_factor(&self.unit)?;
        self.magnitude = self.magnitude + rhs.magnitude.convert(factor);
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
    S: SubAssign + ConvertMagnitude,
{
    fn sub_assign(&mut self, rhs: Self) {
        let factor = rhs
            .unit
            .conversion_factor(&self.unit)
            .expect("Incompatible units used in sub_assign");
        self.magnitude -= rhs.magnitude.convert(factor);
    }
}
impl<N: Number, S: Storage<N>> SubAssign<&Quantity<N, S>> for Quantity<N, S>
where
    S: SubAssign + ConvertMagnitude,
{
    fn sub_assign(&mut self, rhs: &Self) {
        let factor = rhs
            .unit
            .conversion_factor(&self.unit)
            .expect("Incompatible units used in sub_assign");
        self.magnitude -= rhs.magnitude.convert(factor);
    }
}
impl<N: Number, S: Storage<N>> Sub for Quantity<N, S>
where
    S: SubAssign + ConvertMagnitude,
{
    type Output = SmootResult<Quantity<N, S>>;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.magnitude -= rhs
            .unit
            .conversion_factor(&self.unit)
            .map(|f| rhs.magnitude.convert(f))?;
        Ok(self)
    }
}
impl<N: Number, S: Storage<N>> Sub<&Quantity<N, S>> for &Quantity<N, S>
where
    S: SubAssign + ConvertMagnitude,
{
    type Output = SmootResult<Quantity<N, S>>;

    fn sub(self, rhs: &Quantity<N, S>) -> Self::Output {
        let mut new = self.clone();
        new.magnitude -= rhs
            .unit
            .conversion_factor(&self.unit)
            .map(|f| rhs.magnitude.convert(f))?;
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
            return Err(SmootError::InvalidOperation(format!(
                "Invalid Quantity operation '{}' - '{}'",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into()),
                "dimensionaless",
            )));
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
            return Err(SmootError::InvalidOperation(format!(
                "Invalid Quantity operation '{}' - '{}'",
                self.unit
                    .get_units_string(true)
                    .unwrap_or("dimensionless".into()),
                "dimensionaless",
            )));
        }
        self.magnitude = self.magnitude - rhs;
        Ok(self)
    }
}
impl<N: Number> Sub<&Quantity<N, ArrayD<N>>> for Quantity<N, ArrayD<N>>
where
    ArrayD<N>: ConvertMagnitude,
{
    type Output = SmootResult<Self>;

    fn sub(mut self, rhs: &Quantity<N, ArrayD<N>>) -> Self::Output {
        self.magnitude = self.magnitude
            - self
                .unit
                .conversion_factor(&rhs.unit)
                .map(|f| rhs.magnitude.convert(f))?;
        Ok(self)
    }
}

/// Divide quantities
impl<N: Number, S: Storage<N>> Div for Quantity<N, S>
where
    S: DivAssign + ConvertMagnitude,
{
    type Output = Quantity<N, S>;

    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}
impl<N: Number, S: Storage<N>> Div<&Quantity<N, S>> for &Quantity<N, S>
where
    S: DivAssign + ConvertMagnitude,
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
    S: DivAssign + ConvertMagnitude,
{
    fn div_assign(&mut self, rhs: Self) {
        self.magnitude /= rhs.magnitude;
        self.unit.div_assign(&rhs.unit);
        let factor = self.unit.simplify(false);
        self.magnitude.iconvert(factor);
    }
}
impl<N: Number, S: Storage<N>> DivAssign<&Quantity<N, S>> for Quantity<N, S>
where
    S: DivAssign + ConvertMagnitude,
{
    fn div_assign(&mut self, rhs: &Self) {
        self.magnitude /= rhs.magnitude.clone();
        self.unit.div_assign(&rhs.unit);
        let factor = self.unit.simplify(false);
        self.magnitude.iconvert(factor);
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
impl<N: Number> Div<&Quantity<N, ArrayD<N>>> for Quantity<N, ArrayD<N>>
where
    ArrayD<N>: ConvertMagnitude,
{
    type Output = Self;

    fn div(mut self, rhs: &Quantity<N, ArrayD<N>>) -> Self::Output {
        self.magnitude = self.magnitude / rhs.magnitude.clone();
        self.unit.div_assign(&rhs.unit);
        let factor = self.unit.simplify(false);
        self.magnitude.iconvert(factor);
        self
    }
}

// impl From<Quantity<f64, f64>> for Quantity<i64, i64> {
//     fn from(value: Quantity<f64, f64>) -> Self {
//         Self::new(value.magnitude as i64, value.unit)
//     }
// }
// impl From<Quantity<i64, i64>> for Quantity<f64, f64> {
//     fn from(value: Quantity<i64, i64>) -> Self {
//         Self::new(value.magnitude as f64, value.unit)
//     }
// }
// impl From<Quantity<i64, ArrayD<i64>>> for Quantity<f64, ArrayD<f64>> {
//     fn from(value: Quantity<i64, ArrayD<i64>>) -> Self {
//         let m = value.magnitude.mapv(|v| v as f64);
//         Self::new(m, value.unit)
//     }
// }

//==================================================
// Unit
//
// Defined here because these operators produce
// Quantity values.
//==================================================
// unit * N
impl<N: Number> Mul<N> for Unit
where
    N: ConvertMagnitude,
{
    type Output = Quantity<N, N>;

    fn mul(self, rhs: N) -> Self::Output {
        Quantity::new(rhs, self)
    }
}
impl<N: Number> Mul<N> for &Unit
where
    N: ConvertMagnitude,
{
    type Output = Quantity<N, N>;

    fn mul(self, rhs: N) -> Self::Output {
        self.clone() * rhs
    }
}

// unit / N
impl<N: Number> Div<N> for Unit
where
    N: ConvertMagnitude,
{
    type Output = Quantity<N, N>;

    fn div(self, rhs: N) -> Self::Output {
        Quantity::new(N::one() / rhs, self)
    }
}
impl<N: Number> Div<N> for &Unit
where
    N: ConvertMagnitude,
{
    type Output = Quantity<N, N>;

    fn div(self, rhs: N) -> Self::Output {
        self.clone() / rhs
    }
}
// N / unit
macro_rules! unit_div {
    ($type: ident) => {
        impl Div<Unit> for $type {
            type Output = Quantity<$type, $type>;

            fn div(self, mut rhs: Unit) -> Self::Output {
                rhs.ipowi(-1);
                Quantity::new(self, rhs)
            }
        }
        impl Div<&Unit> for $type {
            type Output = Quantity<$type, $type>;

            fn div(self, rhs: &Unit) -> Self::Output {
                Quantity::new(self, rhs.powi(-1))
            }
        }
    };
}
unit_div!(f64);
// unit_div!(i64);

//==================================================
// Unit tests
//==================================================
#[cfg(test)]
mod test_quantity {
    use std::{
        hash::{DefaultHasher, Hasher},
        sync::LazyLock,
    };

    use crate::{assert_is_close, base_unit::BaseUnit, test_utils::TEST_REGISTRY};

    use super::*;

    static UNIT_METER: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("meter").expect("No unit 'meter'"));
    static UNIT_KILOMETER: LazyLock<&BaseUnit> = LazyLock::new(|| {
        TEST_REGISTRY
            .get_unit("kilometer")
            .expect("No unit 'kilometer'")
    });
    static UNIT_SECOND: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("second").expect("No unit 'second`'"));
    static UNIT_NEWTON: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("newton").expect("No unit 'newton`'"));
    static UNIT_JOULE: LazyLock<&BaseUnit> =
        LazyLock::new(|| TEST_REGISTRY.get_unit("joule").expect("No unit 'joule`'"));

    #[test]
    fn test_quantity_ito() -> SmootResult<()> {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let kilometer = Unit::new(vec![BaseUnit::clone(&UNIT_KILOMETER)], vec![]);

        let mut q = Quantity::new(1.0, meter);

        q.ito(&kilometer, None)?;

        assert_is_close!(q.magnitude, 1.0 / 1000.0);
        assert_eq!(q.unit, kilometer);

        Ok(())
    }

    #[test]
    fn test_quantity_ito_incompatible_units() {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let second = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);
        let mut q = Quantity::new(1.0, meter);

        assert!(q.ito(&second, None).is_err());
    }

    #[test]
    fn test_quantity_to() -> SmootResult<()> {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let kilometer = Unit::new(vec![BaseUnit::clone(&UNIT_KILOMETER)], vec![]);

        let q = Quantity::new(1.0, meter);

        let q_converted = q.to(&kilometer, None)?;

        assert_is_close!(q_converted.magnitude, 1.0 / 1000.0);
        assert_eq!(q_converted.unit, kilometer);

        Ok(())
    }

    #[test]
    fn test_quantity_to_incompatible_units() {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let second = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);
        let q = Quantity::new(1.0, meter);

        assert!(q.to(&second, None).is_err());
    }

    #[test]
    fn test_quantity_m_as() -> SmootResult<()> {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let kilometer = Unit::new(vec![BaseUnit::clone(&UNIT_KILOMETER)], vec![]);
        let q = Quantity::new(1.0, meter);

        let magnitude = q.m_as(&kilometer, None)?;

        assert_is_close!(magnitude, 1.0 / 1000.0);

        Ok(())
    }

    #[test]
    fn test_quantity_m_as_incompatible_units() {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let second = Unit::new(vec![BaseUnit::clone(&UNIT_SECOND)], vec![]);
        let q = Quantity::new(1.0, meter);

        assert!(q.m_as(&second, None).is_err());
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

        let q_scaled = q1 * &q2;

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

        let q_scaled = (q1 + &q2)?;

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

        let q = (q1 - &q2)?;

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

        let q = q1 / &q2;

        q.magnitude.for_each(|x| {
            assert_is_close!(*x, 0.5);
        });
    }

    /// Division with a unit results in a quantity
    #[test]
    fn test_unit_div_into_quantity() {
        let meter = Unit::new(vec![BaseUnit::clone(&UNIT_METER)], vec![]);
        let expected = Quantity::new(1.0, Unit::new(vec![], vec![BaseUnit::clone(&UNIT_METER)]));
        assert_eq!(1.0 / &meter, expected);
        assert_eq!(1.0 / meter, expected);
    }

    #[test]
    fn test_ito_root_unit() {
        let mut q = Quantity::new(
            1.0,
            Unit::new(vec![UNIT_JOULE.clone()], vec![UNIT_NEWTON.clone()]),
        );
        let expected = Quantity::new(1.0, Unit::new(vec![UNIT_METER.clone()], vec![]));
        q.ito_root_units(&TEST_REGISTRY);
        assert_eq!(q, expected);
    }

    #[test]
    fn test_hash() {
        let q1 = Quantity::new(
            1.0,
            Unit::new(
                vec![BaseUnit::clone(&UNIT_METER)],
                vec![BaseUnit::clone(&UNIT_SECOND)],
            ),
        );
        assert_eq!(hash(&q1), hash(&q1.clone()));

        let q2 = q1.clone().powi(-1);
        assert_ne!(hash(&q1), hash(&q2));
    }

    fn hash<T: Hash>(val: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        val.hash(&mut hasher);
        hasher.finish()
    }
}
