use ndarray::ArrayD;
use num_traits::{Float, PrimInt};

/// Compute approximate equality between floating point numbers using a
/// relative tolerance bound.
///
/// Return
/// ------
/// true if the difference between values is within tolerance.
pub fn float_eq_rel<N: Float>(a: N, b: N, max_diff: N) -> bool {
    let largest = a.abs().max(b.abs());
    (a - b).abs() <= (largest * max_diff)
}

pub trait ApproxEq {
    fn approx_eq(self, x: Self) -> bool;
}
impl ApproxEq for f64 {
    fn approx_eq(self, x: Self) -> bool {
        float_eq_rel(self, x, 1e-6)
    }
}
impl ApproxEq for i64 {
    fn approx_eq(self, x: Self) -> bool {
        self == x
    }
}
impl<T: ApproxEq + Copy> ApproxEq for &Vec<T> {
    fn approx_eq(self, x: Self) -> bool {
        if self.len() != x.len() {
            return false;
        }
        self.iter().zip(x.iter()).all(|(&a, &b)| a.approx_eq(b))
    }
}

pub trait Powi {
    type Output;

    fn powi(self, p: i32) -> Self::Output;
}
impl Powi for f64 {
    type Output = f64;

    fn powi(self, p: i32) -> Self::Output {
        self.powi(p)
    }
}
impl Powi for i64 {
    type Output = i64;

    fn powi(self, p: i32) -> Self::Output {
        // Any negative power is an integer reciprocal, which must be zero.
        // Mask these cases out.
        let neg_mask = !i64::from(p).signed_shr(63);
        neg_mask & self.pow(p.unsigned_abs())
    }
}
impl Powi for ArrayD<f64> {
    type Output = ArrayD<f64>;

    fn powi(self, p: i32) -> Self::Output {
        self.mapv(|f| f.powi(p))
    }
}
impl Powi for ArrayD<i64> {
    type Output = ArrayD<i64>;

    fn powi(self, p: i32) -> Self::Output {
        self.mapv(|i| i.powi(p))
    }
}

pub trait Sqrt {
    #[must_use]
    fn sqrt(&self) -> Self;
}
impl Sqrt for f64 {
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }
}
impl<N: Sqrt + Copy> Sqrt for ArrayD<N> {
    fn sqrt(&self) -> Self {
        self.mapv(|f| f.sqrt())
    }
}

pub trait LogExp {
    type Output;

    fn ln(&self) -> Self::Output;
    fn log10(&self) -> Self::Output;
    fn log2(&self) -> Self::Output;
    fn exp(&self) -> Self::Output;
}
impl LogExp for f64 {
    type Output = f64;

    fn ln(&self) -> Self::Output {
        f64::ln(*self)
    }

    fn log10(&self) -> Self::Output {
        f64::log10(*self)
    }

    fn log2(&self) -> Self::Output {
        f64::log2(*self)
    }

    fn exp(&self) -> Self::Output {
        f64::exp(*self)
    }
}
impl<N> LogExp for ArrayD<N>
where
    N: LogExp<Output = N> + Copy,
{
    type Output = ArrayD<N>;

    fn ln(&self) -> Self::Output {
        self.mapv(|f| f.ln())
    }

    fn log10(&self) -> Self::Output {
        self.mapv(|f| f.log10())
    }

    fn log2(&self) -> Self::Output {
        self.mapv(|f| f.log2())
    }

    fn exp(&self) -> Self::Output {
        self.mapv(|f| f.exp())
    }
}

pub trait Trigonometry {
    type Output;

    fn sin(&self) -> Self::Output;
    fn cos(&self) -> Self::Output;
    fn tan(&self) -> Self::Output;
    fn arcsin(&self) -> Self::Output;
    fn arccos(&self) -> Self::Output;
    fn arctan(&self) -> Self::Output;
}
impl Trigonometry for f64 {
    type Output = f64;

    fn sin(&self) -> Self::Output {
        f64::sin(*self)
    }

    fn cos(&self) -> Self::Output {
        f64::cos(*self)
    }

    fn tan(&self) -> Self::Output {
        f64::tan(*self)
    }

    fn arcsin(&self) -> Self::Output {
        f64::asin(*self)
    }

    fn arccos(&self) -> Self::Output {
        f64::acos(*self)
    }

    fn arctan(&self) -> Self::Output {
        f64::atan(*self)
    }
}

impl<N: Trigonometry<Output = N> + Copy> Trigonometry for ArrayD<N> {
    type Output = ArrayD<N>;

    fn sin(&self) -> Self::Output {
        self.mapv(|f| f.sin())
    }

    fn cos(&self) -> Self::Output {
        self.mapv(|f| f.cos())
    }

    fn tan(&self) -> Self::Output {
        self.mapv(|f| f.tan())
    }

    fn arcsin(&self) -> Self::Output {
        self.mapv(|f| f.arcsin())
    }

    fn arccos(&self) -> Self::Output {
        self.mapv(|f| f.arccos())
    }

    fn arctan(&self) -> Self::Output {
        self.mapv(|f| f.arctan())
    }
}

pub trait Floor {
    #[must_use]
    fn floor(self) -> Self;
}
impl Floor for f64 {
    fn floor(self) -> Self {
        self.floor()
    }
}
impl Floor for i64 {
    fn floor(self) -> Self {
        self
    }
}
impl Floor for ArrayD<f64> {
    fn floor(self) -> Self {
        self.mapv(f64::floor)
    }
}
impl Floor for ArrayD<i64> {
    fn floor(self) -> Self {
        self
    }
}

pub trait Ceil {
    #[must_use]
    fn ceil(self) -> Self;
}
impl Ceil for f64 {
    fn ceil(self) -> Self {
        self.ceil()
    }
}
impl Ceil for i64 {
    fn ceil(self) -> Self {
        self
    }
}
impl Ceil for ArrayD<f64> {
    fn ceil(self) -> Self {
        self.mapv(f64::ceil)
    }
}
impl Ceil for ArrayD<i64> {
    fn ceil(self) -> Self {
        self
    }
}

pub trait RoundDigits {
    #[must_use]
    fn round_digits(self, ndigits: i32) -> Self;
}
impl RoundDigits for f64 {
    fn round_digits(self, ndigits: i32) -> Self {
        // Rescale to round with expected precision and then scale back.
        let factor = num_traits::Float::powi(10.0, ndigits);
        (self * factor).round() / factor
    }
}
impl RoundDigits for i64 {
    fn round_digits(self, _ndigits: i32) -> Self {
        self
    }
}
impl RoundDigits for ArrayD<f64> {
    fn round_digits(self, ndigits: i32) -> Self {
        let factor = num_traits::Float::powi(10.0, ndigits);
        (self * factor).round() / factor
    }
}
impl RoundDigits for ArrayD<i64> {
    fn round_digits(self, _ndigits: i32) -> Self {
        self
    }
}

pub trait Truncate {
    #[must_use]
    fn trunc(self) -> Self;
}
impl Truncate for f64 {
    fn trunc(self) -> Self {
        self.trunc()
    }
}
impl Truncate for i64 {
    fn trunc(self) -> Self {
        self
    }
}
impl Truncate for ArrayD<f64> {
    fn trunc(self) -> Self {
        self.mapv(f64::trunc)
    }
}
impl Truncate for ArrayD<i64> {
    fn trunc(self) -> Self {
        self
    }
}

/// Convert a magnitude to another scale, handling idiosyncrasies of integer scaling.
pub trait ConvertMagnitude {
    #[must_use]
    fn convert(&self, factor: f64) -> Self;

    fn iconvert(&mut self, factor: f64);
}
impl ConvertMagnitude for f64 {
    fn convert(&self, factor: f64) -> f64 {
        self * factor
    }

    fn iconvert(&mut self, factor: f64) {
        *self *= factor;
    }
}
impl ConvertMagnitude for ArrayD<f64> {
    fn convert(&self, factor: f64) -> ArrayD<f64> {
        self.clone() * factor
    }

    fn iconvert(&mut self, factor: f64) {
        *self *= factor;
    }
}

#[cfg(test)]
mod test_utils {
    use test_case::case;

    use super::*;

    #[test]
    fn test_powi_i64() {
        assert_eq!(0, 1.powi(-1));
        assert_eq!(1, 1.powi(0));
        assert_eq!(1, 1.powi(1));
        assert_eq!(4, 2.powi(2));
    }

    #[case(1.1, 0, 1.0; "round down")]
    #[case(1.11, 1, 1.1; "round down 1 decimal")]
    #[case(1.5, 0, 2.0; "round up")]
    #[case(1.49, 0, 1.0)]
    #[case(1.49, 1, 1.5; "round up 1 decimal")]
    #[case(-1.1, 0, -1.0; "negative")]
    fn test_round_digits_f64(value: f64, ndigits: i32, expected: f64) {
        assert_eq!(expected, value.round_digits(ndigits));
    }
}
