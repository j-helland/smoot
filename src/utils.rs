use num_traits::{Float, PrimInt};
use numpy::ndarray::ArrayD;

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

pub trait Powf {
    fn powf(self, p: f64) -> Self;
}
impl Powf for f64 {
    fn powf(self, p: f64) -> Self {
        self.powf(p)
    }
}
impl Powf for i64 {
    fn powf(self, p: f64) -> Self {
        (self as f64).powf(p) as i64
    }
}
impl Powf for ArrayD<f64> {
    fn powf(self, p: f64) -> Self {
        self.mapv(|f| f.powf(p))
    }
}
impl Powf for ArrayD<i64> {
    fn powf(self, p: f64) -> Self {
        self.mapv(|i| i.powf(p))
    }
}

pub trait Powi {
    fn powi(self, p: i32) -> Self;
}
impl Powi for f64 {
    fn powi(self, p: i32) -> Self {
        self.powi(p)
    }
}
impl Powi for i64 {
    fn powi(self, p: i32) -> Self {
        // Any negative power is an integer reciprocal, which must be zero.
        // Mask these cases out.
        let neg_mask = !(p as i64).signed_shr(63);
        neg_mask & self.pow(p as u32)
    }
}
impl Powi for ArrayD<f64> {
    fn powi(self, p: i32) -> Self {
        self.mapv(|f| f.powi(p))
    }
}
impl Powi for ArrayD<i64> {
    fn powi(self, p: i32) -> Self {
        self.mapv(|i| i.powi(p))
    }
}

pub trait Floor {
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    fn convert(&self, factor: f64) -> Self;

    fn iconvert(&mut self, factor: f64);
}
impl ConvertMagnitude for i64 {
    fn convert(&self, factor: f64) -> i64 {
        if factor < 1.0 {
            self / (1.0 / factor) as i64
        } else {
            self * factor as i64
        }
    }

    fn iconvert(&mut self, factor: f64) {
        if factor < 1.0 {
            *self /= (1.0 / factor) as i64;
        } else {
            *self *= factor as i64;
        }
    }
}
impl ConvertMagnitude for f64 {
    fn convert(&self, factor: f64) -> f64 {
        self * factor
    }

    fn iconvert(&mut self, factor: f64) {
        *self *= factor;
    }
}
impl ConvertMagnitude for ArrayD<i64> {
    fn convert(&self, factor: f64) -> ArrayD<i64> {
        if factor < 1.0 {
            self.clone() / (1.0 / factor) as i64
        } else {
            self.clone() * factor as i64
        }
    }

    fn iconvert(&mut self, factor: f64) {
        if factor < 1.0 {
            *self /= (1.0 / factor) as i64;
        } else {
            *self *= factor as i64;
        }
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
