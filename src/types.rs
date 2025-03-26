use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::hash::Hash;
use std::fmt::Debug;

use num_traits::{Float, One, Pow, Zero};
use numpy::ndarray::ScalarOperand;

pub trait DataOps:
    Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Copy
    + ScalarOperand
    + PartialOrd
{
}
impl DataOps for f64 {}

pub trait Number: DataOps + From<f64> + From<i32> + Float + Debug {}
impl Number for f64 {}
