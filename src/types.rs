use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use numpy::ndarray::ScalarOperand;

use crate::utils::{ApproxEq, Powf};

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
impl DataOps for i64 {}

pub trait Number:
    DataOps + FromPrimitive + ToPrimitive + Zero + One + ApproxEq + Powf + Debug
{
}
impl Number for f64 {}
impl Number for i64 {}
