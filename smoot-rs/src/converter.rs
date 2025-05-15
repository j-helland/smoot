use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use bitcode::{Decode, Encode};
use num_traits::FromPrimitive;

use crate::base_unit::BaseUnit;
use crate::hash::Hash;

#[derive(Encode, Decode, Clone, Copy, Debug, PartialEq)]
pub enum Converter {
    Multiplicative,
    Offset,
}

impl Converter {
    pub fn convert_from<N, S>(self, value: &mut S, from: &BaseUnit)
    where
        N: FromPrimitive,
        S: MulAssign<N> + AddAssign<N>,
    {
        match self {
            Converter::Multiplicative => {
                *value *= N::from_f64(from.get_multiplier()).unwrap();
            }
            Converter::Offset => {
                *value *= N::from_f64(from.get_multiplier()).unwrap();
                *value += N::from_f64(from.offset).unwrap();
            }
        }
    }

    pub fn convert_to<N, S>(self, value: &mut S, to: &BaseUnit)
    where
        N: FromPrimitive,
        S: DivAssign<N> + SubAssign<N>,
    {
        match self {
            Converter::Multiplicative => {
                *value /= N::from_f64(to.get_multiplier()).unwrap();
            }
            Converter::Offset => {
                *value -= N::from_f64(to.offset).unwrap();
                *value /= N::from_f64(to.get_multiplier()).unwrap();
            }
        }
    }
}

impl Hash for Converter {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&core::mem::discriminant(self), state);
    }
}
