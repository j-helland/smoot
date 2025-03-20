use std::ops::{Div, DivAssign, Mul, MulAssign};

use bitcode::{Decode, Encode};
// use smartstring::alias::String;

use crate::error::{SmootError, SmootResult};
use crate::types::Number;

pub type DimensionType = u64;
pub const DIMENSIONLESS: DimensionType = 0;

#[derive(Encode, Decode, Clone, Eq, PartialEq, Debug)]
pub struct BaseUnit<N: Number> {
    pub name: String,
    pub multiplier: N,
    // TODO(jwh): remove
    pub power: Option<N>,
    pub unit_type: DimensionType,
    pub dimensionality: Vec<N>,
}

fn get_dimensionality<N: Number>(unit_type: DimensionType) -> Vec<N> {
    let mut dimensionality = Vec::new();
    let mut bits = unit_type;
    while bits > 0 {
        let idx = bits.trailing_zeros();
        let diff = idx as usize - dimensionality.len();
        if diff > 0 {
            dimensionality.extend((0..diff).map(|_| N::zero()));
        }
        dimensionality.push(N::one());
        bits &= !(1 << idx);
    }
    dimensionality
}

impl<N: Number> BaseUnit<N> {
    pub fn new(name: String, multiplier: N, unit_type: DimensionType) -> Self {
        Self {
            name,
            multiplier,
            power: None,
            unit_type,
            dimensionality: get_dimensionality(unit_type),
        }
    }

    pub fn new_constant(multiplier: N) -> Self {
        Self {
            name: String::new(),
            multiplier,
            power: None,
            unit_type: DIMENSIONLESS,
            dimensionality: vec![],
        }
    }

    pub fn update_dimensionality(&mut self, mut unit_type: DimensionType) {
        while unit_type > 0 {
            let idx = unit_type.trailing_zeros() as usize;
            if idx < self.dimensionality.len() {
                self.dimensionality[idx] = N::one();
            } else {
                if idx - self.dimensionality.len() > 0 {
                    self.dimensionality
                        .extend((0..idx - self.dimensionality.len()).map(|_| N::zero()));
                }
                self.dimensionality.push(N::one());
            }
            unit_type &= !(1 << idx);
        }
    }

    pub fn mul_dimensionality(&mut self, n: N) {
        self.dimensionality.iter_mut().for_each(|d| *d *= n);
    }

    pub fn get_multiplier(&self) -> N {
        self.power
            .map(|p| self.multiplier.powf(p))
            .unwrap_or(self.multiplier)
    }

    pub fn conversion_factor(&self, target: &Self) -> SmootResult<N> {
        if self.unit_type != target.unit_type {
            return Err(SmootError::IncompatibleUnitTypes(
                self.name.clone(),
                target.name.clone(),
            ));
        }

        // convert to the base unit, then to the target unit
        Ok(self.get_multiplier() / target.get_multiplier())
    }

    pub fn ipowf(&mut self, n: N) {
        self.power = self.power.or(Some(N::one())).map(|p| p * n);
        self.mul_dimensionality(n);
    }

    pub fn powf(&self, n: N) -> Self {
        let mut new = self.clone();
        new.ipowf(n);
        new
    }
}

impl<N: Number> Mul for BaseUnit<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.power.is_some() || rhs.power.is_some() {
            panic!("Cannot multiply base units with exponents");
        }
        let mut new = self.clone();
        new *= rhs;
        new
    }
}

impl<N: Number> MulAssign for BaseUnit<N> {
    fn mul_assign(&mut self, rhs: Self) {
        if self.power.is_some() || rhs.power.is_some() {
            panic!("Cannot multiply base units with exponents");
        }
        self.multiplier *= rhs.multiplier;
        self.unit_type |= rhs.unit_type;

        self.dimensionality.extend(
            (0..rhs
                .dimensionality
                .len()
                .saturating_sub(self.dimensionality.len()))
                .map(|_| N::zero()),
        );
        for i in 0..self.dimensionality.len().min(rhs.dimensionality.len()) {
            if self.dimensionality[i] > N::zero() || rhs.dimensionality[i] > N::zero() {
                self.dimensionality[i] += rhs.dimensionality[i];
            }
        }
    }
}

impl<N: Number> Div for BaseUnit<N> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        if self.power.is_some() || rhs.power.is_some() {
            panic!("Cannot divide base units with exponents");
        }
        let mut new = self.clone();
        new /= rhs;
        new
    }
}

impl<N: Number> DivAssign for BaseUnit<N> {
    fn div_assign(&mut self, rhs: Self) {
        if self.power.is_some() || rhs.power.is_some() {
            panic!("Cannot divide base units with exponents");
        }
        self.multiplier /= rhs.multiplier;
        self.unit_type |= rhs.unit_type;

        self.dimensionality.extend(
            (0..rhs
                .dimensionality
                .len()
                .saturating_sub(self.dimensionality.len()))
                .map(|_| N::zero()),
        );
        for i in 0..self.dimensionality.len().min(rhs.dimensionality.len()) {
            if self.dimensionality[i] > N::zero() || rhs.dimensionality[i] > N::zero() {
                self.dimensionality[i] -= rhs.dimensionality[i];
            }
        }
    }
}

#[cfg(test)]
mod test_base_unit {
    use test_case::case;

    use crate::assert_is_close;

    use super::*;

    #[case(
        BaseUnit::new("left".into(), 2.0, 1),
        BaseUnit::new("right".into(), 3.0, 1 << 1),
        BaseUnit::new("left".into(), 6.0, 1 | (1 << 1))
        ; "Multipliers multiply and unit types combine"
    )]
    fn test_mul(left: BaseUnit<f64>, right: BaseUnit<f64>, expected: BaseUnit<f64>) {
        assert_eq!(left * right, expected);
    }

    // #[test]
    // fn test() {
    //     let u1 = BaseUnit::new("1".to_string(), 1.0, 1);
    //     let u2 = BaseUnit::new("2".to_string(), 1.0, 3);
    //     println!("{:?}", u1 * u2);
    //     assert!(false);
    // }

    // #[test]
    // /// The conversion factor between compatible units is computed correctly.
    // fn test_conversion_factor() -> SmootResult<()> {
    //     // Given two units with the same type
    //     let u1 = BaseUnit {
    //         name: "u1".to_string(),
    //         multiplier: 1.0,
    //         power: None,
    //         unit_type: 0,
    //     };
    //     let u2 = BaseUnit {
    //         name: "u2".to_string(),
    //         multiplier: 2.0,
    //         power: None,
    //         unit_type: 0,
    //     };

    //     // Then a self conversion factor is 1.0
    //     assert_eq!(u1.conversion_factor(&u1)?, 1.0);

    //     // The conversion factor and reciprocal match.
    //     assert_is_close!(u1.conversion_factor(&u2)?, 0.5);
    //     assert_is_close!(u2.conversion_factor(&u1)?, 2.0);

    //     Ok(())
    // }

    // #[test]
    // /// Trying to convert between incompatible units is an error.
    // fn test_conversion_factor_incompatible_types() {
    //     // Given two units with disparate types
    //     let u1 = BaseUnit {
    //         name: "u1".to_string(),
    //         multiplier: 1.0,
    //         power: None,
    //         unit_type: 0,
    //     };
    //     let u2 = BaseUnit {
    //         name: "u2".to_string(),
    //         multiplier: 1.0,
    //         power: None,
    //         unit_type: 0,
    //     };

    //     // Then the result is an error
    //     let result = u1.conversion_factor(&u2);
    //     assert!(result.is_err());
    // }
}
