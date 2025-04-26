use std::{
    hash::{Hash as StdHash, Hasher},
    marker::PhantomData,
};

use ndarray::ArrayD;

pub trait Hash {
    fn hash<H: Hasher>(&self, state: &mut H);
}

impl Hash for f64 {
    /// Floating point hash implementation that treats all NaN values as equal.
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use a consistent NaN value to keep hashes identical.
        const NAN: u64 = f64::NAN.to_bits();

        // Branchless replacement of NaN values with a consistent NaN to stabilize hashing.
        let mask = self.is_nan() as u64;
        let bits = NAN * mask + self.to_bits() * (1 - mask);
        StdHash::hash(&bits, state);
    }
}

impl<T: Hash> Hash for Option<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

impl<T: Hash> Hash for Vec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.iter().for_each(|item| item.hash(state));
        self.len().hash(state);
    }
}

impl<T: Hash> Hash for PhantomData<T> {
    fn hash<H: Hasher>(&self, _state: &mut H) {
        // noop
    }
}

impl<T: Hash> Hash for ArrayD<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.iter().for_each(|n| n.hash(state));
        self.dim().hash(state);
        self.strides().hash(state);
    }
}

/// Hash implementation for non-parameterized types.
macro_rules! impl_hash {
    ($type: ident) => {
        impl Hash for $type {
            fn hash<H: Hasher>(&self, state: &mut H) {
                StdHash::hash(self, state);
            }
        }
    };
}
impl_hash!(i8);
impl_hash!(i32);
impl_hash!(u64);
impl_hash!(String);

//==================================================
// Unit tests
//==================================================
#[cfg(test)]
mod test_hash {
    use std::hash::DefaultHasher;

    use super::*;

    /// All NaN values have the same hash.
    #[test]
    fn test_nan_stable_hashing() {
        let nan1 = f64::NAN;
        let nan2 = f64::NAN + 1.0;
        assert!(nan2.is_nan());
        assert_eq!(hash(&nan1), hash(&nan2));
    }

    fn hash<T: Hash>(val: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        val.hash(&mut hasher);
        hasher.finish()
    }
}
