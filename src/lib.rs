use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use bitcode::{Decode, Encode};
use numpy::{ndarray::ArrayD, PyArray, PyArrayDyn, PyArrayMethods};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
};
use pyo3::{
    pyfunction, pymodule,
    types::{PyDict, PyModule},
    Bound,
};

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hasher};

use crate::hash::Hash;
use crate::quantity::Quantity;
use crate::registry::REGISTRY;
use crate::utils::{ConvertMagnitude, Floor, Powf, RoundDigits};

mod base_unit;
mod error;
mod hash;
mod parser;
mod quantity;
mod registry;
mod types;
mod unit;
mod utils;

#[cfg(test)]
mod test_utils;

create_exception!("smoot.smoot", SmootError, PyException);

impl From<error::SmootError> for PyErr {
    fn from(value: error::SmootError) -> Self {
        SmootError::new_err(value.to_string())
    }
}

#[pyfunction]
fn get_registry_size() -> usize {
    REGISTRY.len()
}

#[pyfunction]
fn get_all_registry_keys() -> Vec<String> {
    REGISTRY.all_keys()
}

macro_rules! create_unit_type {
    ($name_unit: ident, $base_type: ident) => {
        #[pyclass(module = "smoot.smoot")]
        #[derive(Clone)]
        struct $name_unit {
            inner: unit::Unit,
        }
        #[pymethods]
        impl $name_unit {
            #[staticmethod]
            fn parse(expression: &str) -> PyResult<(f64, Self)> {
                let (factor, inner) = unit::Unit::parse(&REGISTRY, expression)?;
                Ok((factor, Self { inner }))
            }

            fn is_compatible_with(&self, other: &Self) -> bool {
                self.inner.is_compatible_with(&other.inner)
            }

            #[getter(dimensionless)]
            fn dimensionless(&self) -> bool {
                self.inner.is_dimensionless()
            }

            #[getter(dimensionality)]
            fn dimensionality(&self) -> Option<HashMap<String, f64>> {
                self.inner.get_dimensionality().map(|dims| dims.0)
            }

            fn to_root_units(&self) -> Self {
                let mut new = self.clone();
                let _ = new.inner.ito_root_units();
                new
            }

            fn ito_root_units(&mut self) {
                self.inner.ito_root_units();
            }

            fn __hash__(&self) -> u64 {
                let mut hasher = DefaultHasher::new();
                self.inner.hash(&mut hasher);
                hasher.finish()
            }

            fn __str__(&self) -> String {
                self.inner
                    .get_units_string(true)
                    .unwrap_or_else(|| "dimensionless".into())
            }

            fn __repr__(&self) -> String {
                self.__str__()
            }

            fn __eq__(&self, other: &Self) -> bool {
                self.inner == other.inner
            }

            fn __mul__(&self, other: &Self) -> Self {
                let mut new = self.clone();
                new.inner *= &other.inner;
                let _ = new.inner.reduce();
                new
            }

            fn __imul__(&mut self, other: &Self) {
                self.inner *= &other.inner;
                let _ = self.inner.reduce();
            }

            fn __truediv__(&self, other: &Self) -> Self {
                let mut new = self.clone();
                new.inner /= &other.inner;
                let _ = new.inner.reduce();
                new
            }

            fn __itruediv__(&mut self, other: &Self) {
                self.inner /= &other.inner;
                let _ = self.inner.reduce();
            }

            fn __pow__(&self, p: f64, _modulo: Option<i64>) -> Self {
                let mut new = self.clone();
                new.inner.ipowf(p);
                let _ = new.inner.reduce();
                new
            }

            fn __ipow__(&mut self, p: f64, _modulo: Option<i64>) {
                self.inner.ipowf(p);
                let _ = self.inner.reduce();
            }
        }
    };
}

/// Create a quantity with a given underlying type.
macro_rules! create_quantity_type {
    ($name_unit: ident, $name_quantity: ident, $base_type: ident, $storage_type: ident) => {
        #[pyclass(module = "smoot.smoot")]
        #[derive(Clone)]
        struct $name_quantity {
            inner: quantity::Quantity<$base_type, $storage_type>,
        }
        #[pymethods]
        impl $name_quantity {
            #[new]
            #[pyo3(signature = (value, units=None, factor=None))]
            fn py_new(
                value: $storage_type,
                units: Option<&$name_unit>,
                factor: Option<f64>,
            ) -> PyResult<Self> {
                let value = factor.map(|f| value.convert(f)).unwrap_or(value);
                let inner = units
                    .map(|u| quantity::Quantity::new(value, u.inner.clone()))
                    .unwrap_or_else(|| quantity::Quantity::new_dimensionless(value));
                Ok(Self { inner })
            }

            #[staticmethod]
            fn parse(expression: &str) -> PyResult<Self> {
                let inner = quantity::Quantity::parse(&REGISTRY, expression)?.into();
                Ok(Self { inner })
            }

            #[getter(dimensionless)]
            fn dimensionless(&self) -> bool {
                self.inner.is_dimensionless()
            }

            #[getter(unitless)]
            fn unitless(&self) -> bool {
                self.inner.is_dimensionless()
            }

            #[getter(dimensionality)]
            fn dimensionality(&self) -> Option<HashMap<String, f64>> {
                self.inner.get_dimensionality().map(|dims| dims.0)
            }

            #[getter(m)]
            fn m(&self) -> $storage_type {
                self.inner.magnitude
            }

            #[getter(magnitude)]
            fn magnitude(&self) -> $storage_type {
                self.m()
            }

            #[getter(u)]
            fn u(&self) -> $name_unit {
                $name_unit {
                    inner: self.inner.unit.clone(),
                }
            }

            #[getter(units)]
            fn units(&self) -> $name_unit {
                self.u()
            }

            #[pyo3(signature = (unit, factor=None))]
            fn to(&self, unit: &$name_unit, factor: Option<f64>) -> PyResult<$name_quantity> {
                let inner = self.inner.to(&unit.inner, factor)?;
                Ok(Self { inner })
            }

            #[pyo3(signature = (unit, factor=None))]
            fn ito(&mut self, unit: &$name_unit, factor: Option<f64>) -> PyResult<()> {
                Ok(self.inner.ito(&unit.inner, factor)?)
            }

            #[pyo3(signature = (unit, factor=None))]
            fn m_as(&self, unit: &$name_unit, factor: Option<f64>) -> PyResult<$storage_type> {
                Ok(self.inner.m_as(&unit.inner, factor)?)
            }

            fn to_root_units(&self) -> Self {
                let mut new = self.clone();
                new.inner.ito_root_units();
                new
            }

            fn ito_root_units(&mut self) {
                self.inner.ito_root_units();
            }

            fn mul_scalar(&self, scalar: $base_type) -> Self {
                Self {
                    inner: &self.inner * scalar,
                }
            }

            //==================================================
            // standard dunder methods
            //==================================================
            fn __str__(&self) -> String {
                format!(
                    "{} {}",
                    self.inner.magnitude,
                    self.inner
                        .unit
                        .get_units_string(false)
                        .unwrap_or("dimensionless".into())
                )
            }

            fn __hash__(&self) -> u64 {
                let mut hasher = DefaultHasher::new();
                self.inner.hash(&mut hasher);
                hasher.finish()
            }

            fn __copy__(&self) -> Self {
                self.clone()
            }

            fn __deepcopy__(&self, _memo: Bound<'_, PyDict>) -> Self {
                self.clone()
            }

            //==================================================
            // operators
            //==================================================
            fn __eq__(&self, other: &Self) -> bool {
                self.inner.approx_eq(&other.inner)
            }

            fn __ne__(&self, other: &Self) -> bool {
                !self.__eq__(other)
            }

            fn __lt__(&self, other: &Self) -> bool {
                self.inner.partial_cmp(&other.inner) == Some(std::cmp::Ordering::Less)
            }

            fn __le__(&self, other: &Self) -> bool {
                match self.inner.partial_cmp(&other.inner) {
                    Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Less) => true,
                    _ => false,
                }
            }

            fn __gt__(&self, other: &Self) -> bool {
                self.inner.partial_cmp(&other.inner) == Some(std::cmp::Ordering::Greater)
            }

            fn __ge__(&self, other: &Self) -> bool {
                match self.inner.partial_cmp(&other.inner) {
                    Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Greater) => true,
                    _ => false,
                }
            }

            fn __add__(&self, other: &Self) -> PyResult<Self> {
                let inner = (&self.inner + &other.inner)?;
                Ok(Self { inner })
            }

            fn __sub__(&self, other: &Self) -> PyResult<Self> {
                let inner = (&self.inner - &other.inner)?;
                Ok(Self { inner })
            }

            fn __mul__(&self, other: &Self) -> Self {
                Self {
                    inner: &self.inner * &other.inner,
                }
            }

            fn __truediv__(&self, other: &Self) -> Self {
                Self {
                    inner: &self.inner / &other.inner,
                }
            }

            fn __floordiv__(&self, other: &Self) -> Self {
                let mut new = self.__truediv__(other);
                new.inner.magnitude = new.inner.magnitude.floor();
                new
            }

            fn __mod__(&self, other: &Self) -> PyResult<Self> {
                let inner = (self.inner.clone() % &other.inner)?;
                Ok(Self { inner })
            }

            fn __pow__(&self, other: f64, _modulo: Option<i64>) -> Self {
                Self {
                    inner: self.inner.clone().powf(other),
                }
            }

            fn __neg__(&self) -> Self {
                Self {
                    inner: -self.inner.clone(),
                }
            }

            #[pyo3(signature = (ndigits=None))]
            fn __round__(&self, ndigits: Option<i32>) -> Self {
                let mut new = self.clone();
                new.inner.magnitude = new.inner.magnitude.round_digits(ndigits.unwrap_or(0));
                new
            }

            fn __trunc__(&self) -> i64 {
                self.inner.magnitude.trunc() as i64
            }

            fn __floor__(&self) -> Self {
                let mut new = self.clone();
                new.inner.magnitude = new.inner.magnitude.floor();
                new
            }

            fn __ceil__(&self) -> Self {
                let mut new = self.clone();
                new.inner.magnitude = new.inner.magnitude.ceil();
                new
            }

            fn __abs__(&self) -> Self {
                let mut new = self.clone();
                new.inner.magnitude = new.inner.magnitude.abs();
                new
            }

            fn __float__(&self) -> f64 {
                self.inner.magnitude as f64
            }

            fn __int__(&self) -> i64 {
                self.inner.magnitude as i64
            }

            //==================================================
            // pickle support
            //==================================================
            fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
                self.inner =
                    bitcode::decode(state).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(())
            }

            fn __getstate__(&self) -> Vec<u8> {
                bitcode::encode(&self.inner)
            }

            fn __getnewargs__(&self) -> ($storage_type,) {
                (self.m(),)
            }
        }
    };
}

/// Used for pickling numpy arrays.
#[derive(Encode, Decode)]
struct ArrayQuantityStorage<N> {
    dims: Vec<usize>,
    data: Vec<N>,
    unit: unit::Unit,
}

/// Create a numpy array version of a unitary quantity type.
/// Expects a Unit type generated by create_quantity_type.
macro_rules! create_array_quantity_type {
    ($name: ident, $unit_type: ident, $base_type: ident) => {
        #[pyclass(module = "smoot.smoot")]
        #[derive(Clone)]
        struct $name {
            inner: quantity::Quantity<$base_type, ArrayD<$base_type>>,
        }
        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (value, units=None, factor=None))]
            fn py_new(
                value: Bound<'_, PyArrayDyn<$base_type>>,
                units: Option<&$unit_type>,
                factor: Option<f64>,
            ) -> PyResult<Self> {
                let mut value = value.to_owned_array();
                if let Some(factor) = factor {
                    value.iconvert(factor);
                }

                let inner = if let Some(u) = units {
                    Quantity::new(value, u.inner.clone())
                } else {
                    Quantity::new_dimensionless(value)
                };
                Ok(Self { inner })
            }

            #[staticmethod]
            fn new(arr: Bound<'_, PyArrayDyn<$base_type>>, unit: &$unit_type) -> Self {
                Self {
                    inner: quantity::Quantity::new(arr.to_owned_array(), unit.inner.clone()),
                }
            }

            #[getter(dimensionless)]
            fn dimensionless(&self) -> bool {
                self.inner.is_dimensionless()
            }

            #[getter(unitless)]
            fn unitless(&self) -> bool {
                self.inner.is_dimensionless()
            }

            #[getter(dimensionality)]
            fn dimensionality(&self) -> Option<HashMap<String, f64>> {
                self.inner.get_dimensionality().map(|dims| dims.0)
            }

            #[getter(m)]
            fn m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<$base_type>>> {
                Ok(PyArray::from_array(py, &self.inner.magnitude))
            }

            #[getter(magnitude)]
            fn magnitude<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArrayDyn<$base_type>>> {
                self.m(py)
            }

            #[getter(u)]
            fn u(&self) -> $unit_type {
                $unit_type {
                    inner: self.inner.unit.clone(),
                }
            }

            #[getter(units)]
            fn units(&self) -> $unit_type {
                self.u()
            }

            #[pyo3(signature = (unit, factor=None))]
            fn to(&self, unit: &$unit_type, factor: Option<f64>) -> PyResult<Self> {
                let inner = self.inner.to(&unit.inner, factor)?;
                Ok(Self { inner })
            }

            #[pyo3(signature = (unit, factor=None))]
            fn ito(&mut self, unit: &$unit_type, factor: Option<f64>) -> PyResult<()> {
                self.inner.ito(&unit.inner, factor)?;
                Ok(())
            }

            #[pyo3(signature = (unit, factor=None))]
            fn m_as<'py>(
                &self,
                py: Python<'py>,
                unit: &$unit_type,
                factor: Option<f64>,
            ) -> PyResult<Bound<'py, PyArrayDyn<$base_type>>> {
                let arr = self.inner.m_as(&unit.inner, factor)?;
                Ok(PyArray::from_array(py, &arr))
            }

            fn to_root_units(&self) -> Self {
                let mut new = self.clone();
                new.inner.ito_root_units();
                new
            }

            fn ito_root_units(&mut self) {
                self.inner.ito_root_units();
            }

            //==================================================
            // standard dunder methods
            //==================================================
            fn __str__(&self) -> String {
                format!(
                    "{} {}",
                    self.inner.magnitude,
                    self.inner
                        .unit
                        .get_units_string(false)
                        .unwrap_or("dimensionless".into())
                )
            }

            fn __hash__(&self) -> u64 {
                let mut hasher = DefaultHasher::new();
                self.inner.hash(&mut hasher);
                hasher.finish()
            }

            fn __copy__(&self) -> Self {
                self.clone()
            }

            fn __deepcopy__(&self, _memo: Bound<'_, PyDict>) -> Self {
                self.clone()
            }

            //==================================================
            // operators
            //==================================================
            fn __eq__<'py>(
                &self,
                py: Python<'py>,
                other: &Self,
            ) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
                Ok(self
                    .inner
                    .approx_eq(&other.inner)
                    .map(|arr| PyArray::from_array(py, &arr))?)
            }

            fn __add__(&self, other: &Self) -> PyResult<Self> {
                let inner = (self.inner.clone() + &other.inner)?;
                Ok(Self { inner })
            }

            fn __sub__(&self, other: &Self) -> PyResult<Self> {
                let inner = (self.inner.clone() - &other.inner)?;
                Ok(Self { inner })
            }

            fn __mul__(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.clone() * &other.inner,
                }
            }

            fn __truediv__(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.clone() / &other.inner,
                }
            }

            fn __floordiv__(&self, other: &Self) -> Self {
                let mut new = self.__truediv__(other);
                new.inner.magnitude = new.inner.magnitude.floor();
                new
            }

            fn __mod__(&self, other: &Self) -> PyResult<Self> {
                let inner = (&self.inner % &other.inner)?;
                Ok(Self { inner })
            }

            fn __pow__(&self, other: f64, _modulo: Option<i64>) -> Self {
                Self {
                    inner: self.inner.clone().powf(other),
                }
            }

            /// Element-wise power
            fn arr_pow(&self, other: &Self) -> PyResult<Self> {
                let inner = self.inner.arr_pow(&other.inner)?;
                Ok(Self { inner })
            }

            fn __matmul__(&self, other: &Self) -> PyResult<Self> {
                let inner = self.inner.clone().dot(&other.inner)?;
                Ok(Self { inner })
            }

            fn __neg__(&self) -> Self {
                Self {
                    inner: -self.inner.clone(),
                }
            }

            fn __abs__(&self) -> Self {
                let magnitude = self.inner.magnitude.mapv(|f| f.abs());
                Self {
                    inner: Quantity::new(magnitude, self.inner.unit.clone()),
                }
            }

            //==================================================
            // pickle support
            //==================================================
            fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
                let s: ArrayQuantityStorage<$base_type> =
                    bitcode::decode(state).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                self.inner.magnitude = ArrayD::from_shape_vec(s.dims, s.data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                self.inner.unit = s.unit;
                Ok(())
            }

            fn __getstate__(&self) -> Vec<u8> {
                let s = ArrayQuantityStorage {
                    dims: self.inner.magnitude.shape().to_vec(),
                    data: self.inner.magnitude.as_slice().unwrap().into(),
                    unit: self.inner.unit.clone(),
                };
                self.inner.magnitude.as_slice();
                bitcode::encode(&s)
            }

            fn __getnewargs__<'py>(
                &self,
                py: Python<'py>,
            ) -> (Bound<'py, PyArrayDyn<$base_type>>,) {
                (PyArray::from_array(py, &self.inner.magnitude),)
            }
        }
    };
}

create_unit_type!(Unit, f64);
create_quantity_type!(Unit, F64Quantity, f64, f64);
create_array_quantity_type!(ArrayF64Quantity, Unit, f64);

//==================================================
// Cross-type unit operators
//
// These can't be defined using dunder methods,
// which are strongly-typed for pyo3.
//==================================================
#[pyfunction]
fn mul_unit(num: f64, unit: &Unit) -> F64Quantity {
    F64Quantity {
        inner: &unit.inner * num,
    }
}

#[pyfunction]
fn div_unit(unit: &Unit, num: f64) -> F64Quantity {
    F64Quantity {
        inner: &unit.inner / num,
    }
}

#[pyfunction]
fn rdiv_unit(num: f64, unit: &Unit) -> F64Quantity {
    F64Quantity {
        inner: num / &unit.inner,
    }
}

// create_quantity_type!(Unit, I64Quantity, i64, i64);
// create_array_quantity_type!(ArrayI64Quantity, Unit, i64);

// #[pyfunction]
// fn i64_to_f64_quantity(q: &I64Quantity) -> F64Quantity {
//     F64Quantity { inner: q.inner.clone().into() }
// }

// #[pyfunction]
// fn array_i64_to_f64_quantity(q: &ArrayI64Quantity) -> ArrayF64Quantity {
//     ArrayF64Quantity { inner: q.inner.clone().into() }
// }

#[pymodule]
fn smoot(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core unit type
    m.add_class::<Unit>()?;

    // float-backed types
    m.add_class::<F64Quantity>()?;
    m.add_class::<ArrayF64Quantity>()?;

    // // int backed types
    // m.add_class::<I64Quantity>()?;
    // m.add_class::<ArrayI64Quantity>()?;

    let _ = m.add_function(wrap_pyfunction!(get_registry_size, m)?);
    let _ = m.add_function(wrap_pyfunction!(get_all_registry_keys, m)?);
    // let _ = m.add_function(wrap_pyfunction!(i64_to_f64_quantity, m)?);
    // let _ = m.add_function(wrap_pyfunction!(array_i64_to_f64_quantity, m)?);
    let _ = m.add_function(wrap_pyfunction!(mul_unit, m)?);
    let _ = m.add_function(wrap_pyfunction!(div_unit, m)?);
    let _ = m.add_function(wrap_pyfunction!(rdiv_unit, m)?);

    Ok(())
}
