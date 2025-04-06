use bitcode::{Decode, Encode};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use numpy::{ndarray::ArrayD, PyArray, PyArrayDyn, PyArrayMethods};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use pyo3::{pyfunction, pymodule, types::PyModule, Bound};

use crate::quantity::Quantity;
use crate::registry::REGISTRY;
use crate::unit::Unit;
use crate::utils::{Ceil, Floor, Powf, RoundDigits, Truncate};

mod base_unit;
mod error;
mod parser;
mod quantity;
mod registry;
mod types;
mod unit;
mod utils;

#[cfg(test)]
mod test_utils;

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
        struct $name_unit {
            inner: Unit<$base_type>,
        }
        #[pymethods]
        impl $name_unit {
            #[staticmethod]
            fn parse(expression: &str) -> PyResult<Self> {
                let inner = Unit::parse(&REGISTRY, expression)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }

            fn __str__(&self) -> String {
                self.inner
                    .get_units_string()
                    .unwrap_or_else(|| "dimensionless".into())
            }

            fn __eq__(&self, other: &Self) -> bool {
                self.inner == other.inner
            }

            fn __mul__(&self, other: &Self) -> Self {
                let mut new = Self {
                    inner: self.inner.clone(),
                };
                new.inner *= &other.inner;
                let _ = new.inner.reduce();
                new
            }

            fn __imul__(&mut self, other: &Self) {
                self.inner *= &other.inner;
                let _ = self.inner.reduce();
            }

            fn __truediv__(&self, other: &Self) -> Self {
                let mut new = Self {
                    inner: self.inner.clone(),
                };
                new.inner /= &other.inner;
                let _ = new.inner.reduce();
                new
            }

            fn __itruediv__(&mut self, other: &Self) {
                self.inner /= &other.inner;
                let _ = self.inner.reduce();
            }

            fn __pow__(&self, p: f64, _modulo: Option<i64>) -> Self {
                let mut new = Self {
                    inner: self.inner.clone(),
                };
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
            #[pyo3(signature = (value, units=None))]
            fn py_new(value: $storage_type, units: Option<&$name_unit>) -> PyResult<Self> {
                let inner = units
                    .map(|u| quantity::Quantity::new(value, u.inner.clone()))
                    .unwrap_or_else(|| quantity::Quantity::new_dimensionless(value));
                Ok(Self { inner })
            }

            #[staticmethod]
            fn parse(expression: &str) -> PyResult<Self> {
                let quantity = quantity::Quantity::parse(&REGISTRY, expression)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self {
                    inner: quantity.into(),
                })
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

            #[getter(unit)]
            fn units(&self) -> $name_unit {
                self.u()
            }

            fn to(&self, unit: &$name_unit) -> PyResult<$name_quantity> {
                let new_q = self
                    .inner
                    .to(&unit.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner: new_q })
            }

            fn ito(&mut self, unit: &$name_unit) -> PyResult<()> {
                self.inner
                    .ito(&unit.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }

            fn m_as(&self, unit: &$name_unit) -> PyResult<$storage_type> {
                self.inner
                    .m_as(&unit.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }

            fn mul_scalar(&self, scalar: $base_type) -> Self {
                Self {
                    inner: &self.inner * scalar,
                }
            }

            // standard dunder methods
            fn __str__(&self) -> String {
                format!(
                    "{} {}",
                    self.inner.magnitude,
                    self.inner
                        .unit
                        .get_units_string()
                        .unwrap_or("dimensionless".into())
                )
            }

            fn __hash__(&self) -> u64 {
                todo!();
            }

            fn __copy__(&self) -> Self {
                self.clone()
            }

            fn __deepcopy__(&self) -> Self {
                self.clone()
            }

            //==================================================
            // operators
            //==================================================
            fn __eq__(&self, other: &Self) -> bool {
                self.inner.partial_cmp(&other.inner) == Some(std::cmp::Ordering::Equal)
            }

            fn __ne__(&self, other: &Self) -> bool {
                self.inner.partial_cmp(&other.inner) != Some(std::cmp::Ordering::Equal)
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
                (&self.inner + &other.inner)
                    .map(|inner| Self { inner })
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }

            fn __sub__(&self, other: &Self) -> PyResult<Self> {
                (&self.inner - &other.inner)
                    .map(|inner| Self { inner })
                    .map_err(|e| PyValueError::new_err(e.to_string()))
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
                let mut new = Self {
                    inner: self.inner.clone(),
                };
                new.inner.magnitude = new.inner.magnitude.round_digits(ndigits.unwrap_or(0));
                new
            }

            fn __trunc__(&self) -> i64 {
                self.inner.magnitude.trunc() as i64
            }

            fn __floor__(&self) -> Self {
                let mut new = Self {
                    inner: self.inner.clone(),
                };
                new.inner.magnitude = new.inner.magnitude.floor();
                new
            }

            fn __ceil__(&self) -> Self {
                let mut new = Self {
                    inner: self.inner.clone(),
                };
                new.inner.magnitude = new.inner.magnitude.ceil();
                new
            }

            fn __abs__(&self) -> Self {
                let mut new = Self {
                    inner: self.inner.clone(),
                };
                new.inner.magnitude = new.inner.magnitude.abs();
                new
            }

            fn __float__(&self) -> f64 {
                self.inner.magnitude as f64
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
    unit: Unit<f64>,
}

/// Create a numpy array version of a unitary quantity type.
/// Expects a Unit type generated by create_quantity_type.
macro_rules! create_array_quantity_type {
    ($name: ident, $unit_type: ident, $base_type: ident) => {
        #[pyclass(module = "smoot.smoot")]
        struct $name {
            inner: quantity::Quantity<$base_type, ArrayD<$base_type>>,
        }
        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (value, units=None))]
            fn py_new(
                value: Bound<'_, PyArrayDyn<$base_type>>,
                units: Option<&$unit_type>,
            ) -> PyResult<Self> {
                let inner = units
                    .map(|u| quantity::Quantity::new(value.to_owned_array(), u.inner.clone()))
                    .unwrap_or_else(|| {
                        quantity::Quantity::new_dimensionless(value.to_owned_array())
                    });
                Ok(Self { inner })
            }

            #[staticmethod]
            fn new(arr: Bound<'_, PyArrayDyn<$base_type>>, unit: &$unit_type) -> Self {
                Self {
                    inner: quantity::Quantity::new(arr.to_owned_array(), unit.inner.clone()),
                }
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

            #[getter(unit)]
            fn unit(&self) -> $unit_type {
                self.u()
            }

            fn to(&self, unit: &$unit_type) -> PyResult<Self> {
                let new_q = self
                    .inner
                    .to(&unit.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner: new_q })
            }

            fn ito(&mut self, unit: &$unit_type) -> PyResult<()> {
                self.inner
                    .ito(&unit.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }

            fn m_as<'py>(
                &self,
                py: Python<'py>,
                unit: &$unit_type,
            ) -> PyResult<Bound<'py, PyArrayDyn<$base_type>>> {
                self.inner
                    .m_as(&unit.inner)
                    .map(|arr| PyArray::from_array(py, &arr))
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }

            //==================================================
            // operators
            //==================================================
            fn __add__(&self, other: &Self) -> PyResult<Self> {
                (self.inner.clone() + &other.inner)
                    .map(|inner| Self { inner })
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }

            fn __sub__(&self, other: &Self) -> PyResult<Self> {
                (self.inner.clone() - &other.inner)
                    .map(|inner| Self { inner })
                    .map_err(|e| PyValueError::new_err(e.to_string()))
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

            fn __pow__(&self, other: f64, _modulo: Option<i64>) -> Self {
                Self {
                    inner: self.inner.clone().powf(other),
                }
            }

            fn __matmul__(&self, other: &Self) -> PyResult<Self> {
                self.inner
                    .clone()
                    .dot(&other.inner)
                    .map(|inner| Self { inner })
                    .map_err(|e| PyValueError::new_err(e.to_string()))
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

create_unit_type!(F64Unit, f64);

create_quantity_type!(F64Unit, F64Quantity, f64, f64);
create_array_quantity_type!(ArrayF64Quantity, F64Unit, f64);

create_quantity_type!(F64Unit, I64Quantity, i64, i64);
create_array_quantity_type!(ArrayI64Quantity, F64Unit, i64);

#[pymodule]
fn smoot(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core unit type
    m.add_class::<F64Unit>()?;

    // float-backed types
    m.add_class::<F64Quantity>()?;
    m.add_class::<ArrayF64Quantity>()?;

    // int backed types
    m.add_class::<I64Quantity>()?;
    m.add_class::<ArrayI64Quantity>()?;

    let _ = m.add_function(wrap_pyfunction!(get_registry_size, m)?);
    let _ = m.add_function(wrap_pyfunction!(get_all_registry_keys, m)?);

    Ok(())
}
