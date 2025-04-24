use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use bitcode::{Decode, Encode};
use numpy::{PyArray, PyArrayDyn, PyArrayMethods, ndarray::ArrayD};
use pyo3::{
    Bound, pyfunction, pymodule,
    types::{PyDict, PyModule},
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
};
use smoot_rs::{
    error,
    hash::Hash,
    quantity,
    registry::Registry,
    unit,
    utils::{ConvertMagnitude, Powf, RoundDigits},
};

use std::sync::Arc;
use std::{
    collections::HashMap,
    ops::Deref,
    sync::{Mutex, MutexGuard},
};
use std::{
    hash::{DefaultHasher, Hasher},
    path::Path,
};

//==================================================
// Error types
//==================================================
create_exception!("smoot.smoot", SmootError, PyException);
create_exception!("smoot.smoot", SmootParseError, PyException);

/// Cannot `impl From<error::SmootError> for PyErr` because of the Rust orphan rule.
fn handle_err<T>(result: Result<T, error::SmootError>) -> PyResult<T> {
    result.map_err(|e| match e {
        error::SmootError::ParseTreeError(msg) => SmootParseError::new_err(msg),
        _ => SmootError::new_err(e.to_string()),
    })
}

//==================================================
// Registry
//==================================================
type InnerRegistry = Arc<Mutex<Registry>>;

#[pyclass(module = "smoot.smoot")]
struct UnitRegistry {
    inner: InnerRegistry,
}

#[pymethods]
impl UnitRegistry {
    #[new]
    fn py_new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Registry::new())),
        }
    }

    #[staticmethod]
    fn new_from_str(data: &str) -> PyResult<Self> {
        let inner = handle_err(Registry::new_from_str(data))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[staticmethod]
    fn new_from_file(path: &str) -> PyResult<Self> {
        let inner = handle_err(Registry::new_from_file(Path::new(path)))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[staticmethod]
    fn new_from_cache_or_file(cache_path: &str, file_path: &str) -> PyResult<Self> {
        let inner = handle_err(Registry::new_from_cache_or_file(
            Path::new(cache_path),
            Path::new(file_path),
        ))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    fn extend(&mut self, data: &str) -> PyResult<()> {
        let mut registry = self.get()?;
        handle_err(registry.extend(data))?;
        Ok(())
    }

    fn get_registry_size(&self) -> PyResult<usize> {
        Ok(self.get()?.len())
    }

    fn get_all_registry_keys(&self) -> PyResult<Vec<String>> {
        Ok(self.get()?.all_keys())
    }

    //==================================================
    // pickle support
    //==================================================
    fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        let registry =
            bitcode::decode(state).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.inner = Arc::new(Mutex::new(registry));
        Ok(())
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        let registry = self.get()?;
        Ok(bitcode::encode(registry.deref()))
    }
}

impl UnitRegistry {
    fn get(&self) -> Result<MutexGuard<Registry>, PyErr> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

impl Deref for UnitRegistry {
    type Target = InnerRegistry;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

//==================================================
// Quantities / Units
//==================================================
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
            fn parse(expression: &str, registry: &UnitRegistry) -> PyResult<(f64, Self)> {
                let registry = registry.get()?;
                let (factor, inner) = handle_err(unit::Unit::parse(&registry, expression))?;
                Ok((factor, Self { inner }))
            }

            fn is_compatible_with(&self, other: &Self) -> bool {
                self.inner.is_compatible_with(&other.inner)
            }

            #[getter(dimensionless)]
            fn dimensionless(&self) -> bool {
                self.inner.is_dimensionless()
            }

            fn dimensionality(
                &self,
                registry: &UnitRegistry,
            ) -> PyResult<Option<HashMap<String, f64>>> {
                let registry = registry.get()?;
                Ok(self.inner.get_dimensionality(&registry).map(|dims| dims.0))
            }

            fn to_root_units(&self, registry: &UnitRegistry) -> PyResult<Self> {
                let registry = registry.get()?;
                let mut new = self.clone();
                let _ = new.inner.ito_root_units(&registry);
                Ok(new)
            }

            fn ito_root_units(&mut self, registry: &UnitRegistry) -> PyResult<()> {
                let registry = registry.get()?;
                self.inner.ito_root_units(&registry);
                Ok(())
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
            fn parse(expression: &str, registry: &UnitRegistry) -> PyResult<Self> {
                let registry = registry.get()?;
                let inner = handle_err(quantity::Quantity::parse(&registry, expression))?.into();
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

            fn dimensionality(
                &self,
                registry: &UnitRegistry,
            ) -> PyResult<Option<HashMap<String, f64>>> {
                let registry = registry.get()?;
                Ok(self.inner.get_dimensionality(&registry).map(|dims| dims.0))
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
                let inner = handle_err(self.inner.to(&unit.inner, factor))?;
                Ok(Self { inner })
            }

            #[pyo3(signature = (unit, factor=None))]
            fn ito(&mut self, unit: &$name_unit, factor: Option<f64>) -> PyResult<()> {
                let result = handle_err(self.inner.ito(&unit.inner, factor))?;
                Ok(result)
            }

            #[pyo3(signature = (unit, factor=None))]
            fn m_as(&self, unit: &$name_unit, factor: Option<f64>) -> PyResult<$storage_type> {
                let result = handle_err(self.inner.m_as(&unit.inner, factor))?;
                Ok(result)
            }

            fn to_root_units(&self, registry: &UnitRegistry) -> PyResult<Self> {
                let registry = registry.get()?;
                let mut new = self.clone();
                new.inner.ito_root_units(&registry);
                Ok(new)
            }

            fn ito_root_units(&mut self, registry: &UnitRegistry) -> PyResult<()> {
                let registry = registry.get()?;
                self.inner.ito_root_units(&registry);
                Ok(())
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
                let inner = handle_err(&self.inner + &other.inner)?;
                Ok(Self { inner })
            }

            fn __sub__(&self, other: &Self) -> PyResult<Self> {
                let inner = handle_err(&self.inner - &other.inner)?;
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
                let inner = handle_err(self.inner.clone() % &other.inner)?;
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
                (self.inner.magnitude,)
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
                    quantity::Quantity::new(value, u.inner.clone())
                } else {
                    quantity::Quantity::new_dimensionless(value)
                };
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

            fn dimensionality(
                &self,
                registry: &UnitRegistry,
            ) -> PyResult<Option<HashMap<String, f64>>> {
                let registry = registry.get()?;
                Ok(self.inner.get_dimensionality(&registry).map(|dims| dims.0))
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
                let inner = handle_err(self.inner.to(&unit.inner, factor))?;
                Ok(Self { inner })
            }

            #[pyo3(signature = (unit, factor=None))]
            fn ito(&mut self, unit: &$unit_type, factor: Option<f64>) -> PyResult<()> {
                handle_err(self.inner.ito(&unit.inner, factor))?;
                Ok(())
            }

            #[pyo3(signature = (unit, factor=None))]
            fn m_as<'py>(
                &self,
                py: Python<'py>,
                unit: &$unit_type,
                factor: Option<f64>,
            ) -> PyResult<Bound<'py, PyArrayDyn<$base_type>>> {
                let arr = handle_err(self.inner.m_as(&unit.inner, factor))?;
                Ok(PyArray::from_array(py, &arr))
            }

            fn to_root_units(&self, registry: &UnitRegistry) -> PyResult<Self> {
                let registry = registry.get()?;
                let mut new = self.clone();
                new.inner.ito_root_units(&registry);
                Ok(new)
            }

            fn ito_root_units(&mut self, registry: &UnitRegistry) -> PyResult<()> {
                let registry = registry.get()?;
                self.inner.ito_root_units(&registry);
                Ok(())
            }

            //==================================================
            // standard dunder methods
            //==================================================
            fn __len__(&self) -> usize {
                self.inner.magnitude.len()
            }

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
                handle_err(
                    self.inner
                        .approx_eq(&other.inner)
                        .map(|arr| PyArray::from_array(py, &arr)),
                )
            }

            fn __add__(&self, other: &Self) -> PyResult<Self> {
                let inner = handle_err(self.inner.clone() + &other.inner)?;
                Ok(Self { inner })
            }

            fn __sub__(&self, other: &Self) -> PyResult<Self> {
                let inner = handle_err(self.inner.clone() - &other.inner)?;
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
                let inner = handle_err(&self.inner % &other.inner)?;
                Ok(Self { inner })
            }

            fn __pow__(&self, other: f64, _modulo: Option<i64>) -> Self {
                Self {
                    inner: self.inner.clone().powf(other),
                }
            }

            /// Element-wise power
            fn arr_pow(&self, other: &Self) -> PyResult<Self> {
                let inner = handle_err(self.inner.arr_pow(&other.inner))?;
                Ok(Self { inner })
            }

            fn __matmul__(&self, other: &Self) -> PyResult<Self> {
                let inner = handle_err(self.inner.clone().dot(&other.inner))?;
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
                    inner: quantity::Quantity::new(magnitude, self.inner.unit.clone()),
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
fn arr_mul_unit(arr: Bound<'_, PyArrayDyn<f64>>, unit: &Unit) -> ArrayF64Quantity {
    // let arr = unsafe { arr.as_array_mut() };
    let arr = arr.to_owned_array();
    ArrayF64Quantity {
        inner: quantity::Quantity::new(arr, unit.inner.clone()),
    }
}

#[pyfunction]
fn div_unit(unit: &Unit, num: f64) -> F64Quantity {
    F64Quantity {
        inner: &unit.inner / num,
    }
}

#[pyfunction]
fn arr_div_unit(unit: &Unit, arr: Bound<'_, PyArrayDyn<f64>>) -> ArrayF64Quantity {
    let arr = arr.to_owned_array();
    ArrayF64Quantity {
        inner: quantity::Quantity::new(1.0 / arr, unit.inner.clone()),
    }
}

#[pyfunction]
fn rdiv_unit(num: f64, unit: &Unit) -> F64Quantity {
    F64Quantity {
        inner: num / &unit.inner,
    }
}

#[pyfunction]
fn arr_rdiv_unit(arr: Bound<'_, PyArrayDyn<f64>>, unit: &Unit) -> ArrayF64Quantity {
    let arr = arr.to_owned_array();
    ArrayF64Quantity {
        inner: quantity::Quantity::new(arr, unit.inner.powi(-1)),
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

//==================================================
// Module construction
//==================================================
#[pymodule]
fn smoot(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Error types
    m.add("SmootError", m.py().get_type::<SmootError>())?;
    m.add("SmootParseError", m.py().get_type::<SmootParseError>())?;

    // Unit registry
    m.add_class::<UnitRegistry>()?;

    // Core unit type
    m.add_class::<Unit>()?;

    // float-backed types
    m.add_class::<F64Quantity>()?;
    m.add_class::<ArrayF64Quantity>()?;

    // // int backed types
    // m.add_class::<I64Quantity>()?;
    // m.add_class::<ArrayI64Quantity>()?;

    // let _ = m.add_function(wrap_pyfunction!(i64_to_f64_quantity, m)?);
    // let _ = m.add_function(wrap_pyfunction!(array_i64_to_f64_quantity, m)?);
    let _ = m.add_function(wrap_pyfunction!(mul_unit, m)?);
    let _ = m.add_function(wrap_pyfunction!(arr_mul_unit, m)?);
    let _ = m.add_function(wrap_pyfunction!(div_unit, m)?);
    let _ = m.add_function(wrap_pyfunction!(arr_div_unit, m)?);
    let _ = m.add_function(wrap_pyfunction!(rdiv_unit, m)?);
    let _ = m.add_function(wrap_pyfunction!(arr_rdiv_unit, m)?);

    Ok(())
}
