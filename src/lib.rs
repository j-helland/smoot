use mimalloc::MiMalloc;                        

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use numpy::{PyArray, PyArrayDyn, PyArrayMethods, ndarray::ArrayD};
use pyo3::{
    pymodule,
    types::PyModule,
    Bound, PyResult, Python,
};
use pyo3::{exceptions::{PyValueError, PyRuntimeError}, prelude::*};

use crate::registry::REGISTRY;

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

macro_rules! create_unit_type {
    ($name_unit: ident, $base_type: ident) => {
        #[pyclass(module = "smoot.smoot")]
        struct $name_unit {
            inner: unit::Unit<$base_type>,
        }
        #[pymethods]
        impl $name_unit {
            #[staticmethod]
            fn parse(expression: &str) -> PyResult<Self> {
                let unit = unit::Unit::parse(&REGISTRY, expression)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner: unit.into() })
            }
        }       
    };
}

/// Create a quantity with a given underlying type.
/// Creates a matching Unit type.
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
                Ok(Self { inner: quantity.into() })
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
                // TODO(jwh): find a way to not copy every call
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
                Self { inner: &self.inner * scalar }
            }

            // standard dunder methods
            fn __str__(&self) -> String {
                format!("{} {}", self.inner.magnitude, self.inner.unit.get_units_string().unwrap_or("dimensionless".into()))
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

            // operators
            fn __eq__(&self, other: &Self) -> bool {
                self.inner.partial_cmp(&other.inner) == Some(std::cmp::Ordering::Equal)
            }

            fn __ne__(&self, other: &Self) -> bool {
                self.inner.partial_cmp(&other.inner) != Some(std::cmp::Ordering::Equal)
            }

            fn __lt__(&self, other: &Self) -> bool {
                self.inner.partial_cmp(&other.inner) == Some(std::cmp::Ordering::Less)
            }

            fn __gt__(&self, other: &Self) -> bool {
                self.inner.partial_cmp(&other.inner) == Some(std::cmp::Ordering::Greater)
            }

            fn __le__(&self, other: &Self) -> bool {
                match self.inner.partial_cmp(&other.inner) {
                    Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Less) => true,
                    _ => false,
                }
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

            fn __radd__(&self, _other: &Self) -> PyResult<Self> {
                todo!();
            }

            fn __sub__(&self, other: &Self) -> PyResult<Self> {
                (&self.inner - &other.inner)
                    .map(|inner| Self { inner })
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }

            fn __rsub__(&self, _other: &Self) -> PyResult<Self> {
                todo!();
            }

            fn __mul__(&self, other: &Self) -> Self {
                Self { inner: &self.inner * &other.inner }
            }

            fn __rmul__(&self, _other: &Self) -> PyResult<Self> {
                todo!();
            }

            fn __matmul__(&self, _other: &Self) -> PyResult<Self> {
                todo!();
            }

            fn __rmatmul__(&self, _other: &Self) -> PyResult<Self> {
                todo!();
            }

            fn __truediv__(&self, other: &Self) -> Self {
                Self { inner: &self.inner / &other.inner }
            }            

            fn __rtruediv__(&self, _other: &Self) -> PyResult<Self> {
                todo!();
            }

            fn __floordiv__(&self, _other: &Self) -> Self {
                todo!();
            }

            fn __rfloordiv__(&self, _other: &Self) -> Self {
                todo!();
            }

            fn __neg__(&self) -> Self {
                Self { inner: -self.inner.clone() }
            }

            fn __round__(&self) -> Self {
                todo!();
            }

            fn __trunc__(&self) -> Self {
                todo!();
            }

            fn __floor__(&self) -> Self {
                todo!();
            }

            fn __ceil__(&self) -> Self {
                todo!();
            }

            fn __abs__(&self) -> Self {
                todo!();
            }

            // pickle support
            fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
                self.inner = bitcode::decode(state)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
            #[staticmethod]
            fn new(arr: Bound<'_, PyArrayDyn<$base_type>>, unit: &$unit_type) -> Self {
                Self { inner: quantity::Quantity::new(arr.to_owned_array(), unit.inner.clone()) }
            }

            #[getter(m)]
            fn m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<$base_type>>> {
                Ok(PyArray::from_array(py, &self.inner.magnitude))
            }

            #[getter(magnitude)]
            fn magnitude<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<$base_type>>> {
                self.m(py)
            }

            #[getter(u)]
            fn u(&self) -> $unit_type {
                // TODO(jwh): find a way to not copy every call
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

            fn m_as<'py>(&self, py: Python<'py>, unit: &$unit_type) -> PyResult<Bound<'py, PyArrayDyn<$base_type>>> {
                self.inner
                    .m_as(&unit.inner)
                    .map(|arr| PyArray::from_array(py, &arr))
                    .map_err(|e| PyValueError::new_err(e.to_string()))
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

    Ok(())
}
