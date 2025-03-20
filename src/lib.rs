use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use numpy::ndarray::{
    Array1, ArrayBase, ArrayD, ArrayView, ArrayView1, ArrayViewD, ArrayViewMutD, OwnedRepr,
    ScalarOperand, Zip,
};
use numpy::{
    datetime::{units, Timedelta},
    Complex64, IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArrayDyn,
};
use pyo3::{
    exceptions::PyIndexError,
    pymodule,
    types::{PyAnyMethods, PyDict, PyDictMethods, PyModule},
    Bound, FromPyObject, PyAny, PyObject, PyResult, Python,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use registry::REGISTRY;

mod base_unit;
mod error;
mod parse;
mod quantity;
mod registry;
mod types;
mod unit;
mod utils;

#[cfg(test)]
mod test_utils;

macro_rules! create_interface {
    ($name_unit: ident, $name_quantity: ident, $type1: ident, $type2: ident) => {
        #[pyclass]
        struct $name_unit {
            inner: unit::Unit<$type1>,
        }
        #[pymethods]
        impl $name_unit {}

        #[pyclass]
        struct $name_quantity {
            inner: quantity::Quantity<$type1, $type2>,
        }
        #[pymethods]
        impl $name_quantity {
            #[staticmethod]
            fn parse(expression: &str) -> PyResult<Self> {
                let quantity = quantity::Quantity::parse(&REGISTRY, expression)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner: quantity })
            }

            #[getter(m)]
            fn m(&self) -> $type1 {
                self.inner.magnitude
            }

            #[getter(magnitude)]
            fn magnitude(&self) -> $type1 {
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
            fn unit(&self) -> $name_unit {
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

            fn m_as(&self, unit: &$name_unit) -> PyResult<$type2> {
                self.inner
                    .m_as(&unit.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }
        }
    };
}

create_interface!(Float64Unit, Float64Quantity, f64, f64);

// #[pyfunction]
// fn parse_unit_f64<'u>(unit: &str) -> PyResult<()> {
//     unit.parse::<quantity::Quantity<f64, f64>>()
//         .map_err(|e| PyValueError::new_err(e.to_string()))?;
//     Ok(())
// }

#[pymodule]
fn pyo3_test(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Float64Unit>()?;
    m.add_class::<Float64Quantity>()?;
    // m.add_function(wrap_pyfunction!(parse_unit_f64, m)?)?;
    Ok(())
}

// // #[pyclass(eq, eq_int)]
// // #[derive(Clone, Copy, PartialEq, Display)]
// // #[strum(serialize_all = "snake_case")]
// // enum Unit {
// //     Meter,
// // }

// // type Dimension = u32;
// // const LENGTH: Dimension = 1;

// // /// py public
// // #[pymethods]
// // impl Unit {
// //     #[inline(always)]
// //     fn compatible_with(&self, other: &Self) -> bool {
// //         self.dimension() & other.dimension() > 0
// //     }

// //     fn __str__(&self) -> String {
// //         self.to_string()
// //     }
// // }

// // /// py private
// // impl Unit {
// //     #[inline(always)]
// //     const fn dimension(&self) -> Dimension {
// //         match self {
// //             Unit::Meter => LENGTH,
// //         }
// //     }
// // }

// // #[pyclass]
// // struct NPQuantityF64 {
// //     magnitude: ArrayD<f64>,
// //     unit: Unit,
// // }

// // #[pymethods]
// // impl NPQuantityF64 {
// //     #[getter]
// //     fn get_magnitude<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
// //         self.magnitude.clone().into_pyarray(py)
// //     }

// //     fn m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
// //         self.get_magnitude(py)
// //     }

// //     #[getter]
// //     fn get_unit(&self) -> Unit {
// //         self.unit
// //     }

// //     fn u(&self) -> Unit {
// //         self.unit
// //     }

// //     fn __add__(&self, other: &Self) -> Self {
// //         // TODO: unit conversions
// //         assert!(self.unit.compatible_with(&other.unit));
// //         Self {
// //             magnitude: &self.magnitude + &other.magnitude,
// //             unit: self.unit,
// //         }
// //     }

// //     fn __iadd__(&mut self, other: &Self) -> () {
// //         // TODO: unit conversions
// //         assert!(self.unit.compatible_with(&other.unit));
// //         self.magnitude += &other.magnitude;
// //     }

// //     fn __mul__(&self, other: &Self) -> Self {
// //         // TODO: unit conversions
// //         assert!(self.unit.compatible_with(&other.unit));
// //         Self {
// //             magnitude: &self.magnitude * &other.magnitude,
// //             unit: self.unit,
// //         }
// //     }

// //     fn __imul__(&mut self, other: &Self) -> () {
// //         // TODO: unit conversions
// //         assert!(self.unit.compatible_with(&other.unit));
// //         self.magnitude *= &other.magnitude;
// //     }

// //     fn __str__(&self) -> String {
// //         format!("{:?} {}", self.magnitude.as_slice().unwrap(), self.unit)
// //     }

// //     fn __repr__(&self) -> String {
// //         format!(
// //             "Quantity<({:?}, {})>",
// //             self.magnitude.as_slice().unwrap(),
// //             self.unit
// //         )
// //     }
// // }

// // // #[pyclass]
// // // #[derive(Debug)]
// // // struct NPQuantityI64 {
// // //     value: ArrayD<i64>,
// // //     unit: Unit,
// // // }

// // #[pyfunction]
// // fn np_quantity<'py>(
// //     py: Python<'py>,
// //     x: SupportedArrayType<'py>,
// //     unit: Unit,
// // ) -> PyResult<NPQuantityF64> {
// //     match x {
// //         SupportedArrayType::F64(x) => {
// //             let q = NPQuantityF64 {
// //                 magnitude: x.to_owned_array(),
// //                 unit,
// //             };
// //             return Ok(q);
// //         }
// //         _ => todo!(),
// //         // SupportedArrayType::I64(x) => {
// //         //     let q = NPQuantityI64{ value: x.to_owned_array(), unit };
// //         //     println!("{:?}", q);
// //         // },
// //     }
// // }

// // #[derive(FromPyObject)]
// // enum SupportedArrayType<'py> {
// //     F64(Bound<'py, PyArrayDyn<f64>>),
// //     I64(Bound<'py, PyArrayDyn<i64>>),
// // }

// // #[pymodule]
// // fn pyo3_test(m: &Bound<'_, PyModule>) -> PyResult<()> {
// //     m.add_class::<Unit>()?;
// //     m.add_function(wrap_pyfunction!(np_quantity, m)?)?;
// //     Ok(())
// // }
