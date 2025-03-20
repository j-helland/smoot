/// Asserts that two floating point numbers are close to each other.
///
/// The default tolerance is 1e-6.
///
/// # Examples
///
/// ```
/// assert_is_close!(1.0, 1.0);
/// assert_is_close!(1.0, 1.0, 1e-10);
/// ```
#[macro_export]
macro_rules! assert_is_close {
    ($a:expr, $b:expr) => {
        assert!((($a as f64 - $b as f64).abs() / $a as f64).abs() < 1e-6);
    };
    ($a:expr, $b:expr, $rel_tol:expr) => {
        assert!((($a as f64 - $b as f64).abs() / $a as f64).abs() < $rel_tol);
    };
}
