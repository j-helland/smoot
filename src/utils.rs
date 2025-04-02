use num_traits::Float;

pub fn float_eq_rel<N: Float>(a: N, b: N, max_diff: N) -> bool {
    let largest = a.abs().max(b.abs());
    (a - b).abs() <= (largest * max_diff)
}

pub trait ApproxEq {
    fn approx_eq(self, x: Self) -> bool;
}
impl ApproxEq for f64 {
    fn approx_eq(self, x: Self) -> bool {
        float_eq_rel(self, x, 1e-6)
    }
}
impl ApproxEq for i64 {
    fn approx_eq(self, x: Self) -> bool {
        self == x
    }
}

pub trait Powf {
    fn powf(self, p: f64) -> Self;
}
impl Powf for f64 {
    fn powf(self, p: f64) -> Self {
        self.powf(p)
    }
}
impl Powf for i64 {
    fn powf(self, p: f64) -> Self {
        (self as f64).powf(p) as i64
    }
}
