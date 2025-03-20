use crate::types::Number;

pub fn float_eq_rel<N: Number>(a: N, b: N, max_diff: N) -> bool {
    let largest = a.abs().max(b.abs());
    (a - b).abs() <= (largest * max_diff)
}
