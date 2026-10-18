//! Shared 128-bit arithmetic and the explicitly bounded rational exponential.
use ::model::AFloat;
use anyhow::{Result, ensure};
use rug::{Assign, Float};

pub const PRECISION: u32 = 128;
pub const EXP_SQUARINGS: usize = 10;
pub const EXP_MIN: f64 = -128.0;
pub fn scalar(value: f64) -> AFloat {
    AFloat(Float::with_val(PRECISION, value))
}
pub fn zero() -> AFloat {
    scalar(0.0)
}
pub fn one() -> AFloat {
    scalar(1.0)
}
pub fn add(a: &AFloat, b: &AFloat) -> AFloat {
    AFloat(Float::with_val(PRECISION, &a.0 + &b.0))
}
pub fn sub(a: &AFloat, b: &AFloat) -> AFloat {
    AFloat(Float::with_val(PRECISION, &a.0 - &b.0))
}
pub fn mul(a: &AFloat, b: &AFloat) -> AFloat {
    AFloat(Float::with_val(PRECISION, &a.0 * &b.0))
}
pub fn div(a: &AFloat, b: &AFloat) -> AFloat {
    AFloat(Float::with_val(PRECISION, &a.0 / &b.0))
}
pub fn scale(a: &AFloat, b: f64) -> AFloat {
    AFloat(Float::with_val(PRECISION, &a.0 * b))
}
pub fn sqrt(a: &AFloat) -> AFloat {
    AFloat(a.0.clone().sqrt())
}

/// exp(x) = (P(x/1024)/Q(x/1024))^1024 on the enforced domain [-128,0].
/// Coefficients are integers, so no fitted or rounded coefficients enter the circuit.
pub fn exp_negative(x: &AFloat) -> Result<AFloat> {
    ensure!(
        x.0.is_finite() && x.0 >= EXP_MIN && x.0 <= 0,
        "rational exp input outside [{EXP_MIN},0]: {}",
        x.0
    );
    let z = scale(x, 1.0 / 1024.0);
    let z2 = mul(&z, &z);
    let z3 = mul(&z2, &z);
    let even = add(&scalar(120.0), &scale(&z2, 12.0));
    let odd = add(&scale(&z, 60.0), &z3);
    let mut e = div(&add(&even, &odd), &sub(&even, &odd));
    for _ in 0..EXP_SQUARINGS {
        e = mul(&e, &e);
    }
    Ok(e)
}

pub fn silu(x: &AFloat) -> Result<AFloat> {
    let negative_abs = AFloat(-x.0.clone().abs());
    let e = exp_negative(&negative_abs)?;
    let denominator = add(&one(), &e);
    let sigmoid = if x.0 >= 0 {
        div(&one(), &denominator)
    } else {
        div(&e, &denominator)
    };
    Ok(mul(x, &sigmoid))
}

/// Dot product with reusable MPFR temporaries (no allocation per multiplication).
pub fn dot(a: &[AFloat], b: &[AFloat]) -> AFloat {
    assert_eq!(a.len(), b.len());
    let mut sum = Float::with_val(PRECISION, 0);
    let mut product = Float::with_val(PRECISION, 0);
    for (a, b) in a.iter().zip(b) {
        product.assign(&a.0 * &b.0);
        sum += &product;
    }
    AFloat(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pade_is_positive_and_accurate_on_entire_domain() {
        for i in 0..=2048 {
            let x = -128.0 * i as f64 / 2048.0;
            let actual = exp_negative(&scalar(x)).unwrap().0.to_f64();
            assert!(actual > 0.0);
            assert!((actual / x.exp() - 1.0).abs() < 6e-9, "x={x}");
        }
        assert!(exp_negative(&scalar(-128.1)).is_err());
        assert!(exp_negative(&scalar(0.1)).is_err());
    }
    #[test]
    fn stable_silu_has_both_signs() {
        for x in [-128.0, -16.0, -1.0, 0.0, 1.0, 16.0, 128.0] {
            let actual = silu(&scalar(x)).unwrap().0.to_f64();
            let expected = x / (1.0 + (-x).exp());
            assert!((actual - expected).abs() < 1e-8);
        }
    }
}
