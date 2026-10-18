//! Exercise the original checked-in floating-point fixtures without requiring
//! separately exported ONNX circuits or downloaded GPT-2 weights.
use std::path::PathBuf;

use model::AFloat;
use parse::{generalized::InjectionInfo, mat::Matrix};
use spain::{
    inputs::{Metadata, R1CSMatrices},
    simulate::{SpainConfig, stateful_simulate_with_config},
    traits::R1CSInstance,
};

#[derive(Clone)]
struct NumericalFixture {
    matrices: R1CSMatrices<f64>,
    witness: Matrix<f64>,
}

impl NumericalFixture {
    fn read(name: &str) -> Self {
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../parse/data");
        let read = |suffix| {
            Matrix::<f64>::from_file(&directory.join(format!("{name}_{suffix}.bin")))
                .expect("checked-in numerical fixture missing or invalid")
                .1
        };
        Self {
            matrices: R1CSMatrices {
                a: read("A"),
                b: read("B"),
                c: read("C"),
            },
            witness: read("Z"),
        }
    }

    fn scaled_witness(&self, scale: AFloat) -> Matrix<i128> {
        let mut witness = Matrix::<i128>::from_f64(&self.witness, scale, None);
        witness.set_ranges(&self.get_meta().get_ranges());
        witness
    }
}

impl R1CSInstance<AFloat, i128> for NumericalFixture {
    fn get_matrices(
        &self,
        scale: AFloat,
        _: Option<&Vec<i128>>,
    ) -> (R1CSMatrices<i128>, Option<InjectionInfo>) {
        (
            R1CSMatrices {
                a: Matrix::<i128>::from_f64(&self.matrices.a, scale.clone(), None),
                b: Matrix::<i128>::from_f64(&self.matrices.b, scale.clone(), None),
                c: Matrix::<i128>::from_f64(&self.matrices.c, scale, None),
            },
            None,
        )
    }

    fn get_meta(&self) -> Metadata {
        Metadata {
            num_public_values: 1,
            num_witness_values: self.witness.height() - 1,
            ..Default::default()
        }
    }

    fn compute_commit_witness(&mut self, scale: AFloat, batch: usize) -> Matrix<i128> {
        assert_eq!(batch, 1);
        self.scaled_witness(scale)
            .extract_rows(&self.get_meta().get_ranges()[1])
    }

    fn compute_full_witness(
        &mut self,
        _: &Metadata,
        random: Vec<AFloat>,
        scale: AFloat,
    ) -> Matrix<i128> {
        assert!(random.is_empty());
        self.scaled_witness(scale)
    }
}

fn prove_fixture(name: &str, constraints: usize, epsilon: f64) {
    let instance = NumericalFixture::read(name);
    let config = SpainConfig {
        // The stored witnesses have FP32 rounding residuals, so the smaller
        // denominator keeps the exact squared residual safely inside I512.
        scale_factor_bits: 40,
        max_epsilon: epsilon,
        num_chunks: 4,
        ..Default::default()
    };
    let result = stateful_simulate_with_config::<AFloat, _, i128>(instance, config);
    assert_eq!(result.num_constraints, constraints);
}

#[test]
fn legacy_layer_norm_numerical_proof() {
    prove_fixture("layer_norm", 16, 1e-6);
}

#[test]
fn legacy_random_vars_numerical_proof() {
    prove_fixture("random_vars", 206, 1e-5);
}
