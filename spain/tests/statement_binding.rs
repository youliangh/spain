use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use i256::I512;
use model::{F128, FromPrimitive};
use parse::{
    generalized::{HighPrecisionInt, InjectionInfo},
    mat::Matrix,
};
use protocol::machine::ProtocolState;
use spain::{
    actor::{SpainMessage, SpainProver, SpainVerifier},
    inputs::{Metadata, R1CSMatrices},
    prover::{ProverState, compute_squared_error, scale_factor},
    simulate::{SpainConfig, stateful_simulate_with_config},
    traits::R1CSInstance,
    verifier::{VerifierState, ZRangeOpening, epsilon_check},
};

/// A tiny post-commit projection of the fixed linear relation 2 + 3 = 5.
/// Both projected equations change with verifier randomness; a scalar computed
/// after the challenge is public secondary witness, as in interactive R1CS.
#[derive(Clone, Debug)]
struct Projection {
    projected_matrices: Arc<AtomicUsize>,
    public_claim: i128,
    expected_claim: i128,
    tamper_primary: bool,
}

impl Default for Projection {
    fn default() -> Self {
        Self {
            projected_matrices: Arc::new(AtomicUsize::new(0)),
            public_claim: 5,
            expected_claim: 5,
            tamper_primary: false,
        }
    }
}

impl R1CSInstance<F128, i128> for Projection {
    fn requires_challenge_matrices(&self) -> bool {
        true
    }

    fn expected_public_values(&self, scale: F128, batch: usize) -> Option<Vec<i128>> {
        assert_eq!(batch, 1);
        let d = i128::from_hp(scale);
        Some(vec![d, self.expected_claim * d])
    }

    fn get_meta(&self) -> Metadata {
        Metadata {
            num_public_values: 2,
            num_random_values: 1,
            num_witness_values: 4,
            num_secondary_witness_values: 1,
            ..Default::default()
        }
    }

    fn get_matrices(
        &self,
        scale: F128,
        random: Option<&Vec<i128>>,
    ) -> (R1CSMatrices<i128>, Option<InjectionInfo>) {
        let d = i128::from_hp(scale);
        let r = random.map_or(0, |values| {
            self.projected_matrices.fetch_add(1, Ordering::SeqCst);
            values[0]
        });
        let a = vec![
            (0, 3, r),
            (0, 4, r),
            (1, 7, d),
            (2, 3, 2 * d),
            (3, 3, d),
            (3, 4, d),
        ];
        let b = (0..4).map(|row| (row, 0, d)).collect();
        let c = vec![(0, 7, d), (1, 1, r), (2, 5, d), (3, 6, d)];
        (
            R1CSMatrices {
                a: Matrix::from_coo(a, 8, 4),
                b: Matrix::from_coo(b, 8, 4),
                c: Matrix::from_coo(c, 8, 4),
            },
            None,
        )
    }

    fn compute_commit_witness(&mut self, scale: F128, batch: usize) -> Matrix<i128> {
        assert_eq!(batch, 1);
        let d = i128::from_hp(scale);
        Matrix::from_vec(vec![
            (2 + i128::from(self.tamper_primary)) * d,
            3 * d,
            4 * d,
            5 * d,
        ])
    }

    fn compute_full_witness(
        &mut self,
        meta: &Metadata,
        random: Vec<F128>,
        scale: F128,
    ) -> Matrix<i128> {
        let d = i128::from_hp(scale.clone());
        let r = i128::from_hp(random[0].clone() * scale);
        let mut z = Matrix::from_vec(vec![
            d,
            self.public_claim * d,
            r,
            (2 + i128::from(self.tamper_primary)) * d,
            3 * d,
            4 * d,
            5 * d,
            5 * r,
        ]);
        z.set_ranges(&meta.get_ranges());
        z
    }
}

fn config() -> SpainConfig {
    SpainConfig {
        scale_factor_bits: 20,
        max_epsilon: 1e-6,
        num_chunks: 1,
        precision: 128,
        q_bits: 30_000,
        batch_size: 1,
        spartan_poly: false,
    }
}

fn verifier(exec: Projection) -> VerifierState<i128, F128, Projection> {
    let cfg = config();
    let meta = exec.get_meta();
    VerifierState::new(
        cfg.max_epsilon,
        cfg.batch_size,
        cfg.scale_factor_bits,
        cfg.q_bits,
        cfg.precision,
        cfg.num_chunks,
        false,
        exec,
        meta,
    )
}

fn openings() -> (
    VerifierState<i128, F128, Projection>,
    Vec<ZRangeOpening<i128>>,
) {
    let mut state = verifier(Projection::default());
    state.randomness = Some(vec![F128::from_f64(0.25).unwrap()]);
    let d = 1_i128 << config().scale_factor_bits;
    let values = vec![
        ZRangeOpening {
            range_index: 0,
            width: 1,
            values: vec![d, 5 * d, d / 4],
        },
        ZRangeOpening {
            range_index: 2,
            width: 1,
            values: vec![5 * d / 4],
        },
    ];
    state.bind_secondary_openings(&values[1..]);
    (state, values)
}

#[test]
fn statement_binding_checks_public_constant_output_and_randomness() {
    let (state, valid) = openings();
    state.validate_witness_openings(&valid);
    for index in 0..3 {
        let mut bad = valid.clone();
        bad[0].values[index] += 1;
        assert!(catch_unwind(AssertUnwindSafe(|| state.validate_witness_openings(&bad))).is_err());
    }
}

#[test]
fn statement_binding_is_fixed_before_challenges() {
    let (mut state, valid) = openings();
    state.wit_exec.expected_claim = 99;
    state.validate_witness_openings(&valid);
    let mut changed = valid;
    changed[0].values[1] = 99 * (1_i128 << config().scale_factor_bits);
    assert!(
        catch_unwind(AssertUnwindSafe(
            || state.validate_witness_openings(&changed)
        ))
        .is_err()
    );
}

#[test]
fn statement_binding_rejects_malformed_openings() {
    let (state, valid) = openings();
    let mut cases = Vec::new();
    let mut duplicate = valid.clone();
    duplicate.push(valid[0].clone());
    cases.push(duplicate);
    cases.push(vec![valid[0].clone()]);
    cases.push(vec![valid[1].clone()]);
    let mut wrong_width = valid.clone();
    wrong_width[0].width = 2;
    cases.push(wrong_width);
    let mut short = valid.clone();
    short[0].values.pop();
    cases.push(short);
    let mut primary = valid.clone();
    primary.push(ZRangeOpening {
        range_index: 1,
        width: 1,
        values: vec![0; 4],
    });
    cases.push(primary);
    let mut out_of_range = valid.clone();
    out_of_range[1].range_index = 8;
    cases.push(out_of_range);
    for bad in cases {
        assert!(catch_unwind(AssertUnwindSafe(|| state.validate_witness_openings(&bad))).is_err());
    }
}

#[test]
fn statement_binding_stateful_real_protocol() {
    let exec = Projection::default();
    let counter = exec.projected_matrices.clone();
    let result = stateful_simulate_with_config::<F128, _, i128>(exec, config());
    assert_eq!(result.num_constraints, 4);
    assert_eq!(counter.load(Ordering::SeqCst), 2);
}

#[test]
fn statement_binding_preserves_legacy_executor_defaults() {
    let exec = spain::synthetic::SyntheticR1CS::<i128>::new(4, 1, 20);
    assert!(!<_ as R1CSInstance<F128, i128>>::requires_challenge_matrices(&exec));
    assert!(
        <_ as R1CSInstance<F128, i128>>::expected_public_values(&exec, scale_factor(20), 1)
            .is_none()
    );
    let mut cfg = config();
    cfg.spartan_poly = true;
    let result = stateful_simulate_with_config::<F128, _, i128>(exec, cfg);
    assert_eq!(result.num_constraints, 4);
}

#[derive(Clone, Copy)]
enum Tamper {
    None,
    Public,
    Secondary,
    OmitSecondaryBinding,
    EarlyDarkFinal,
    InnerDimension,
    OuterFinalArity,
    InnerFinalArity,
}

fn actor_protocol(exec: Projection, tamper: Tamper) {
    let meta = exec.get_meta();
    let scale = scale_factor::<F128>(config().scale_factor_bits);
    let mut prover = SpainProver::new(ProverState::new(exec.clone(), scale, meta, 1, false));
    let mut verifier = SpainVerifier::new(verifier(exec));
    let mut message = verifier.init_message().unwrap();
    let mut completed = false;
    for _ in 0..128 {
        let mut response = prover.handle_message(&message).expect("prover ended early");
        match tamper {
            Tamper::Public | Tamper::Secondary => {
                if let SpainMessage::WitnessOpenings(ref mut values) = response {
                    let (range, index) = match tamper {
                        Tamper::Public => (0, 1),
                        _ => (2, 0),
                    };
                    values
                        .iter_mut()
                        .find(|v| v.range_index == range)
                        .unwrap()
                        .values[index] += 1;
                }
            }
            Tamper::OmitSecondaryBinding => {
                if let SpainMessage::BoundErrorClaim { squared_error, .. } = &response {
                    response = SpainMessage::ErrorClaim(squared_error.clone());
                }
            }
            Tamper::EarlyDarkFinal => {
                if let SpainMessage::DarkRoundResponse(ref mut round) = response {
                    if round.final_claim.is_none() {
                        round.final_claim = Some(dark::prover::FinalClaim {
                            final_constant: round.y_l,
                            final_constant_int: rug::Integer::from(0),
                        });
                    }
                }
            }
            Tamper::InnerDimension => {
                if let SpainMessage::InnerStart {
                    ref mut num_vars, ..
                } = response
                {
                    *num_vars -= 1;
                }
            }
            Tamper::OuterFinalArity => {
                if let SpainMessage::OuterFinalEvals(ref mut values) = response {
                    values.push(values[0]);
                }
            }
            Tamper::InnerFinalArity => {
                if let SpainMessage::InnerFinalEvals(ref mut values) = response {
                    values.push(values[0]);
                }
            }
            Tamper::None => {}
        }
        match verifier.handle_message(&response) {
            Some(next) => message = next,
            None => {
                completed = true;
                break;
            }
        }
    }
    assert!(completed, "protocol exceeded round bound");
    assert!(prover.is_done());
    assert!(verifier.is_done());
}

#[test]
fn statement_binding_actor_real_protocol() {
    actor_protocol(Projection::default(), Tamper::None);
}

#[test]
fn statement_binding_rejects_tampered_actual_protocol() {
    assert!(
        catch_unwind(AssertUnwindSafe(|| actor_protocol(
            Projection::default(),
            Tamper::Public
        )))
        .is_err()
    );
}

#[test]
fn statement_binding_rejects_adaptive_secondary_witness() {
    for tamper in [Tamper::Secondary, Tamper::OmitSecondaryBinding] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| actor_protocol(
                Projection::default(),
                tamper
            )))
            .is_err()
        );
    }
}

#[test]
fn statement_binding_rejects_malformed_protocol_rounds() {
    for tamper in [
        Tamper::EarlyDarkFinal,
        Tamper::InnerDimension,
        Tamper::OuterFinalArity,
        Tamper::InnerFinalArity,
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| actor_protocol(
                Projection::default(),
                tamper
            )))
            .is_err()
        );
    }
}

#[test]
fn statement_binding_rejects_different_claim_and_bad_primary_witness() {
    let mismatched = Projection {
        expected_claim: 6,
        ..Default::default()
    };
    assert!(
        catch_unwind(AssertUnwindSafe(|| actor_protocol(
            mismatched,
            Tamper::None
        )))
        .is_err()
    );
    let wrong_witness = Projection {
        tamper_primary: true,
        ..Default::default()
    };
    assert!(
        catch_unwind(AssertUnwindSafe(|| actor_protocol(
            wrong_witness,
            Tamper::None
        )))
        .is_err()
    );
}

#[test]
fn statement_binding_rejects_invalid_epsilon_claims() {
    epsilon_check(&I512::from(0), 20, 0.0, false);
    assert!(catch_unwind(|| epsilon_check(&I512::from(-1), 20, 0.1, false)).is_err());
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
        assert!(catch_unwind(|| epsilon_check(&I512::from(0), 20, threshold, false)).is_err());
    }
}

#[test]
fn statement_binding_detects_integer_error_overflow() {
    let huge = I512::from(1) << 300;
    let matrices = R1CSMatrices {
        a: Matrix::from_coo(vec![(0, 0, huge)], 1, 1),
        b: Matrix::from_coo(vec![(0, 0, I512::from(1))], 1, 1),
        c: Matrix::from_coo(vec![(0, 0, I512::from(0))], 1, 1),
    };
    let z = Matrix::from_vec(vec![I512::from(1)]);
    assert!(
        catch_unwind(AssertUnwindSafe(|| compute_squared_error(
            &matrices,
            &z,
            &I512::from(1),
            false
        )))
        .is_err()
    );
    let overflow_z = Matrix::from_vec(vec![huge]);
    assert!(
        catch_unwind(AssertUnwindSafe(|| compute_squared_error(
            &matrices,
            &overflow_z,
            &I512::from(1),
            false
        )))
        .is_err()
    );
}
