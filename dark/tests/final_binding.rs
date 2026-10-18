#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use dark::{
    prover::{ChunkedComm, FinalClaim, ProverState},
    public::PublicParams,
    verifier::VerifierState,
};
use ff::poly::int::MLE;
use rug::{Integer, ops::Pow};

fn rejected(f: impl FnOnce()) {
    assert!(catch_unwind(AssertUnwindSafe(f)).is_err());
}

fn setup(num_vars: usize, chunks: usize, precision: u16) -> (PublicParams, VerifierState) {
    let mut public = PublicParams::new(30_000, num_vars, chunks, precision);
    let mut verifier = VerifierState::new(&public);
    verifier.compute_const_comms(&mut public);
    public.build_pippenger_bases();
    (public, verifier)
}

fn final_state(value: &Integer) -> (PublicParams, VerifierState) {
    let public = PublicParams::new(30_000, 2, 1, 128);
    let mut verifier = VerifierState::new(&public);
    verifier.round = public.total_rounds;
    verifier.comm = public.mont.fast_exp(&public.rsa_gen, value, &verifier.car);
    verifier.y_claim = Some(public.small_mont.from_bigint(value.clone()));
    (public, verifier)
}

#[test]
fn final_integer_and_field_openings_must_agree() {
    let integer = Integer::from(7);
    let (public, mut verifier) = final_state(&integer);
    let honest = public.small_mont.from_bigint(integer.clone());
    verifier.final_check(&honest, &integer, &public);
    // Both legacy equality checks pass: the field claim agrees with y and
    // the integer opens the commitment, but they describe different values.
    let forged = public.small_mont.add(honest, public.small_mont.one());
    verifier.y_claim = Some(forged);
    rejected(|| verifier.final_check(&forged, &integer, &public));
}

#[test]
fn out_of_bound_integer_and_incomplete_proof_are_rejected() {
    let (public, mut verifier) = final_state(&Integer::from(7));
    let growth: Integer = Integer::from(public.small_mont.modulus()) * 2 - 3;
    let too_large = public.int_shift_base() * growth.pow(public.total_rounds as u32) + 1;
    verifier.comm = public
        .mont
        .fast_exp(&public.rsa_gen, &too_large, &verifier.car);
    let field = public.small_mont.from_bigint(too_large.clone());
    verifier.y_claim = Some(field);
    rejected(|| verifier.final_check(&field, &too_large, &public));

    let (public, mut verifier) = final_state(&Integer::from(7));
    verifier.round -= 1;
    rejected(|| verifier.final_check(&verifier.y_claim.unwrap(), &Integer::from(7), &public));
}

fn honest_protocol(num_vars: usize, chunks: usize, precision: u16) {
    let (public, mut verifier) = setup(num_vars, chunks, precision);
    let mut prover = ProverState::default();
    let magnitude: Integer = public.int_shift_base() / 8;
    let poly = MLE::from_buffer(
        (0..1 << num_vars)
            .map(|i| {
                if i % 2 == 0 {
                    magnitude.clone()
                } else {
                    -magnitude.clone()
                }
            })
            .collect(),
        num_vars,
    );
    let commitment = prover.commit(poly, &public);
    verifier.set_commit(commitment);
    let point = vec![public.small_mont.to_mont(3); num_vars];
    let y = prover.gen_y_claim(point.clone(), &public);
    verifier.set_claim(y, point);
    for _ in 0..num_vars {
        let challenge = verifier.start_round(&public, &mut rand::rng());
        let claim = prover.respond_to_challenge(&challenge, &public);
        verifier.verify_round(&claim, &public);
    }
    rejected(|| {
        verifier.start_round(&public, &mut rand::rng());
    });
}

#[test]
fn real_protocol_checks_final_opening_for_all_supported_precisions() {
    for precision in [64, 128, 256, 512] {
        honest_protocol(4, 1, precision);
    }
}

#[test]
fn real_protocol_accepts_256_chunks() {
    honest_protocol(10, 256, 128);
}

#[test]
fn final_claim_cannot_end_protocol_early_or_be_omitted() {
    let (public, mut verifier) = setup(2, 1, 128);
    let mut prover = ProverState::default();
    let commitment = prover.commit(MLE::from_buffer(vec![Integer::from(7); 4], 2), &public);
    verifier.set_commit(commitment);
    let point = vec![public.small_mont.to_mont(3); 2];
    let y = prover.gen_y_claim(point.clone(), &public);
    verifier.set_claim(y, point);
    let challenge = verifier.start_round(&public, &mut rand::rng());
    let mut claim = prover.respond_to_challenge(&challenge, &public);
    let honest = claim.clone();
    claim.final_claim = Some(FinalClaim {
        final_constant: public.small_mont.to_mont(7),
        final_constant_int: Integer::from(7),
    });
    rejected(|| verifier.verify_round(&claim, &public));
    verifier.verify_round(&honest, &public);
    let challenge = verifier.start_round(&public, &mut rand::rng());
    let mut claim = prover.respond_to_challenge(&challenge, &public);
    let honest = claim.clone();
    claim.final_claim = None;
    rejected(|| verifier.verify_round(&claim, &public));
    verifier.verify_round(&honest, &public);
}

#[test]
fn malformed_commitment_layout_and_evaluation_point_are_rejected() {
    let (public, verifier) = setup(2, 1, 128);
    for (chunks, chunk_size, point_length) in [(0, 4, 2), (2, 4, 2), (1, 2, 2), (1, 4, 1)] {
        let mut verifier = verifier.clone();
        verifier.set_commit(ChunkedComm {
            comms: vec![public.mont.one(); chunks],
            chunk_size,
        });
        verifier.set_claim(
            public.small_mont.zero(),
            vec![public.small_mont.one(); point_length],
        );
        rejected(|| {
            verifier.start_round(&public, &mut rand::rng());
        });
    }
}
