//! JSON payload accounting for the local interactive protocol.
//!
//! Counters exclude JSONL newline delimiters. The complete P→V payload and
//! Spain's public-opening-discounted proof convention are reported separately.

use anyhow::{Context, Result};
use serde::Serialize;
use spain::actor::SpainMessage;

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct WireMetrics {
    pub prover_message_bytes: usize,
    pub verifier_message_bytes: usize,
    /// All P→V JSON payloads, including the public opening.
    pub proof_size_bytes: usize,
    /// P→V payloads with range zero omitted from the final witness opening,
    /// matching SpainBroker. Repeated secondary openings are counted each time.
    pub proof_size_excluding_public_bytes: usize,
}

impl WireMetrics {
    /// Record the already serialized message without modifying its transcript.
    /// `to_prover` identifies the recipient, so true means verifier-originated.
    pub fn record(
        &mut self,
        message: &SpainMessage<i128>,
        to_prover: bool,
        encoded_bytes: usize,
    ) -> Result<()> {
        if to_prover {
            self.verifier_message_bytes = self
                .verifier_message_bytes
                .checked_add(encoded_bytes)
                .context("verifier message byte count overflow")?;
            return Ok(());
        }

        let discounted_bytes = match message {
            SpainMessage::WitnessOpenings(openings) => {
                let discounted = SpainMessage::WitnessOpenings(
                    openings
                        .iter()
                        .filter(|opening| opening.range_index != 0)
                        .cloned()
                        .collect(),
                );
                serde_json::to_vec(&discounted)?.len()
            }
            _ => encoded_bytes,
        };
        let prover_message_bytes = self
            .prover_message_bytes
            .checked_add(encoded_bytes)
            .context("prover message byte count overflow")?;
        let proof_size_bytes = self
            .proof_size_bytes
            .checked_add(encoded_bytes)
            .context("proof byte count overflow")?;
        let proof_size_excluding_public_bytes = self
            .proof_size_excluding_public_bytes
            .checked_add(discounted_bytes)
            .context("discounted proof byte count overflow")?;
        self.prover_message_bytes = prover_message_bytes;
        self.proof_size_bytes = proof_size_bytes;
        self.proof_size_excluding_public_bytes = proof_size_excluding_public_bytes;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spain::verifier::ZRangeOpening;

    fn encoded(message: &SpainMessage<i128>) -> Vec<u8> {
        serde_json::to_vec(message).unwrap()
    }

    fn secondary() -> ZRangeOpening<i128> {
        ZRangeOpening {
            range_index: 2,
            width: 1,
            values: vec![7, -11],
        }
    }

    fn final_opening() -> SpainMessage<i128> {
        SpainMessage::WitnessOpenings(vec![
            ZRangeOpening {
                range_index: 0,
                width: 1,
                values: vec![1, 42],
            },
            secondary(),
        ])
    }

    const DISCOUNTED_FINAL: &str =
        r#"{"WitnessOpenings":[{"range_index":2,"width":1,"values":["7","-11"]}]}"#;

    #[test]
    fn counts_message_origin_from_recipient() {
        let mut metrics = WireMetrics::default();
        let challenge = SpainMessage::Randomness(vec![123, 456]);
        let claim = SpainMessage::ErrorClaim("9".to_string());
        metrics
            .record(&challenge, true, encoded(&challenge).len())
            .unwrap();
        metrics
            .record(&claim, false, encoded(&claim).len())
            .unwrap();
        assert_eq!(metrics.verifier_message_bytes, encoded(&challenge).len());
        assert_eq!(metrics.prover_message_bytes, encoded(&claim).len());
        assert_eq!(metrics.proof_size_bytes, encoded(&claim).len());
        assert_eq!(
            metrics.proof_size_excluding_public_bytes,
            encoded(&claim).len()
        );
    }

    #[test]
    fn discounts_only_public_opening_without_changing_transcript() {
        let mut metrics = WireMetrics::default();
        let message = final_opening();
        let original = encoded(&message);
        metrics.record(&message, false, original.len()).unwrap();
        assert_eq!(metrics.proof_size_bytes, original.len());
        assert_eq!(
            metrics.proof_size_excluding_public_bytes,
            DISCOUNTED_FINAL.len()
        );
        assert!(metrics.proof_size_excluding_public_bytes < metrics.proof_size_bytes);
        assert_eq!(encoded(&message), original);
    }

    #[test]
    fn counts_secondary_openings_at_both_transmissions() {
        let mut metrics = WireMetrics::default();
        let binding = SpainMessage::BoundErrorClaim {
            squared_error: "0".to_string(),
            secondary_openings: vec![secondary()],
        };
        let opening = final_opening();
        let binding_size = encoded(&binding).len();
        let opening_size = encoded(&opening).len();
        metrics.record(&binding, false, binding_size).unwrap();
        metrics.record(&opening, false, opening_size).unwrap();
        assert_eq!(metrics.prover_message_bytes, binding_size + opening_size);
        assert_eq!(metrics.proof_size_bytes, binding_size + opening_size);
        assert_eq!(
            metrics.proof_size_excluding_public_bytes,
            binding_size + DISCOUNTED_FINAL.len()
        );
        assert_eq!(metrics.verifier_message_bytes, 0);
    }
}
