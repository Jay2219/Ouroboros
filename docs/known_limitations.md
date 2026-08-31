# Known Limitations & Future Work

While the AI Defense Lab successfully models multiple advanced generative fraud attacks, there are intentional boundaries to its scope and fidelity due to time constraints and the cutting-edge nature of the simulated attacks.

## 1. Attack 4: Agentic Injection Lacks Ground Truth
**Limitation:** The Agentic Commerce Prompt Injection simulation represents a theoretical future attack vector. Because consumer-facing autonomous checkout agents are not yet widely deployed, there is no historical dataset of real-world agentic fraud to compare against.
**Validation Approach:** Our fidelity validation for Attack 4 relies strictly on internal consistency: we verify that the generated anomaly matches the semantic intent of the poisoned payload (e.g., if the payload demands an excessive quantity of an item, the transaction record reflects that specific anomaly).

## 2. Audio Synthesis Omission
**Limitation:** While Attack 2 simulates Voice-Cloned Vishing, the actual audio generation (Kokoro TTS) was scoped out to focus on the end-to-end Machine Learning detection loop.
**Impact:** The system currently relies purely on the generated transcripts and metadata tags. The taxonomy remains valid, as the text payload correctly models the social engineering aspect of the attack.

## 3. Class Imbalance
**Limitation:** Real-world payment systems see massive class imbalance (often 99.9% legitimate, 0.1% fraud). Our training environment uses a less extreme ratio to ensure the classifiers could converge within the short training window.

## Future Expansion
- Integration of actual Voice Cloning engines to validate cross-modal (Audio -> Text) detection pipelines.
- Deepening the sequence-length of the Adversarial Camouflage simulation (Attack 3) to span weeks instead of a 24-hour window, requiring more advanced LSTM/Transformer detectors instead of XGBoost.
