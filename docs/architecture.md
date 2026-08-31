# AI Defense Lab Architecture

## The Closed-Loop Pattern
The system is built on four core pillars that operate cyclically, not linearly.

1. **Identify**: Attack vectors are classified via the canonical `taxonomy/attacks.md`.
2. **Generate**: Domain-specific synthetic pipelines run in `generators/`, producing standardized `PaymentEvent` records.
3. **Defend**: `detectors/features/pipeline.py` extracts a shared feature space, feeding our primary supervised model (XGBoost) and secondary unsupervised model (Isolation Forest).
4. **Evaluate (The Loop)**: `eval/` enforces statistical fidelity and fixed-FPR detection scoring.

## Technical Decisions
- **Datasets**: We split our foundation. Card-not-present vectors (Phishing, Vishing, Agentic Commerce) rely on **Sparkov**. Structural sequences (Camouflage) rely on **PaySim**.
- **Unified Schema**: `schema/event.py` uses Pydantic. If an LLM hallucinates an invalid field, it never enters the dataset.
- **Explainability**: SHAP is built directly into the detection output to fulfill the real-world operational requirement.
- **Sparse Positive Handling**: We utilize an Isolation Forest to flag anomalous agent behaviors in Attack 4, acknowledging that supervised training data for zero-day prompt injection does not yet exist.
