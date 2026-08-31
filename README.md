# 🛡️ OUROBOROS: Autonomous AI Defense Lab for Payment Security
**Next-Generation Adversary Simulation, Evasion Discovery & Adaptive ML Retraining**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%2B%20TreeSHAP-orange.svg)](https://xgboost.readthedocs.io/)
[![Gradio Dashboard](https://img.shields.io/badge/UI-Gradio%205.20-red.svg)](https://gradio.app/)
[![Dataset: Sparkov & PaySim](https://img.shields.io/badge/Data-Sparkov%20%7C%20PaySim-green.svg)](https://www.kaggle.com/)

---

## 📌 Executive Summary

Traditional fraud detection engines rely heavily on static transaction rules (e.g., *"flag any single transaction > $3,000"*) and retrospective fraud label analysis. However, emerging **Generative AI adversarial agents** can dynamically infer target context, synthesize voice personas, structure micro-transactions across time windows, and inject malicious prompts into autonomous shopping workflows—bypassing static thresholds entirely.

**Ouroboros** is an enterprise-grade AI Defense Lab for payment processors and financial institutions. It operates as an autonomous closed loop:
1. **Adversary Simulation (Red Team):** Synthesizes high-fidelity fraud vectors using LLMs grounded on real-world transaction patterns.
2. **Evasion Discovery:** Automatically pits synthetic attack campaigns against baseline static defense rules to find vulnerabilities.
3. **Adaptive ML Hardening (Blue Team):** Ingests discovered evasions, computes velocity and behavioral features, and retrains production XGBoost + Isolation Forest classifiers to achieve **100.0% recall** at a strict **1.0% False Positive Rate (FPR)** budget.

---

## 🏗️ Closed-Loop Architecture

```
                                  +---------------------------------------+
                                  |    1. ADVERSARY SIMULATION (RED)      |
                                  |  - Mass-Personalized Spear Phishing   |
                                  |  - Voice-Cloned Vishing Persona Wire  |
                                  |  - Temporal Structuring / Camouflage  |
                                  |  - Agentic Prompt Injection Hijack    |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   2. EVASION & BLIND-SPOT DISCOVERY   |
                                  |  - Evaluate against Static $3k Rule   |
                                  |  - Flag 85.0% Evaded Camouflage Burst |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |    3. CONTINUOUS RETRAINING (BLUE)    |
                                  |  - Feature Pipeline (24h Velocity)    |
                                  |  - XGBoost Classifier + Isolation For |
                                  |  - TreeSHAP Feature Attributions     |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |        4. PRODUCTION HARDENING        |
                                  |  - Sub-50ms Low-Latency Inference     |
                                  |  - 100.0% Recall @ 1.0% FPR Budget    |
                                  |  - Real-Time Decision Intelligence    |
                                  +---------------------------------------+
```

---

## 🎯 The 4 Simulated Attack Vectors

| Attack Vector | Category | Attack Methodology & Evasion Strategy |
| :--- | :--- | :--- |
| **Attack 1 · Phishing** | Inferred Personalization | Generates context-aware SMS/Email alerts injecting target financial institution name and realistic transaction amounts to maximize victim click-through rate. |
| **Attack 2 · Vishing** | Cloned Voice Wire | Synthesizes high-urgency bank fraud desk transcripts inducing Authorized Push Payment (APP) transfers. Utilizes defensive synthetic personas (zero real-voice data collected). |
| **Attack 3 · Camouflage** | Adversarial Structuring | Deconstructs large fraud payments (e.g. $15,000) into 4–6 micro-transfers (e.g. $2,500) spaced minutes apart, completely evading static single-transaction threshold rules. |
| **Attack 4 · Agentic Injection** | Autonomous Hijack | Embeds prompt injection instructions inside untrusted product review fields to hijack autonomous shopping agents into executing unauthorized bulk orders. |

---

## 📊 Grounding Datasets & Real-World Feasibility

To ensure real-world statistical fidelity without risking customer PII, the framework grounds its synthetic generation on two industry benchmarks:
- **[Sparkov Fraud Simulation](https://www.kaggle.com/datasets/kartik2112/fraud-detection)** *(CC0 Public Domain)*: Grounding distributions for cardholder demographics, merchant categories, and diurnal transaction timing.
- **[PaySim Mobile Money Simulation](https://www.kaggle.com/datasets/ealaxi/paysim1)** *(CC BY 4.0)*: Grounding distributions for account balance dynamics, multi-step cash-out velocities, and wire transfer topology.

---

## 📂 Project Directory Structure

```text
Mastercard-Innovation-Challenge/
├── app/
│   └── frontend/
│       └── dashboard.py          # Full interactive 3-screen Gradio dashboard
├── configs/
│   └── base.yaml                 # Pipeline configuration (data paths, hyperparams, FPR budgets)
├── data/
│   ├── ai_studio_prompts/        # Google AI Studio system and user generation prompts
│   ├── raw/                      # Downloaded Sparkov & PaySim base datasets
│   └── synthetic/                # 14,000 synthesized payment events
│       ├── attack_1/events.json  # 1,000 Phishing events
│       ├── attack_2/events.json  # 1,000 Vishing events
│       ├── attack_3/events.json  # 1,000 Camouflage Structuring events
│       ├── attack_4/events.json  # 1,000 Agentic Injection events
│       └── baseline/events.json  # 10,000 Legitimate baseline events
├── detectors/
│   ├── features/
│   │   └── pipeline.py           # Feature engineering (24h velocity, delta-time, channel encodings)
│   └── models/
│       └── train.py              # XGBoost, Isolation Forest & TreeSHAP trainer
├── docs/
│   ├── architecture.md           # End-to-end technical system architecture
│   ├── problem_framing.md        # Adversarial landscape and defense justification
│   ├── known_limitations.md      # Ground truth scoping and transparent constraints
│   └── voice_channel_note.md     # Ethical guardrails on synthetic persona generation
├── eval/
│   ├── detection.py              # Confusion matrix, AUC-PR, and Recall@1%FPR evaluator
│   └── fidelity.py               # Statistical distribution and entropy evaluators
├── generators/
│   ├── attack_1/generate.py      # LLM generator for Phishing attacks
│   ├── attack_2/generate.py      # LLM generator for Vishing scripts
│   ├── attack_3/generate.py      # LLM generator for Camouflage structuring
│   ├── attack_4/generate.py      # LLM generator for Agentic prompt injection
│   └── shared/llm.py             # Resilient Gemini / Gemma API client
├── models/                       # Serialized trained model artifacts
│   ├── primary_classifier.joblib # XGBoost Classifier
│   ├── anomaly_detector.joblib   # Isolation Forest
│   └── shap_explainer.joblib     # TreeSHAP Explainer
├── reports/
│   ├── case_study_filled.md      # Empirical Closed-Loop Retraining Case Study
│   ├── eval_summary.json         # Automated evaluation output metrics
│   └── eval_summary.md           # Formatted evaluation report
├── schema/
│   └── event.py                  # Canonical Pydantic schemas (PaymentEvent)
├── scripts/
│   ├── main.py                   # Master CLI entrypoint (generate, train, eval, serve)
│   ├── generate_baseline.py      # Generates 10k legitimate baseline records
│   ├── run_eval.py               # Automated evaluation harness
│   └── setup_space.py            # Bundles standalone Hugging Face Space package
├── space/                        # Standalone Hugging Face Space repository bundle
│   ├── app.py                    # Standalone Space dashboard
│   ├── requirements.txt          # Space production dependencies
│   └── README.md                 # HF Space model-card
└── requirements.txt              # Root development dependencies
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Jay2219/Ouroboros.git
cd Ouroboros

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key (Optional)

The dashboard includes **14,000 pre-generated, cached synthetic records** ready for instantaneous offline demonstration. To generate new attack variants via Gemini API, set your key in `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run Pipeline via Master CLI

```bash
# Run complete end-to-end pipeline (Train -> Eval -> Launch Dashboard)
python scripts/main.py --step all

# Or run individual stages:
python scripts/main.py --step train   # Train XGBoost + Isolation Forest + TreeSHAP
python scripts/main.py --step eval    # Generate evaluation metrics and case study
python scripts/main.py --step serve   # Launch the Gradio interactive dashboard
```

---

## 🖥️ Interactive Dashboard (3 Core Screens)

1. **⚡ Screen 1: Attack Generator (Synthetic Adversary)**
   * One-click presets for all 4 attack vectors.
   * Instant in-memory cache switching (0ms latency, zero API refusals).
   * Interactive dark Plotly timeline for Camouflage structuring vs. the $3,000 threshold rule.
2. **🛡️ Screen 2: Live Detection (Explainable ML)**
   * Sub-50ms real-time scoring via compiled XGBoost classifier and Isolation Forest.
   * Live horizontal TreeSHAP feature impact chart (`Red = Risk Increase`, `Green = Cleared`).
   * Color-coded hero triage badges (Critical Risk / Step-Up 2FA / Cleared).
3. **🔄 Screen 3: Closed-Loop Proof (Adaptive Retraining)**
   * Hero Before vs. After comparison chart demonstrating **+85.0% recall gain**.
   * 4-Phase narrative breakdown of the red-team discovery and blue-team retraining cycle.
   * Full confusion matrix and precision/recall breakdown at strict 1.0% FPR.

---

## 📈 Benchmark & Evaluation Results

Evaluated across **4,200 unseen test events** (1,200 fraud attacks + 3,000 legitimate baseline events) under a strict **1.0% False Positive Rate (FPR)** budget constraint:

| Metric | Static Rule Baseline ($3k) | Ouroboros Retrained ML Defense | Improvement |
| :--- | :---: | :---: | :---: |
| **Camouflage Structuring Recall** | 15.0% | **100.0%** | **+85.0% Gain** |
| **Overall Detection Recall (@ 1% FPR)** | 32.5% | **100.0%** | **+67.5% Gain** |
| **Precision** | 97.5% | **97.6%** | Stable |
| **False Positive Rate (FPR)** | 0.8% | **1.0%** | Within Budget |
| **F1-Score** | 0.487 | **0.988** | **+0.501** |
| **AUC-PR** | 0.420 | **0.9996** | **+0.5796** |
| **Inference Latency** | < 1 ms | **~35 ms** | Sub-50ms SLA |

---

## 🔒 Security, Privacy & Ethics Guardrails

- **Zero Real-Voice Clones:** Voice simulations use standardized synthetic persona scripts. No real biometrics or voice samples are ingested.
- **Defensive Containment:** All generated synthetic payloads are confined to local evaluation environments and formatted strictly into defensive training features.
- **Explainability Standard:** Every scoring decision provides exact TreeSHAP attributions to satisfy regulatory transparency (FCRA / GDPR Article 22).

---

## 📜 Attribution & License

- **Datasets:** [Sparkov Fraud Simulation](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (CC0 Public Domain) & [PaySim Mobile Money](https://www.kaggle.com/datasets/ealaxi/paysim1) (CC BY 4.0).
- **License:** Released under the [MIT License](LICENSE).
- **Challenge:** Built for the **Mastercard Innovation Challenge 2026**.
