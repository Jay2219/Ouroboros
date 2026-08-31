import argparse
import yaml
import sys
import os
import joblib
import json
import subprocess

# Ensure we can import from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generators.attack_1.generate import generate_batch as gen_1
from generators.attack_2.generate import generate_batch as gen_2
from generators.attack_3.generate import generate_batch as gen_3
from generators.attack_4.generate import generate_batch as gen_4
from detectors.models.train import train_primary_classifier, train_anomaly_detector
from scripts.run_eval import run_evaluation
from schema.event import PaymentEvent

def parse_args():
    parser = argparse.ArgumentParser(description="AI Defense Lab Main Entrypoint")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to experiment config")
    parser.add_argument("--step", type=str, choices=["generate", "train", "eval", "serve", "all"], default="all")
    return parser.parse_args()

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_generation(config):
    print("--- Running Generation Phase ---")
    # Generating attacks automatically saves them to data/synthetic/attack_X/events.json
    samples = config.get("generation", {}).get("target_samples_per_attack", 5)
    
    print("Generating Attack 1 (Phishing)...")
    gen_1(samples)
    print("Generating Attack 2 (Vishing)...")
    gen_2(samples)
    print("Generating Attack 3 (Camouflage)...")
    gen_3(samples // 2 if samples > 2 else 2) # sequence generator produces multiple per call
    print("Generating Attack 4 (Agentic Injection)...")
    gen_4(samples)
    print("Generation complete and cached to disk.")

def load_all_events():
    events = []
    base_dir = "data/synthetic"
    for attack_dir in os.listdir(base_dir):
        path = os.path.join(base_dir, attack_dir, "events.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                for item in data:
                    events.append(PaymentEvent(**item))
    return events

def run_training(config):
    print("--- Running Training Phase ---")
    events = load_all_events()
    if not events:
        print("No events found. Please run the generation step first.")
        return

    os.makedirs("models", exist_ok=True)
    
    print(f"Training on {len(events)} synthetic events...")
    
    # Train Primary Classifier (XGBoost) and SHAP explainer
    clf, explainer = train_primary_classifier(events)
    if clf:
        joblib.dump(clf, "models/primary_classifier.joblib")
        joblib.dump(explainer, "models/shap_explainer.joblib")
        print("Primary classifier and explainer saved to models/")

    # Train Anomaly Detector (Isolation Forest)
    iso_forest = train_anomaly_detector(events)
    if iso_forest:
        joblib.dump(iso_forest, "models/anomaly_detector.joblib")
        print("Anomaly detector saved to models/")

def run_eval(config):
    print("--- Running Evaluation Phase ---")
    run_evaluation()

def run_serve(config):
    print("--- Running Serve Phase (Gradio) ---")
    # Launch gradio app in a subprocess so it stays running
    subprocess.run([sys.executable, "app/frontend/dashboard.py"])

def main():
    args = parse_args()
    config = load_config(args.config)
    print(f"Loaded config: {config.get('experiment_name', 'AI Defense Lab')}")
    
    if args.step in ["generate", "all"]:
        run_generation(config)
    if args.step in ["train", "all"]:
        run_training(config)
    if args.step in ["eval", "all"]:
        run_eval(config)
    if args.step in ["serve", "all"]:
        run_serve(config)

if __name__ == "__main__":
    main()
