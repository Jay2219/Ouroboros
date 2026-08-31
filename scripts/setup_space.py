import os
import shutil
import json

def setup_space():
    space_dir = "space"
    os.makedirs(space_dir, exist_ok=True)
    os.makedirs(os.path.join(space_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(space_dir, "schema"), exist_ok=True)
    os.makedirs(os.path.join(space_dir, "detectors", "features"), exist_ok=True)
    os.makedirs(os.path.join(space_dir, "docs"), exist_ok=True)
    
    # 1. Copy Models
    for m in ["primary_classifier.joblib", "anomaly_detector.joblib", "shap_explainer.joblib"]:
        src = os.path.join("models", m)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(space_dir, "models", m))
            print(f"Copied {src} -> space/models/{m}")
            
    # 2. Copy Synthetic Datasets (Attacks + Baseline)
    for category in ["attack_1", "attack_2", "attack_3", "attack_4", "baseline"]:
        dst_dir = os.path.join(space_dir, "data", "synthetic", category)
        os.makedirs(dst_dir, exist_ok=True)
        src = os.path.join("data", "synthetic", category, "events.json")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, "events.json"))
            print(f"Copied {src} -> {dst_dir}/events.json")
            
    # 3. Copy schema & feature extraction
    shutil.copy2(os.path.join("schema", "event.py"), os.path.join(space_dir, "schema", "event.py"))
    shutil.copy2(os.path.join("detectors", "features", "pipeline.py"), os.path.join(space_dir, "detectors", "features", "pipeline.py"))
    open(os.path.join(space_dir, "schema", "__init__.py"), "a").close()
    open(os.path.join(space_dir, "detectors", "__init__.py"), "a").close()
    open(os.path.join(space_dir, "detectors", "features", "__init__.py"), "a").close()
    
    # 4. Copy docs
    for doc in ["architecture.md", "problem_framing.md", "known_limitations.md", "voice_channel_note.md"]:
        src = os.path.join("docs", doc)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(space_dir, "docs", doc))
            
    print("Space directory asset copying complete.")

if __name__ == "__main__":
    setup_space()
