import os
import subprocess
import sys

def download_dataset(dataset: str, out_path: str):
    print(f"Downloading {dataset}...")
    try:
        # We use sys.executable -m kaggle to avoid Windows PATH resolution issues
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", out_path, "--unzip"], check=True)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure you have configured your kaggle.json file in ~/.kaggle/kaggle.json")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure raw data directory exists
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"Downloading datasets to: {raw_dir}")
    
    # 1. Sparkov (for Attacks 1, 2, 4)
    download_dataset("kartik2112/fraud-detection", raw_dir)
    
    # 2. PaySim (for Attack 3)
    download_dataset("ealaxi/paysim1", raw_dir)
    
    print("\nDatasets successfully downloaded and unzipped!")
