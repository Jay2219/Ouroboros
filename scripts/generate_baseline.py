import json
import os
import uuid
import random
from datetime import datetime, timedelta

def generate_baseline(num_samples=10000, output_path="data/synthetic/baseline/events.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    channels = ["WEB", "MOBILE", "IN_PERSON"]
    
    events = []
    now = datetime.utcnow()
    
    # Generate some actors to simulate return users
    actors = [f"usr_{uuid.uuid4().hex[:8]}" for _ in range(num_samples // 5)]
    
    for _ in range(num_samples):
        # Random timestamp in the last 30 days
        days_ago = random.uniform(0, 30)
        ts = now - timedelta(days=days_ago)
        
        # Log-normal distribution for amounts (mostly small, some large)
        amount = round(random.lognormvariate(3.5, 1.0), 2)
        if amount < 1.0:
            amount = 1.0
            
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": ts.isoformat() + "Z",
            "channel": random.choice(channels),
            "amount": amount,
            "actor": random.choice(actors),
            "attack_label": False,
            "attack_type_id": "BASELINE",
            "generation_method": "synthetic-baseline",
            "metadata": {}
        }
        events.append(event)
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)
        
    print(f"Generated {num_samples} baseline (legitimate) events at {output_path}")

if __name__ == "__main__":
    generate_baseline()
