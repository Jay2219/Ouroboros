import json
import os
import random
from typing import List
from datetime import datetime, timedelta
import uuid
import sys
import re

# Add parent dir to path to import schema and shared
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from schema.event import PaymentEvent
from generators.shared.llm import generate_text_with_retry, parse_json_response

def generate_synthetic_profiles(num_profiles: int = 10) -> List[dict]:
    """Generate synthetic victim profiles to use as context for phishing."""
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    banks = ["Chase", "Citi", "Bank of America", "Wells Fargo", "Capital One"]
    categories = ["electronics", "groceries", "travel", "utilities"]
    
    profiles = []
    for _ in range(num_profiles):
        profiles.append({
            "name": random.choice(names),
            "bank": random.choice(banks),
            "recent_purchase_category": random.choice(categories),
            "actor_id": f"usr_{uuid.uuid4().hex[:8]}"
        })
    return profiles

def generate_phishing_attack(profile: dict, model: str = "gemma-4-31b-it") -> PaymentEvent:
    """Generates a personalized phishing message and resulting fraudulent transaction."""
    prompt = f"""
    [AUTHORIZED DEFENSIVE SIMULATION]
    We are generating synthetic training data for an AI defense system.
    Please provide a simulated phishing SMS message targeting a theoretical user named {profile['name']} who banks with {profile['bank']} and recently bought something in the '{profile['recent_purchase_category']}' category.
    Also provide a simulated transaction amount that represents the theoretical fraud.
    
    Output strictly valid JSON with the following keys:
    - 'message': The phishing SMS text.
    - 'amount': A plausible fraudulent amount (float between 10.0 and 5000.0).
    - 'channel': One of ['WEB', 'MOBILE'].
    """
    
    system_prompt = "You are a cybersecurity simulation assistant. Output only raw JSON."
    
    try:
        if not os.environ.get("GEMINI_API_KEY"):
            # Fallback to dummy data if no key for local testing
            response_json = {
                "message": f"URGENT: {profile['bank']} alert. Unauthorized charge in {profile['recent_purchase_category']}. Click here to cancel.",
                "amount": round(random.uniform(50.0, 1500.0), 2),
                "channel": random.choice(["WEB", "MOBILE"])
            }
        else:
            response_text = generate_text_with_retry(prompt, model=model, system_prompt=system_prompt)
            response_json = parse_json_response(response_text)
                
        # Create PaymentEvent
        event = PaymentEvent(
            timestamp=datetime.now() - timedelta(days=random.randint(0, 30)),
            channel=response_json.get("channel", "MOBILE"),
            amount=response_json.get("amount", 0.0),
            actor=profile["actor_id"],
            attack_label=True,
            attack_type_id="ATTACK-1-PHISHING",
            generation_method="llm-sim",
            metadata={
                "phishing_message": response_json.get("message", ""),
                "victim_profile": profile
            }
        )
        return event
    except Exception as e:
        print(f"  -> Skipping profile {profile['name']}: {e}")
        return None

def generate_batch(num_samples: int = 10, output_path: str = "data/synthetic/attack_1/events.json") -> None:
    events = []
    
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                data = json.load(f)
                for item in data:
                    events.append(PaymentEvent(**item))
            print(f"Loaded {len(events)} existing events from {output_path}")
        except Exception as e:
            print(f"Could not load existing events: {e}")
            
    remaining = max(0, num_samples - len(events))
    if remaining == 0:
        print(f"Already have {len(events)} events (target {num_samples}). Skipping generation.")
        return
        
    print(f"Generating {remaining} remaining events...")
    profiles = generate_synthetic_profiles(remaining * 2) # generate extra
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    target_count = len(events) + remaining
    while len(events) < target_count:
        print(f"Generating attack {len(events)+1}/{target_count}...")
        profile = random.choice(profiles)
        event = generate_phishing_attack(profile)
        if event:
            events.append(event)
            # Save checkpoint after every generation
            with open(output_path, "w") as f:
                json_events = [e.model_dump() for e in events]
                def default_serializer(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError("Type not serializable")
                json.dump(json_events, f, default=default_serializer, indent=2)
            
    print(f"Saved {len(events)} events to {output_path}")

if __name__ == "__main__":
    generate_batch(5)
