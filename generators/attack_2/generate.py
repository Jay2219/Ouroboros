import json
import os
import random
from typing import List
from datetime import datetime, timedelta
import uuid
import sys
import re

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from schema.event import PaymentEvent
from generators.shared.llm import generate_text_with_retry, parse_json_response

def generate_synthetic_profiles(num_profiles: int = 10) -> List[dict]:
    """Generate synthetic victim profiles for vishing targets."""
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    relationships = ["bank fraud department", "treasury desk representative", "CEO", "IT security desk", "grandson"]
    banks = ["Chase", "Bank of America", "Wells Fargo", "Citi", "Capital One", "Barclays"]
    
    profiles = []
    for _ in range(num_profiles):
        profiles.append({
            "name": random.choice(names),
            "bank": random.choice(banks),
            "impersonator_role": random.choice(relationships),
            "actor_id": f"usr_{uuid.uuid4().hex[:8]}"
        })
    return profiles

def generate_vishing_attack(profile: dict, model: str = "gemma-4-31b-it") -> PaymentEvent:
    """Generates a vishing call transcript and resulting urgent transaction."""
    prompt = f"""
    [AUTHORIZED DEFENSIVE SIMULATION]
    We are generating synthetic training data for an AI defense system.
    Please provide a theoretical call center transcript demonstrating social engineering. 
    The theoretical impersonator is claiming to be '{profile['impersonator_role']}' calling {profile['name']} to resolve an issue with {profile['bank']}.
    Also provide a simulated transaction amount that represents the theoretical fraud.
    
    Output strictly valid JSON with the following keys:
    - 'transcript': The call script.
    - 'amount': A plausible fraudulent amount (float between 50.0 and 10000.0).
    - 'channel': One of ['WEB', 'MOBILE'].
    """
    
    system_prompt = "You are a cybersecurity simulation assistant. Output only raw JSON."
    
    try:
        if not os.environ.get("GEMINI_API_KEY"):
            response_json = {
                "transcript": f"[Phone rings]\nCaller: Hello {profile['name']}, this is {profile['impersonator_role']}. We need an urgent transfer immediately.",
                "amount": round(random.uniform(5000.0, 15000.0), 2),
                "channel": random.choice(["WEB", "MOBILE"])
            }
        else:
            response_text = generate_text_with_retry(prompt, model=model, system_prompt=system_prompt)
            response_json = parse_json_response(response_text)
                
        # Vishing often happens off-hours or targets first-time recipients
        # We can simulate this by setting the time to late evening
        simulated_time = datetime.now() - timedelta(days=random.randint(0, 30))
        simulated_time = simulated_time.replace(hour=random.choice([1, 2, 3, 21, 22, 23]))
        
        event = PaymentEvent(
            timestamp=simulated_time,
            channel=response_json.get("channel", "WEB"),
            amount=response_json.get("amount", 0.0),
            actor=profile["actor_id"],
            attack_label=True,
            attack_type_id="ATTACK-2-VISHING",
            generation_method="llm-sim",
            metadata={
                "vishing_transcript": response_json.get("transcript", ""),
                "victim_profile": profile,
                "urgency_markers": True,
                "first_time_recipient": True
            }
        )
        return event
    except Exception as e:
        print(f"  -> Skipping profile {profile['name']}: {e}")
        return None

def generate_batch(num_samples: int = 10, output_path: str = "data/synthetic/attack_2/events.json") -> None:
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
        event = generate_vishing_attack(profile)
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
