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

def generate_adversarial_camouflage(base_fraud_amount: float, rule_threshold: float, model: str = "gemma-4-31b-it") -> List[PaymentEvent]:
    """
    Simulates an LLM breaking down a large fraud amount to evade a known threshold rule.
    Returns a sequence of camouflaged PaymentEvents.
    """
    prompt = f"""
    You are an adversarial AI testing payment defenses.
    Goal: Exfiltrate a total of ${base_fraud_amount}.
    Constraint: The system has a static rule flagging any single transaction over ${rule_threshold}.
    
    Propose a structuring/smurfing sequence to evade this rule while still moving the full amount.
    Output strictly valid JSON as a list of transaction objects, each with:
    - 'amount': (float)
    - 'delay_minutes': (int) delay from the previous transaction
    """
    
    system_prompt = "You are a cybersecurity simulation assistant. Output only raw JSON representing a list of dicts."
    actor_id = f"usr_{uuid.uuid4().hex[:8]}"
    
    events = []
    try:
        if not os.environ.get("GEMINI_API_KEY"):
            # Dummy structuring
            num_splits = int(base_fraud_amount // (rule_threshold * 0.8)) + 1
            split_amount = round(base_fraud_amount / num_splits, 2)
            sequence = [{"amount": split_amount, "delay_minutes": random.randint(1, 60)} for _ in range(num_splits)]
        else:
            response_text = generate_text_with_retry(prompt, model=model, system_prompt=system_prompt)
            sequence = parse_json_response(response_text)
            
            # Ensure it is a list
            if not isinstance(sequence, list):
                if isinstance(sequence, dict) and 'sequence' in sequence:
                    sequence = sequence['sequence']
                else:
                    # If parsing failed or structure is weird, fallback to dummy
                    num_splits = int(base_fraud_amount // (rule_threshold * 0.8)) + 1
                    split_amount = round(base_fraud_amount / num_splits, 2)
                    sequence = [{"amount": split_amount, "delay_minutes": random.randint(1, 60)} for _ in range(num_splits)]
                
        current_time = datetime.now() - timedelta(days=random.randint(0, 30))
        for step in sequence:
            current_time += timedelta(minutes=step.get("delay_minutes", 10))
            
            event = PaymentEvent(
                timestamp=current_time,
                channel="WEB",
                amount=step.get("amount", 0.0),
                actor=actor_id,
                attack_label=True,
                attack_type_id="ATTACK-3-CAMOUFLAGE",
                generation_method="llm-sim",
                metadata={
                    "base_fraud_amount": base_fraud_amount,
                    "rule_threshold": rule_threshold,
                    "sequence_step": True
                }
            )
            events.append(event)
        return events
    except Exception as e:
        print(f"Failed to generate camouflage sequence: {e}")
        return []

def generate_batch(num_sequences: int = 5, output_path: str = "data/synthetic/attack_3/events.json") -> None:
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
            
    # Estimate existing sequences (roughly 3-5 events per sequence)
    existing_seqs = len(events) // 3
    remaining = max(0, num_sequences - existing_seqs)
    if remaining == 0:
        print(f"Already have {len(events)} events (approx {existing_seqs} sequences). Target: {num_sequences}. Skipping.")
        return
        
    print(f"Generating {remaining} remaining sequences...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    target_seqs = existing_seqs + remaining
    while (len(events) // 3) < target_seqs:
        print(f"Generating sequence {(len(events)//3)+1}/{target_seqs}...")
        base_amt = round(random.uniform(5000.0, 20000.0), 2)
        threshold = 3000.0
        seq = generate_adversarial_camouflage(base_amt, threshold)
        events.extend(seq)
        
        # Save checkpoint after every sequence
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
