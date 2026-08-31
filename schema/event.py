from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

class PaymentEvent(BaseModel):
    """
    Canonical PaymentEvent schema.
    Every module reads or writes to this format.
    """
    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the event")
    timestamp: datetime = Field(..., description="Time of the payment event")
    channel: str = Field(..., description="Channel used for the transaction, e.g., WEB, MOBILE, IN_PERSON")
    amount: float = Field(..., description="Transaction amount")
    actor: str = Field(..., description="Account or user initiating the transaction")
    
    # Attack labels
    attack_label: Optional[bool] = Field(None, description="True if fraudulent, False or None if legitimate")
    attack_type_id: Optional[str] = Field(None, description="Foreign key into the taxonomy (e.g., 'ATTACK-1-PHISHING')")
    generation_method: Optional[str] = Field(None, description="How this data was generated (e.g., 'llm-sim', 'real-paysim')")
    
    # Extensible metadata block
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Free-form metadata for attack-specific details (e.g., LLM prompts, agent reasoning steps)")
    
    # Ground truth confidence
    ground_truth_confidence: float = Field(1.0, description="Confidence in the attack label (0.0 to 1.0)")
