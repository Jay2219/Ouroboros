import pandas as pd
import numpy as np
from typing import List, Tuple
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from schema.event import PaymentEvent

def extract_features(events: List[PaymentEvent]) -> pd.DataFrame:
    """
    Converts a list of PaymentEvents into a pandas DataFrame of numeric/categorical features.
    This is a shared pipeline across all attack types.
    """
    records = []
    for event in events:
        records.append({
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "amount": event.amount,
            "channel": event.channel,
            "actor": event.actor,
            "attack_label": event.attack_label if event.attack_label is not None else False,
            "attack_type_id": event.attack_type_id
        })
        
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Feature Engineering
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # 1. Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 2. Sequence/Velocity features (crucial for Attack 3 - Camouflage)
    # Sort by time to calculate rolling features
    df = df.sort_values(by=['actor', 'timestamp'])
    
    # Calculate time diff from previous transaction for the same actor
    df['time_since_last_tx_min'] = df.groupby('actor')['timestamp'].diff().dt.total_seconds() / 60.0
    df['time_since_last_tx_min'] = df['time_since_last_tx_min'].fillna(9999.0) # Large value for first tx
    
    # Calculate rolling count and sum of transactions in the last 24 hours per actor
    # We set index to timestamp for rolling, then reset
    df_rolling = df.set_index('timestamp').groupby('actor')['amount'].rolling('24h')
    df['tx_count_24h'] = df_rolling.count().values
    df['tx_amount_24h'] = df_rolling.sum().values
    
    # 3. Categorical Encodings (Dummy variables for channel)
    # Ensure all expected channels exist
    expected_channels = ['WEB', 'MOBILE', 'IN_PERSON']
    for ch in expected_channels:
        df[f'channel_{ch}'] = (df['channel'] == ch).astype(int)
        
    # Drop raw categorical columns and IDs for model readiness
    # Keep target label and event_id for splitting/tracking
    feature_cols = [
        'amount', 'hour', 'day_of_week', 'is_weekend',
        'time_since_last_tx_min', 'tx_count_24h', 'tx_amount_24h',
        'channel_WEB', 'channel_MOBILE', 'channel_IN_PERSON'
    ]
    
    return df[['event_id', 'attack_label', 'attack_type_id'] + feature_cols]

def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separates features (X) and target (y)."""
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
        
    y = df['attack_label'].astype(int)
    # Drop identifiers and labels from X
    X = df.drop(columns=['event_id', 'attack_label', 'attack_type_id'])
    return X, y
