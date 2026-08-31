import numpy as np
from scipy.stats import ks_2samp, entropy

def evaluate_fidelity_ks(real_data: np.ndarray, synthetic_data: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov test for continuous distributions (e.g., transaction amounts).
    Returns the KS statistic and p-value.
    """
    if len(real_data) == 0 or len(synthetic_data) == 0:
        return {"statistic": None, "p_value": None, "passed": False}
        
    stat, p_value = ks_2samp(real_data, synthetic_data)
    # Passed if p-value > 0.05 (cannot reject null hypothesis that they are from the same distribution)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "passed": p_value > 0.05
    }

def evaluate_fidelity_js(real_data_probs: np.ndarray, synthetic_data_probs: np.ndarray) -> float:
    """
    Jensen-Shannon divergence for categorical/binned distributions.
    Expects probability vectors that sum to 1.
    """
    m = 0.5 * (real_data_probs + synthetic_data_probs)
    js_divergence = 0.5 * entropy(real_data_probs, m) + 0.5 * entropy(synthetic_data_probs, m)
    return float(js_divergence)

def run_fidelity_report(attack_id: str, real_features: np.ndarray, synthetic_features: np.ndarray) -> dict:
    """Stub to generate the fidelity report for a specific attack type."""
    return {
        "attack_id": attack_id,
        "ks_test": evaluate_fidelity_ks(real_features, synthetic_features)
    }
