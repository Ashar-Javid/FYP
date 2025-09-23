"""
RIS Multi-User Optimization Framework Configuration
Handles all configuration settings and API setup for the framework.
"""

import os
from typing import Dict, Any
import json

# API Configuration
CEREBRAS_API_KEY = "csk-9m6k32m2tytp959v3rhttjrwm66645cymhchjj83yxkyh8nw"
CEREBRAS_MODEL = "llama3.1-8b"

# Framework Settings
FRAMEWORK_CONFIG = {
    "max_iterations": 10,
    "power_range_dB": [20, 45],
    "convergence_tolerance": 0.1,
    "memory_file": "framework_memory.json",
    "results_dir": "results",
    "plots_dir": "plots",
    "logs_dir": "logs",
    "metrics_dir": "metrics"
}

# RAG Configuration
RAG_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.7,
    "max_retrieved_scenarios": 5,
    "feature_weights": {
        "user_locations": 0.3,
        "channel_conditions": 0.3,
        "delta_snr_values": 0.4
    }
}

# Coordinator Agent Configuration
COORDINATOR_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 1000,
    "system_prompt": """You are a Coordinator Agent responsible for optimizing Quality of Service (QoS) in a multi-user RIS-assisted 6G system.

Your key responsibilities:
1. Analyze user scenarios including locations, channel conditions, and delta SNR values
2. Learn patterns from historical data to make optimal user selection decisions
3. Select the best optimization algorithm from {Analytical, GD, Manifold, AO}
4. Decide on base station transmit power adjustments within 20-45 dB range
5. Use historical iteration feedback to improve future decisions

Key Guidelines:
- Select users for optimization based on learned patterns, not fixed thresholds
- Consider user locations, channel conditions (LoS/NLoS, fading), and current delta SNR
- Prioritize scenarios where optimization can provide meaningful improvements
- Adapt power levels based on scenario complexity and user distribution
- Learn from evaluator feedback to avoid repeating poor decisions

Output format should be JSON with: selected_users, selected_algorithm, base_station_power_change"""
}

# Evaluator Agent Configuration
EVALUATOR_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 800,
    "system_prompt": """You are an Evaluator Agent that assesses the performance of RIS optimization decisions.

Your responsibilities:
1. Compare delta SNR values before and after optimization
2. Evaluate power efficiency - flag if users have excessively positive delta SNR (wasted power)
3. Assess fairness across users
4. Provide structured feedback for the Coordinator Agent
5. Recommend power adjustments when needed

Power Efficiency Guidelines:
- If multiple users have delta SNR > 5 dB, recommend power reduction
- If users struggle to meet requirements, recommend power increase (within 20-45 dB range)
- Balance individual user needs with overall system efficiency

Output format should include: performance_summary, power_efficiency_status, fairness_assessment, recommendations"""
}

def ensure_directories():
    """Create necessary directories for the framework."""
    os.makedirs(FRAMEWORK_CONFIG["results_dir"], exist_ok=True)
    os.makedirs(FRAMEWORK_CONFIG["plots_dir"], exist_ok=True)
    os.makedirs(FRAMEWORK_CONFIG["logs_dir"], exist_ok=True)
    os.makedirs(FRAMEWORK_CONFIG["metrics_dir"], exist_ok=True)

def load_framework_memory() -> Dict[str, Any]:
    """Load persistent framework memory."""
    memory_file = FRAMEWORK_CONFIG["memory_file"]
    if os.path.exists(memory_file):
        try:
            with open(memory_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            # Backup corrupted file
            backup_path = memory_file + ".corrupt"
            try:
                import shutil
                shutil.copy(memory_file, backup_path)
                print(f"⚠️  Corrupted memory file detected. Backup created at {backup_path}. Reinitializing memory.")
            except Exception as be:
                print(f"⚠️  Failed to backup corrupted memory file: {be}")
        except Exception as e:
            print(f"⚠️  Error loading memory: {e}. Reinitializing.")
    return {
        "learned_patterns": [],
        "iteration_history": [],
        "successful_strategies": [],
        "failed_strategies": []
    }

def save_framework_memory(memory: Dict[str, Any]):
    """Save framework memory to persistent storage."""
    memory_file = FRAMEWORK_CONFIG["memory_file"]
    
    # Make memory JSON serializable by converting numpy arrays
    def make_serializable(obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    serializable_memory = make_serializable(memory)
    
    with open(memory_file, 'w') as f:
        json.dump(serializable_memory, f, indent=2)

# Export key configurations
__all__ = [
    'CEREBRAS_API_KEY', 'CEREBRAS_MODEL', 'FRAMEWORK_CONFIG', 'RAG_CONFIG',
    'COORDINATOR_CONFIG', 'EVALUATOR_CONFIG', 'ensure_directories',
    'load_framework_memory', 'save_framework_memory'
]