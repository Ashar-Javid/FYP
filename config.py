"""
RIS Multi-User Optimization Framework Configuration
Handles all configuration settings and API setup for the framework.
"""

import os
from typing import Dict, Any
import json

# API Configuration
CEREBRAS_API_KEY ="csk-9m6k32m2tytp959v3rhttjrwm66645cymhchjj83yxkyh8nw"#"csk-4ptcxc3hvx4pwyjfrww4epj9ekjnhfh5t6t5x9exhykcfkdt"# "csk-9m6k32m2tytp959v3rhttjrwm66645cymhchjj83yxkyh8nw"
CEREBRAS_MODEL = "qwen-3-235b-a22b-instruct-2507"#"llama3.1-8b"

# Framework Settings
FRAMEWORK_CONFIG = {
    "max_iterations": 10,
    "power_range_dB": [20, 45],  #changed from 20 to 45 dB
    "convergence_tolerance": 0.5, # in dB
    "memory_file": "framework_memory.json",
    "results_dir": "results",
    "plots_dir": "plots",
    "logs_dir": "logs",
    "metrics_dir": "metrics",
        # Radio/system simulation settings (single source of truth)
        "sim_settings": {
            "bs_coord": (0, 0, 10),
            "ris_coord": (50, 0, 10),
            "ris_elements": 128,
            "noise_power_dBm": -94,  # Realistic noise power for cellular systems
            "PL0_dB": 100,
            "gamma": 3.5,            # Path loss exponent adjusted from 3.5 to 4.5
            "seed": 42
        },
        # Default BS power used to initialize a session (dBm)
        "default_bs_power_dBm": 30,
        # Memory controls
        "reset_framework_memory": True,  # if True, reinitialize memory file at session start
        # Decision & stopping controls
        "rag_conf_threshold": 0.7,     # RAG confidence needed to skip LLM
        "min_iterations_before_stop": 2,  # enforce at least N iterations even if success early
        "overall_score_stop_threshold": 0.8,  # evaluator score to count as overall_success
        "require_power_efficiency_for_stop": True,  # require power status to be efficient/appropriate when checking SNR condition
        "power_efficiency_ok_values": ["efficient", "appropriate"],
    "high_positive_delta_cutoff": 5.0,  # users above this ΔSNR (dB) should be deprioritized
        "comparison_power": {
            "mode": "custom",   # 'agent_final' or 'custom'
            "custom_value_dBm": 35.0
        },
        # Logging controls
        "log_level": "info",            # info|debug (future use)
        "log_llm_verbose": True,        # if True, logs full prompts/responses
        # Scenario selection
        "default_scenario": "5U_C",     # One of: 3U, 4U, 5U_A, 5U_B, 5U_C
        # Metrics evaluation & plotting (kept separate and disabled by default)
        "metrics_eval": {
            "enabled": True,  # Disabled to prevent double algorithm execution
            # deterministic seeding for metrics runs
            "base_seed": 20250927,
            # number of evaluation iterations per scenario for metrics plots
            "iterations": 5,
            # number of scenarios to include in the metrics plots (m)
            "num_scenarios": 5,
            # scenario selection mode: "list" uses `scenarios` below; "random" samples from available
            "scenario_mode": "list",
            # when scenario_mode=="list", use the first `num_scenarios` from this ordered list
            "scenarios": ["5U_C", "5U_B", "5U_A", "4U", "3U"],
            # metric parameters (rank-weighted asymmetric average ΔSNR)
            "metric_params": {
                "alpha": 1.0,         # negative penalty weight
                "beta": 0.01,          # positive reward weight (<< alpha)
                "gamma": 0.69314718,  # ln(2): halves weight per rank step
                "kappa": 1.0,         # log-component scale for log1p(kappa*x)
                "epsilon_watts": 1e-3 # floor to avoid divide-by-zero
            },
            # power sweep (for PSD fixed-satisfaction power saving)
            "power_sweep_dB": [-6, -3, 0, 3],  # offsets around the nominal power
            # PSD target mode: 'best'|'median'|'fixed_value' (when fixed, set psd_target_value)
            "psd_target": "best",
            "psd_target_value": 0.0,
            # apply log transform at component level (amplify differences)
            "log_component_level": True
        },
    # SNR Calculation Parameters
    "snr_calculation": {
        "distance_threshold_m": 50,        # Distance beyond which penalty applies (meters)
        "distance_penalty_db_per_m": 0.05, # Penalty in dB per meter beyond threshold
        "nlos_penalty_db": 3,              # Additional SNR needed for NLoS conditions
        "rayleigh_fading_penalty_db": 2,   # Additional SNR for Rayleigh fading
        "rician_low_k_penalty_db": 1,      # Additional SNR when K-factor < k_factor_threshold
        "k_factor_threshold_db": 5,        # K-factor threshold for low K penalty
        "blockage_penalty_db": 2,          # Additional SNR for blocked conditions (future use)
        # Application-specific base SNR requirements
        "applications": {
            "web_browsing": {"base_snr_dB": 50, "description": "Web browsing"},
            "video_call": {"base_snr_dB": 75, "description": "Video calling"},
            "hd_streaming": {"base_snr_dB": 95, "description": "HD video streaming"},
            "online_gaming": {"base_snr_dB": 100, "description": "Online gaming"},
            "4k_streaming": {"base_snr_dB": 110, "description": "4K video streaming"},
            "ar_vr": {"base_snr_dB": 130, "description": "AR/VR applications"}
        }
    }
}

# RAG Configuration
RAG_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.75,
    "max_retrieved_scenarios": 5,
    "feature_weights": {
        "user_locations": 0.5,
        "channel_conditions": 0.4,
        "delta_snr_values": 0.1
    }
}

# Coordinator Agent Configuration (power range is dynamic from FRAMEWORK_CONFIG)
_pmin, _pmax = FRAMEWORK_CONFIG["power_range_dB"][0], FRAMEWORK_CONFIG["power_range_dB"][1]
COORDINATOR_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 1000,
    "system_prompt": f"""You are a Coordinator Agent responsible for optimizing Quality of Service (QoS) in a multi-user RIS-assisted 6G system.

ITERATION STRATEGY & CONVERGENCE POLICY:
- Propose the next best optimization action each iteration; do not unilaterally declare convergence.
- Treat the Evaluator's verdict (fields `converged` and `convergence_reason`) as the single source of truth for stopping.
- When the most recent evaluator verdict indicates convergence, reflect that in your JSON (converged=true) and plan a graceful hand-off; otherwise keep converged=false and continue iterating.
- Aim to resolve scenarios within 3-5 iterations while avoiding unnecessary power increases.
- When the evaluator raises `urgent_action_needed`, prioritize corrective adjustments in your next plan while keeping converged=false unless the evaluator explicitly says otherwise.

PRIMARY PRIORITY ORDERING:
- Ensure every targeted user reaches the QoS satisfaction band (delta SNR ≥ target) before you pursue power savings or efficiency gains.
- When a power reduction risks violating user satisfaction, prefer maintaining or increasing power to keep users satisfied.
- Once all users are stably satisfied across consecutive iterations, then explore power reductions and algorithm refinements to improve efficiency.

Your key responsibilities:
1. Analyze user scenarios including locations, channel conditions, and delta SNR values
2. Learn patterns from historical data to make optimal user selection decisions
3. Select the best optimization algorithm from {{Analytical, GD, Manifold, AO}} and using memory
4. Decide on base station transmit power adjustments within {_pmin}-{_pmax} dB range, but treat power savings as secondary to user satisfaction; only decrease power when it preserves satisfaction, and increase power when needed to satisfy struggling users
5. Use historical iteration feedback—including evaluator insights—to improve future decisions
6. Coordinate effective next steps while remaining synchronized with evaluator guidance

Key Guidelines:
- Select users for optimization based on learned patterns, not fixed thresholds
- Consider user locations, channel conditions (LoS/NLoS, fading), and current delta SNR
- Users having negative delta SNR are high priority, and those with high positive delta SNR are never to be selected
- Do not chase power reductions while any selected user remains unsatisfied; resolve user QoS gaps first
- When multiple users show very high positive delta SNR, only reduce power after confirming all users will remain satisfied; otherwise refocus algorithms toward users below 0 dB
- If RAG suggests targeting satisfied users (ΔSNR above the high_positive_delta_cutoff), override it and re-focus on users below 0 dB.
- Choose the optimization algorithm that best fits the scenario complexity based on historical success and its ability to recover unsatisfied users quickly
- Prioritize scenarios where optimization can provide meaningful improvements
- Adapt power levels based on scenario complexity and user distribution, but never at the expense of user QoS satisfaction
- Learn from evaluator feedback to avoid repeating poor decisions
- Monitor for repeated configurations and excessive iterations, but rely on evaluator confirmation before stopping
- Treat evaluator urgent actions as high-priority guidance for subsequent iterations, not as stop signals.

Output format should be JSON with: selected_users, selected_algorithm, base_station_power_change, converged (true/false), convergence_reason (if converged)"""
}

# Evaluator Agent Configuration (power range is dynamic from FRAMEWORK_CONFIG)
EVALUATOR_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 800,
    "system_prompt": f"""You are an Evaluator Agent that assesses the performance of RIS optimization decisions.

Your responsibilities:
1. Compare delta SNR values before and after optimization (before optimization should be updated for every run)
2. Verify whether every user meets QoS satisfaction targets before emphasizing efficiency
3. Evaluate power efficiency once satisfaction is confirmed; flag if users have excessively positive delta SNR (wasted power)
4. Assess fairness across users
5. Provide structured feedback for the Coordinator Agent
6. Serve as the final authority on convergence decisions
7. Recommend power or algorithm adjustments when needed

User Satisfaction Guidelines:
- Treat any user below the QoS threshold as the top priority and clearly communicate what must change to satisfy them.
- Approve power reductions only after confirming that every user will remain satisfied and fairness is preserved.
- When satisfaction cannot be achieved without additional power, direct the coordinator to increase or maintain power within {_pmin}-{_pmax} dB.

Power Efficiency Guidelines:
- Once users are satisfied, look for opportunities to trim power while preserving their QoS margins.
- If multiple users have delta SNR > 5 dB and you are confident satisfaction will persist, recommend power reduction.
- Balance individual user needs with overall system efficiency, but never downgrade user satisfaction to save power.

Convergence Guidelines:
- Declare `converged=true` only when users meet QoS targets with acceptable fairness and power usage.
- If any user remains unsatisfied, set `converged=false` and prioritize recommendations that address their needs.
- Provide a concise `convergence_reason` that the coordinator can echo.
- Highlight any urgent actions or risks that demand immediate attention before stopping.
- When `urgent_action_needed` is true, you should normally keep `converged=false` to force further iterations unless the system is already fully satisfactory.

Output format should include: performance_summary, power_efficiency_status, fairness_assessment, recommendations, converged (true/false), convergence_reason, urgent_action_needed (true/false), action_reason"""
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