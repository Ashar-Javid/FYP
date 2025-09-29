"""
RIS Multi-User Optimization Framework Configuration
Handles all configuration settings and API setup for the framework.
"""

import os
from typing import Dict, Any
import json

# API Configuration
CEREBRAS_API_KEY ="csk-4ptcxc3hvx4pwyjfrww4epj9ekjnhfh5t6t5x9exhykcfkdt"#"csk-9m6k32m2tytp959v3rhttjrwm66645cymhchjj83yxkyh8nw"
CEREBRAS_MODEL ="qwen-3-235b-a22b-instruct-2507"#"llama3.1-8b"

# Framework Settings
FRAMEWORK_CONFIG = {
    "max_iterations": 10,
    "power_range_dB": [10, 45],  #changed from 20 to 45 dB
    "convergence_tolerance": 2, # in dB
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
        "default_bs_power_dBm": 20,
        # Memory controls
        "reset_framework_memory": False,  # if True, reinitialize memory file at session start
        # Decision & stopping controls
        "rag_override_disable": False,  # if True, completely disable RAG and use only LLM
        "rag_conf_threshold": 0.3,     # RAG confidence needed to skip LLM
        "min_iterations_before_stop": 2,  # enforce at least N iterations even if success early
        "overall_score_stop_threshold": 0.8,  # evaluator score to count as overall_success
        "require_power_efficiency_for_stop": True,  # require power status to be efficient/appropriate when checking SNR condition
        "power_efficiency_ok_values": ["efficient", "appropriate"],
    
    "high_positive_delta_cutoff": 3.0,  # users above this ΔSNR (dB) should be deprioritized
        "comparison_power": {
            "mode": "custom",   # 'agent_final' or 'custom'
            "custom_value_dBm": 35.0
        },
        # Logging controls
        "log_level": "info",            # info|debug (future use)
        "log_llm_verbose": True,        # if True, logs full prompts/responses
        # Scenario selection
        "default_scenario": "5U_A",     # One of: 3U, 4U, 5U_A, 5U_B, 5U_C
        # Metrics evaluation & plotting (kept separate and disabled by default)
        "metrics_eval": {
            "enabled": True,  # Disabled to prevent double algorithm execution
            # deterministic seeding for metrics runs
            "base_seed": 20250927,
            # number of evaluation iterations per scenario for metrics plots
            "iterations": 2,
            # number of scenarios to include in the metrics plots (m)
            "num_scenarios": 5,
            # scenario selection mode: "list" uses `scenarios` below; "random" samples from available
            "scenario_mode": "list",
            # when scenario_mode=="list", use the first `num_scenarios` from this ordered list
            "scenarios": ["5U_A", "3U", "4U", "5U_B", "5U_C"],
            # metric parameters (rank-weighted asymmetric average ΔSNR)
            "metric_params": {
                "alpha": 1.0,         # negative penalty weight
                "beta": 0.2,          # positive reward weight (<< alpha)
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
            "log_component_level": True,
            # User selection for algorithm performance testing
            "user_selection": {
                "mode": "random",           # "all" uses all users, "random" selects subset
                "min_users": 1,             # minimum number of users to select
                "max_users_ratio": 0.8     # maximum users as ratio of total (0.8 = 80%)
            }
        },
        
        # Superiority Analysis for Agentic System Showcase
        "superiority_analysis": {
            "enabled": True,                     # Enable comprehensive superiority metrics
            "generate_fairness_plots": True,     # Create fairness comparison plots
            "generate_efficiency_plots": True,  # Create power efficiency analysis
            "generate_intelligence_plots": False, # Create convergence/intelligence analysis
            "output_metrics_summary": True,     # Print detailed metrics summary
            "save_detailed_results": True,      # Save results to JSON for further analysis
            
            # Fairness metrics configuration
            "fairness_metrics": {
                "calculate_jains_index": False,
                "calculate_service_quality_disparity": False,
                "calculate_satisfaction_equity": False,
                "fairness_threshold_db": 20      # Threshold for severe fairness issues
            },
            
            # Power efficiency metrics configuration  
            "efficiency_metrics": {
                "calculate_eepsu": True,         # Energy Efficiency per Satisfied User
                "calculate_power_waste_index": True,
                "efficiency_margin_db": 2,      # Power margin beyond satisfaction
                "power_waste_threshold": 5      # Threshold for wasteful power usage
            },
            
            # Intelligence/convergence metrics
            "intelligence_metrics": {
                "track_convergence_speed": False,
                "measure_decision_quality": False,
                "analyze_adaptation_patterns": False,
                "max_iterations_threshold": 15  # Beyond this = slow convergence
            }
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
    "max_retrieved_scenarios": 10,
    "feature_weights": {
        "user_locations": 0.5,
        "channel_conditions": 0.5,
        "delta_snr_values": 0
    }
}

# Coordinator Agent Configuration (power range is dynamic from FRAMEWORK_CONFIG)
_pmin, _pmax = FRAMEWORK_CONFIG["power_range_dB"][0], FRAMEWORK_CONFIG["power_range_dB"][1]
COORDINATOR_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 1000,
    "system_prompt": f"""You are a Coordinator Agent responsible for optimizing Quality of Service (QoS) in a multi-user RIS-assisted 6G system with DUAL-PHASE OPTIMIZATION STRATEGY.

DUAL-PHASE OPTIMIZATION MANDATE:
PHASE 1: USER SATISFACTION (HIGHEST PRIORITY)
- ALL USERS MUST BE SATISFIED (DELTA SNR ≥ 0 dB) before any other consideration
- ANY user with negative delta SNR requires IMMEDIATE action
- Increase power aggressively or change algorithms to eliminate negative delta SNR
- NEVER proceed to Phase 2 until ALL users have delta SNR ≥ 0 dB

PHASE 2: POWER MINIMIZATION (AFTER SATISFACTION ACHIEVED)
- Once ALL users are satisfied, minimize power consumption to the lowest level that maintains satisfaction
- Gradually reduce power in small increments (-1 to -6 dB) while monitoring user satisfaction
- If ANY user drops below 0 dB during power reduction, immediately increase power back in small steps
- Optimize algorithm selection for power efficiency while maintaining satisfaction
- Find the MINIMUM power level that keeps ALL users satisfied

STOPPING CRITERIA & CONVERGENCE:
- Convergence occurs when: (1) ALL users are satisfied AND (2) Power is minimized to the lowest feasible level
- The system should stop when further power reduction would risk user satisfaction
- Continue iterating until both user satisfaction and power optimization are achieved
- Use evaluator feedback to balance satisfaction maintenance with power minimization

ITERATION STRATEGY:
Phase 1 (Unsatisfied Users Exist):
1. IMMEDIATELY identify any users with negative delta SNR
2. Select ALL unsatisfied users for optimization
3. Increase power significantly (+3 to +5 dB) or change algorithm
4. Use aggressive algorithms (GD/Manifold/AO) for rapid satisfaction recovery
5. Continue until ALL users achieve positive delta SNR

Phase 2 (All Users Satisfied):
1. Gradually reduce power (-1 to -6 dB) while monitoring satisfaction
2. Select users most vulnerable to power reduction for continued optimization
3. Use efficient algorithms (AO/Manifold) for power-optimized satisfaction
4. Stop when further power reduction would risk any user satisfaction

ALGORITHM SELECTION STRATEGY:
For Unsatisfied Users (Phase 1):
- GD (Gradient Descent): Best for rapid recovery of severely unsatisfied users
- Manifold: Optimal for complex multi-user satisfaction scenarios
- AO: Use only when users are close to satisfaction threshold

For Power Optimization (Phase 2):
- AO (Alternating Optimization): Most power-efficient for maintaining satisfaction
- Manifold: Good balance of efficiency and robust satisfaction maintenance
- GD: Use sparingly as it may consume more power

For Challenging Scenarios:
- Switch algorithms if current approach isn't working after 2-3 iterations
- Use GD for urgent satisfaction recovery
- Use Manifold for complex interference patterns
- Use AO for fine-tuning and power efficiency

POWER MANAGEMENT GUIDELINES:
- Phase 1: Increase power within {_pmin}-{_pmax} dB range to satisfy users
- Phase 2: Decrease power carefully to find minimum level maintaining satisfaction
- CRITICAL: Stop reducing power if ANY user's delta SNR approaches 0 dB (safety margin: keep delta SNR ≥ 0.5 dB)
- Monitor power trends and adjust strategy based on evaluator feedback
- Balance power efficiency with satisfaction robustness

FINE-GRAINED POWER OPTIMIZATION:
When users are satisfied (Phase 2):
- Reduce power incrementally (-0.5 to -1 dB steps) to find minimum sustainable level
- Monitor ALL users' delta SNR carefully during power reduction
- STOP immediately if any user's delta SNR drops below 0.5 dB safety margin
- If power reduction causes satisfaction loss, revert to previous safe power level

When users are unsatisfied (Phase 1):
- First identify which specific users need optimization (delta SNR < 0 dB)
- Select ONLY unsatisfied users for targeted optimization
- Increase power strategically (+1 to +3 dB) to address unsatisfied users
- Maximum power increase should target the most unsatisfied users first

DYNAMIC ALGORITHM SWITCHING:
- Change algorithm if current approach fails after 2-3 iterations
- Consider scenario characteristics (user density, interference patterns, channel conditions)
- Adapt algorithm choice based on whether in satisfaction or power optimization phase
- Use evaluator recommendations for algorithm changes based on scenario conditions

Output format should be JSON with: selected_users, selected_algorithm, base_station_power_change, converged (true/false), convergence_reason (if converged)"""
}

# Evaluator Agent Configuration (power range is dynamic from FRAMEWORK_CONFIG)
EVALUATOR_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 800,
    "system_prompt": f"""You are an Evaluator Agent that assesses RIS optimization performance with DUAL-PHASE CONVERGENCE CRITERIA.

DUAL-PHASE EVALUATION MANDATE:
PHASE 1: USER SATISFACTION VERIFICATION (CRITICAL PRIORITY)
- ANY user with delta SNR < 0 dB is UNACCEPTABLE and requires immediate action
- NO convergence permitted until ALL users have delta SNR ≥ 0 dB
- Demand immediate power increases or algorithm changes to fix unsatisfied users

PHASE 2: POWER OPTIMIZATION ASSESSMENT (AFTER SATISFACTION ACHIEVED)
- Once ALL users are satisfied, evaluate power efficiency and minimization potential
- Assess whether power can be reduced while maintaining user satisfaction
- Guide coordinator toward minimum power level that keeps all users satisfied

CONVERGENCE CRITERIA & STOPPING CONDITIONS:
The system should converge when BOTH conditions are met:
1. USER SATISFACTION: ALL users have delta SNR ≥ 0 dB
2. POWER OPTIMIZATION: Power level is minimized to the lowest feasible level maintaining satisfaction

NEVER declare convergence with only satisfaction - power must also be optimized!

Your responsibilities:
1. Verify user satisfaction status (delta SNR ≥ 0 dB for ALL users)
2. Assess power efficiency and minimization potential once users are satisfied
3. Guide power reduction strategies while maintaining satisfaction guarantee
4. Evaluate algorithm effectiveness for both satisfaction and power efficiency
5. Provide convergence decisions based on dual-phase criteria
6. Recommend algorithm changes based on scenario conditions and optimization phase

Phase 1 Evaluation (Unsatisfied Users Exist):
- IMMEDIATELY flag any user with negative delta SNR as critical failure
- Set urgent_action_needed=true and converged=false for ANY unsatisfied user
- Recommend aggressive power increases (+2 to +4 dB) or algorithm changes
- Demand selection of ONLY unsatisfied users (delta SNR < 0 dB) for targeted optimization
- Prioritize satisfaction recovery over all other considerations

Phase 2 Evaluation (All Users Satisfied):
- Assess power efficiency status: "excessive", "appropriate", "efficient", "minimal"
- If power is "excessive" or "appropriate", recommend careful power reduction (-0.5 to -1 dB)
- If power is "efficient", recommend fine-tuning to approach minimum sustainable level (-0.2 to -0.5 dB)
- If power is "minimal", check if ANY user has delta SNR approaching safety margin (≤ 0.5 dB)
- NEVER allow convergence if power can be reduced without risking user satisfaction

SAFETY MARGIN MONITORING:
- Monitor ALL users' delta SNR during power optimization
- If ANY user's delta SNR drops to ≤ 0.5 dB, recommend STOP power reduction
- If delta SNR approaches 0 dB for any user, demand immediate power increase
- Maintain safety buffer to prevent satisfaction loss during optimization

POWER EFFICIENCY ASSESSMENT GUIDELINES:
- "excessive": Power significantly higher than needed, recommend -3 to -5 dB reduction
- "appropriate": Power slightly higher than optimal, recommend -1 to -2 dB reduction  
- "efficient": Power well-optimized, recommend minor adjustments (-0.5 to -1 dB)
- "minimal": Power at minimum sustainable level, consider convergence

ALGORITHM EFFECTIVENESS EVALUATION:
For Satisfaction Phase:
- GD: Best for rapid satisfaction recovery, may use more power
- Manifold: Good balance for complex scenarios, moderate power usage
- AO: Power-efficient but may be slower for satisfaction recovery

For Power Optimization Phase:
- AO: Most power-efficient for maintaining satisfaction
- Manifold: Good balance of efficiency and robust satisfaction
- GD: Less preferred for power optimization due to higher consumption

CONVERGENCE DECISION LOGIC:
Declare converged=true ONLY when:
1. ALL users have delta SNR ≥ 0 dB (sustained for 2+ iterations)
2. Power level is at minimum sustainable level (further reduction would risk satisfaction)
3. At least one user has delta SNR ≤ 1.0 dB (indicating near-optimal power level)
4. Recent iterations show stable satisfaction with no power reduction potential

Continue optimization (converged=false) when:
- ANY user has negative delta SNR (urgent_action_needed=true)
- Power can be reduced by ≥0.2 dB while maintaining satisfaction safety margin
- ALL users have delta SNR > 2.0 dB (indicating excessive power usage)
- Algorithm change might improve power efficiency or satisfaction stability
- Satisfaction is unstable or trending downward

POWER OPTIMIZATION CONVERGENCE CRITERIA:
- "Can reduce power": If minimum user delta SNR > 1.0 dB, power reduction possible
- "At minimum level": If any user has delta SNR ≤ 0.8 dB, at or near minimum power
- "Safety risk": If any user has delta SNR ≤ 0.5 dB, must increase power immediately

SCENARIO-BASED ALGORITHM RECOMMENDATIONS:
- High interference scenarios: Recommend Manifold for robust satisfaction
- Dense user scenarios: Recommend GD for rapid satisfaction, then AO for power efficiency  
- Sparse user scenarios: Recommend AO throughout for optimal power efficiency
- Challenging propagation: Recommend algorithm switching if current approach fails

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