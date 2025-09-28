"""
RIS Framework Tools
Modular functions for scenario handling, algorithm execution, memory management, and visualization.
All tools are designed to be scalable and easy to debug.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import random

# Import from existing modules
from scenario import CASES, SIM_SETTINGS, generate_csi, calculate_snr
from multiuserdatasetgenerator import RISAlgorithms, RISDatasetGenerator
from config import FRAMEWORK_CONFIG, ensure_directories


class ScenarioTool:
    """
    Tool for handling scenario selection and preparation.
    Provides scenario data to the Coordinator Agent with visualization.
    """
    
    def __init__(self):
        self.available_scenarios = CASES
        # Keep SIM_SETTINGS in scenario synchronized with FRAMEWORK_CONFIG
        self.sim_settings = SIM_SETTINGS
        
    def get_scenario(self, name: str) -> Dict[str, Any]:
        """Prepare a scenario by name (e.g., '3U','4U','5U_A','5U_B','5U_C')."""
        if name not in self.available_scenarios:
            raise ValueError(f"Unknown scenario '{name}'. Available: {list(self.available_scenarios.keys())}")
        scenario = self.available_scenarios[name].copy()
        # Generate current CSI for all users
        for user in scenario["users"]:
            csi_data = generate_csi(user, self.sim_settings["bs_coord"], 
                                  self.sim_settings["ris_coord"], self.sim_settings)
            current_snr = calculate_snr(csi_data["h_eff"], 
                                      self.sim_settings["default_bs_power_dBm"],
                                      self.sim_settings["noise_power_dBm"])
            user.update({
                "current_csi": csi_data,
                "achieved_snr_dB": current_snr,
                "delta_snr_dB": current_snr - user["req_snr_dB"]
            })
        return {
            "scenario_name": name,
            "scenario_data": scenario,
            "sim_settings": self.sim_settings,
            "timestamp": datetime.now().isoformat()
        }

    def get_scenario_5ub(self) -> Dict[str, Any]:
        """
        Get the 5U_B scenario as specified.
        Returns complete scenario data with current channel conditions.
        """
        scenario = self.available_scenarios["5U_B"].copy()
        
        # Generate current CSI for all users
        for user in scenario["users"]:
            csi_data = generate_csi(user, self.sim_settings["bs_coord"], 
                                  self.sim_settings["ris_coord"], self.sim_settings)
            current_snr = calculate_snr(csi_data["h_eff"], 
                                      self.sim_settings["default_bs_power_dBm"],
                                      self.sim_settings["noise_power_dBm"])
            
            # Add current state to user data
            user.update({
                "current_csi": csi_data,
                "achieved_snr_dB": current_snr,
                "delta_snr_dB": current_snr - user["req_snr_dB"]
            })
            
        return {
            "scenario_name": "5U_B",
            "scenario_data": scenario,
            "sim_settings": self.sim_settings,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_random_scenario(self) -> Dict[str, Any]:
        """
        Alternative method for random scenario selection (kept commented for future use).
        """
        # scenario_name = random.choice(list(self.available_scenarios.keys()))
        # return self._prepare_scenario(scenario_name)
        pass
    
    def plot_scenario(self, scenario_data: Dict[str, Any], save_path: str = None) -> str:
        """
        Create visualization of the scenario showing BS, RIS, and users.
        Returns path to saved plot.
        """
        scenario = scenario_data["scenario_data"]
        sim_settings = scenario_data["sim_settings"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: System layout
        bs_pos = sim_settings["bs_coord"][:2]
        ris_pos = sim_settings["ris_coord"][:2]
        
        ax1.scatter(*bs_pos, marker='s', s=200, c='red', label='Base Station', zorder=5)
        ax1.scatter(*ris_pos, marker='^', s=200, c='blue', label='RIS', zorder=5)
        
        for user in scenario["users"]:
            user_pos = user["coord"][:2]
            delta_snr = user["delta_snr_dB"]
            
            # Color coding based on delta SNR
            if delta_snr >= 0:
                color = 'green'
                status = 'Satisfied'
            elif delta_snr >= -3:
                color = 'orange'
                status = 'Marginal'
            else:
                color = 'red'
                status = 'Poor'
                
            ax1.scatter(*user_pos, c=color, s=100, label=f'User {user["id"]} ({status})', zorder=4)
            
            # Add SNR information
            ax1.annotate(f'U{user["id"]}\nReq: {user["req_snr_dB"]:.1f} dB\nAch: {user["achieved_snr_dB"]:.1f} dB\nΔ: {delta_snr:.1f} dB',
                        xy=user_pos, xytext=(10, 10), textcoords='offset points',
                        fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
            
            # Draw channel lines
            if user["csi"]["los"] == "LoS":
                ax1.plot([bs_pos[0], user_pos[0]], [bs_pos[1], user_pos[1]], 'g-', alpha=0.6, linewidth=1.5)
            else:
                ax1.plot([bs_pos[0], user_pos[0]], [bs_pos[1], user_pos[1]], 'r--', alpha=0.6, linewidth=1.5)
            
            # RIS connection (always available)
            ax1.plot([ris_pos[0], user_pos[0]], [ris_pos[1], user_pos[1]], 'b:', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title(f'Scenario {scenario_data["scenario_name"]}: System Layout')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: SNR comparison
        user_ids = [f'U{u["id"]}' for u in scenario["users"]]
        req_snrs = [u["req_snr_dB"] for u in scenario["users"]]
        ach_snrs = [u["achieved_snr_dB"] for u in scenario["users"]]
        
        x = np.arange(len(user_ids))
        width = 0.35
        
        ax2.bar(x - width/2, req_snrs, width, label='Required SNR', alpha=0.8, color='skyblue')
        ax2.bar(x + width/2, ach_snrs, width, label='Achieved SNR', alpha=0.8, color='lightcoral')
        
        ax2.set_xlabel('Users')
        ax2.set_ylabel('SNR (dB)')
        ax2.set_title('SNR Requirements vs Achievement')
        ax2.set_xticks(x)
        ax2.set_xticklabels(user_ids)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            ensure_directories()
            save_path = os.path.join(FRAMEWORK_CONFIG["plots_dir"], 
                                   f'scenario_{scenario_data["scenario_name"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path


class AlgorithmTool:
    """
    Tool for executing RIS optimization algorithms.
    Interfaces with multiuserdatasetgenerator.py algorithms.
    """
    
    def __init__(self):
        self.ris_algorithms = RISAlgorithms()
        self.available_algorithms = {
            "analytical": "analytical_solution",  # Placeholder - implement if needed
            "GD": "gradient_descent_adam_multi",
            "manifold": "manifold_optimization_adam_multi", 
            "AO": "alternating_optimization_multi"
        }
    
    def execute_algorithm(self, algorithm_name: str, scenario_data: Dict[str, Any], 
                         selected_users: List[int], bs_power_dBm: float) -> Dict[str, Any]:
        """
        Execute the specified algorithm on selected users.
        Returns optimization results and performance metrics.
        """
        # Prepare channel data for all users once; subset for optimization
        h_d_all_list, H, h_1_all_list, user_id_order = self._prepare_channel_data_all(scenario_data)
        id_to_idx = {uid: i for i, uid in enumerate(user_id_order)}
        sel_idx = [id_to_idx[uid] for uid in selected_users if uid in id_to_idx]
        h_d_list = [h_d_all_list[i] for i in sel_idx]
        h_1_list = [h_1_all_list[i] for i in sel_idx]
        
        # Convert power to linear scale
        transmit_power = 10 ** ((bs_power_dBm - 30) / 10)  # dBm to Watts
        # Noise power (Watts) from scenario settings if available
        noise_power_W = 10 ** ((scenario_data["sim_settings"].get("noise_power_dBm", -104) - 30) / 10)
        
        # Execute algorithm - handle both short names and full names
        start_time = datetime.now()
        algorithm_key = algorithm_name.lower()
        
        # Map full algorithm names to short keys
        if "gradient_descent" in algorithm_key or algorithm_key == "gd":
            algorithm_key = "gd"
        elif "manifold" in algorithm_key:
            algorithm_key = "manifold"
        elif "alternating" in algorithm_key or algorithm_key == "ao":
            algorithm_key = "ao"
        
        if algorithm_key == "gd":
            theta_opt, rates_hist, snrs_hist, phases_hist = self.ris_algorithms.gradient_descent_adam_multi(
                h_d_list, H, h_1_list, len(H), transmit_power, max_iterations=500, noise_power=noise_power_W
            )
        elif algorithm_key == "manifold":
            theta_opt, rates_hist, snrs_hist, phases_hist = self.ris_algorithms.manifold_optimization_adam_multi(
                h_d_list, H, h_1_list, len(H), transmit_power, max_iterations=500, noise_power=noise_power_W
            )
        elif algorithm_key == "ao":
            theta_opt, rates_hist, snrs_hist, phases_hist = self.ris_algorithms.alternating_optimization_multi(
                h_d_list, H, h_1_list, len(H), transmit_power, max_iterations=200, noise_power=noise_power_W
            )
        else:
            raise ValueError(f"Algorithm {algorithm_name} not implemented. Available: GD, Manifold, AO")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate final SNRs for all users (not just selected ones) using optimized phases
        e = np.exp(1j * theta_opt)
        per_user_snrs_dB = self.ris_algorithms.compute_per_user_snrs(
            h_d_all_list, H, h_1_all_list, e, transmit_power,
            noise_power=noise_power_W
        )
        final_snrs = {uid: float(per_user_snrs_dB[i]) for i, uid in enumerate(user_id_order)}
        
        return {
            "algorithm": algorithm_name,
            "selected_users": selected_users,
            "optimal_phases": theta_opt,
            "convergence_history": {
                "rates": rates_hist,
                "snrs": snrs_hist
            },
            "final_snrs": final_snrs,
            "execution_time": execution_time,
            "converged": len(rates_hist) < 500  # Didn't hit max iterations
        }
    
    def _prepare_channel_data(self, scenario_data: Dict[str, Any], selected_users: List[int]) -> Tuple[List, np.ndarray, List]:
        """Prepare channel data for algorithm execution."""
        scenario = scenario_data["scenario_data"]
        selected_user_data = [u for u in scenario["users"] if u["id"] in selected_users]
        
        # Generate channel matrices (simplified for this implementation)
        K = len(selected_user_data)
        N = scenario_data["sim_settings"]["ris_elements"]
        
        h_d_list = []
        h_1_list = []
        
        for user_data in selected_user_data:
            # Direct channel (BS to user)
            h_d = user_data["current_csi"]["h_eff"]
            h_d_list.append(np.array([h_d]))
            
            # RIS to user channel
            h_1 = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
            h_1_list.append(h_1)
        
        # BS to RIS channel
        H = (np.random.randn(N, 1) + 1j*np.random.randn(N, 1)) / np.sqrt(2)
        
        return h_d_list, H, h_1_list

    def _prepare_channel_data_all(self, scenario_data: Dict[str, Any]) -> Tuple[List, np.ndarray, List, List[int]]:
        """Prepare channel data for all users (consistent H for all).
        Returns h_d_all_list, H, h_1_all_list, user_id_order
        """
        scenario = scenario_data["scenario_data"]
        users = scenario["users"]
        N = scenario_data["sim_settings"]["ris_elements"]

        h_d_all_list = []
        h_1_all_list = []

        # Direct channel proxy: use current effective direct component as baseline (best available in current model)
        for user_data in users:
            h_d = user_data["current_csi"]["h_eff"]
            h_d_all_list.append(np.array([h_d]))
            # RIS to user channel (Rayleigh)
            h_1 = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
            h_1_all_list.append(h_1)

        # BS to RIS channel shared across users
        H = (np.random.randn(N, 1) + 1j*np.random.randn(N, 1)) / np.sqrt(2)
        user_id_order = [u["id"] for u in users]
        return h_d_all_list, H, h_1_all_list, user_id_order
    
    def _calculate_all_user_snrs(self, scenario_data: Dict[str, Any], theta_opt: np.ndarray, bs_power_dBm: float) -> Dict[int, float]:
        """Calculate final SNRs for all users after optimization."""
        # Deprecated placeholder retained for compatibility; now computed inline in execute_algorithm
        final_snrs = {}
        for user in scenario_data["scenario_data"]["users"]:
            final_snrs[user["id"]] = float(user.get("achieved_snr_dB", 0.0))
        return final_snrs

    def run_algorithm_comparison(self, scenario_data: Dict[str, Any], bs_power_dBm: float) -> Dict[str, Dict[int, float]]:
        """Run all implemented algorithms plus a random baseline.
        Supports experimental multi-run mode (mean/std of Δ SNR) if enabled in FRAMEWORK_CONFIG.

        Return format (single-run mode / default):
            { algo_full_name: { user_id: final_snr }, ... , 'random_baseline': {...} }

        Return format (multi-run mode):
            {
              '_multi_run': True,
              'runs': <int>,
              'mean_delta': { short_label: {user_id: mean_delta_snr} },
              'std_delta':  { short_label: {user_id: std_delta_snr} },
              'raw_delta_runs': { short_label: [ {user_id: delta_snr}, ... per run ] },
              'required_snr': { user_id: req_snr },
              'labels_map': { original_algo_key: short_label, ... },
              'random_baseline_mean_delta': {user_id: mean_delta},
              'random_baseline_std_delta': {user_id: std_delta}
            }
        """
        multi_cfg = FRAMEWORK_CONFIG.get("algorithm_comparison_multi_run", {})
        multi_enabled = bool(multi_cfg.get("enabled", False))
        runs = int(multi_cfg.get("runs", 3)) if multi_enabled else 1

        # Scenario info
        users_struct = scenario_data["scenario_data"]["users"]
        user_ids = [u["id"] for u in users_struct]
        req_snr_map = {u["id"]: u.get("req_snr_dB", 0.0) for u in users_struct}

        algorithms = [
            ("gradient_descent_adam_multi", "GD"),
            ("manifold_optimization_adam_multi", "Manifold"),
            ("alternating_optimization_multi", "AO")
        ]

        if not multi_enabled:
            # Legacy single-run path (returns absolute final SNRs mapping)
            results = {}
            for algo_full, _short in algorithms:
                try:
                    res = self.execute_algorithm(algo_full, scenario_data, user_ids, bs_power_dBm)
                    results[algo_full] = res.get("final_snrs", {})
                except Exception:
                    results[algo_full] = {uid: np.nan for uid in user_ids}
            # Random baseline (absolute final SNR approximation)
            baseline = {}
            for u in users_struct:
                baseline[u["id"]] = u["achieved_snr_dB"] + np.random.uniform(0, 3)
            results["random_baseline"] = baseline
            return results

        # Multi-run statistical path (produce delta SNR stats)
        mean_delta = {}
        std_delta = {}
        raw_delta_runs = {}
        labels_map = {}

        for algo_full, short_lbl in algorithms:
            labels_map[algo_full] = short_lbl
            per_run_deltas = []
            for _ in range(runs):
                try:
                    res = self.execute_algorithm(algo_full, scenario_data, user_ids, bs_power_dBm)
                    final_snrs = res.get("final_snrs", {})
                except Exception:
                    final_snrs = {uid: np.nan for uid in user_ids}
                deltas = {uid: (final_snrs.get(uid, np.nan) - req_snr_map[uid]) for uid in user_ids}
                per_run_deltas.append(deltas)
            # Compute mean/std per user
            mean_delta[short_lbl] = {uid: float(np.nanmean([run[uid] for run in per_run_deltas])) for uid in user_ids}
            std_delta[short_lbl] = {uid: float(np.nanstd([run[uid] for run in per_run_deltas], ddof=0)) for uid in user_ids}
            raw_delta_runs[short_lbl] = per_run_deltas

        # Random baseline multi-run
        rb_runs = []
        for _ in range(runs):
            run = {}
            for u in users_struct:
                init = u["achieved_snr_dB"]
                final = init + np.random.uniform(0, 3)
                run[u["id"]] = final - req_snr_map[u["id"]]
            rb_runs.append(run)
        random_mean = {uid: float(np.nanmean([run[uid] for run in rb_runs])) for uid in user_ids}
        random_std = {uid: float(np.nanstd([run[uid] for run in rb_runs], ddof=0)) for uid in user_ids}

        return {
            '_multi_run': True,
            'runs': runs,
            'mean_delta': mean_delta,
            'std_delta': std_delta,
            'raw_delta_runs': raw_delta_runs,
            'required_snr': req_snr_map,
            'labels_map': labels_map,
            'random_baseline_mean_delta': random_mean,
            'random_baseline_std_delta': random_std
        }


class PowerControlTool:
    """
    Tool for adaptive base station power control.
    Operates within the specified range of 20-45 dB.
    """
    
    def __init__(self):
        self.min_power_dBm = FRAMEWORK_CONFIG["power_range_dB"][0]
        self.max_power_dBm = FRAMEWORK_CONFIG["power_range_dB"][1]
    
    def calculate_power_adjustment(self, current_power_dBm: float, scenario_data: Dict[str, Any], 
                                 target_improvement_dB: float = 5.0) -> Tuple[float, str]:
        """
        Calculate adaptive power adjustment based on scenario requirements.
        Returns new power level and reasoning.
        """
        users = scenario_data["scenario_data"]["users"]
        
        # Analyze current situation
        delta_snrs = [u["delta_snr_dB"] for u in users]
        avg_deficit = np.mean([d for d in delta_snrs if d < 0])
        max_deficit = min(delta_snrs)
        
        # Determine adjustment strategy
        if max_deficit < -10:  # Severe deficits
            adjustment = min(8.0, self.max_power_dBm - current_power_dBm)
            reason = f"Large SNR deficits detected (worst: {max_deficit:.1f} dB)"
        elif max_deficit < -5:  # Moderate deficits
            adjustment = min(5.0, self.max_power_dBm - current_power_dBm)
            reason = f"Moderate SNR deficits detected (worst: {max_deficit:.1f} dB)"
        elif avg_deficit < -2:  # Minor deficits
            adjustment = min(3.0, self.max_power_dBm - current_power_dBm)
            reason = f"Minor SNR deficits detected (avg: {avg_deficit:.1f} dB)"
        else:  # No significant deficits
            adjustment = 0.0
            reason = "No significant power adjustment needed"
        
        new_power = max(self.min_power_dBm, min(self.max_power_dBm, current_power_dBm + adjustment))
        
        return new_power, reason
    
    def suggest_power_reduction(self, current_power_dBm: float, excess_snr_data: List[float]) -> Tuple[float, str]:
        """
        Suggest power reduction when users have excessive delta SNR.
        """
        avg_excess = np.mean(excess_snr_data)
        
        if avg_excess > 8:
            reduction = min(5.0, current_power_dBm - self.min_power_dBm)
            reason = f"High power waste detected (avg excess: {avg_excess:.1f} dB)"
        elif avg_excess > 5:
            reduction = min(3.0, current_power_dBm - self.min_power_dBm)
            reason = f"Moderate power waste detected (avg excess: {avg_excess:.1f} dB)"
        else:
            reduction = 0.0
            reason = "Power levels appropriate"
        
        new_power = max(self.min_power_dBm, current_power_dBm - reduction)
        return new_power, reason


class MemoryTool:
    """
    Tool for managing framework memory and learning.
    Handles both persistent storage and runtime memory.
    """
    
    def __init__(self):
        from config import load_framework_memory, save_framework_memory
        self.load_memory = load_framework_memory
        self.save_memory = save_framework_memory
        self.runtime_memory = {
            "current_session": [],
            "active_patterns": []
        }
    
    def store_iteration_result(self, iteration_data: Dict[str, Any]):
        """Store results from a completed iteration."""
        # Load current memory
        memory = self.load_memory()
        
        # Add to iteration history
        memory["iteration_history"].append({
            "timestamp": datetime.now().isoformat(),
            "iteration_id": len(memory["iteration_history"]) + 1,
            **iteration_data
        })
        
        # Analyze for patterns
        self._extract_patterns(memory, iteration_data)
        
        # Save updated memory
        self.save_memory(memory)
        
        # Update runtime memory
        self.runtime_memory["current_session"].append(iteration_data)
    
    def _extract_patterns(self, memory: Dict[str, Any], iteration_data: Dict[str, Any]):
        """Extract learnable patterns from iteration results."""
        # Simple pattern extraction - can be enhanced
        if iteration_data.get("success", False):
            pattern = {
                "scenario_type": iteration_data.get("scenario_name"),
                "selected_algorithm": iteration_data.get("algorithm"),
                "user_selection_strategy": iteration_data.get("user_selection_reasoning"),
                "power_level": iteration_data.get("final_power_dBm"),
                "success_metrics": iteration_data.get("performance_metrics")
            }
            memory["successful_strategies"].append(pattern)
        else:
            failure_pattern = {
                "scenario_type": iteration_data.get("scenario_name"),
                "failed_algorithm": iteration_data.get("algorithm"),
                "failure_reason": iteration_data.get("failure_reason"),
                "avoid_strategy": iteration_data.get("strategy_details")
            }
            memory["failed_strategies"].append(failure_pattern)
    
    def get_relevant_patterns(self, scenario_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve patterns relevant to current scenario."""
        memory = self.load_memory()
        
        # Simple relevance matching - can be enhanced with embeddings
        scenario_name = scenario_data.get("scenario_name", "")
        num_users = len(scenario_data["scenario_data"]["users"])
        
        relevant_patterns = []
        
        # Look for successful strategies with similar characteristics
        for pattern in memory.get("successful_strategies", []):
            if (pattern.get("scenario_type") == scenario_name or 
                abs(len(pattern.get("user_selection_strategy", [])) - num_users) <= 1):
                relevant_patterns.append(pattern)
        
        return relevant_patterns


class VisualizationTool:
    """
    Tool for creating comprehensive visualizations of framework performance.
    """
    
    def __init__(self):
        ensure_directories()
        self.plots_dir = FRAMEWORK_CONFIG["plots_dir"]

    def plot_selected_scenario_scatter(self, scenario_data: Dict[str, Any], save_path: str = None) -> Optional[str]:
        """Compact scatter plot: BS, RIS, Users colored by Δ SNR status."""
        if not scenario_data:
            return None
        scenario = scenario_data["scenario_data"]
        sim_settings = scenario_data["sim_settings"]

        bs_pos = sim_settings["bs_coord"][:2]
        ris_pos = sim_settings["ris_coord"][:2]
        plt.figure(figsize=(8,6))
        plt.scatter(*bs_pos, marker='s', s=200, c='red', label='BS')
        plt.scatter(*ris_pos, marker='^', s=200, c='blue', label='RIS')

        for user in scenario["users"]:
            x, y = user["coord"][0], user["coord"][1]
            delta = user.get("delta_snr_dB", 0)
            color = 'green' if delta >= 0 else ('orange' if delta >= -3 else 'red')
            plt.scatter(x, y, c=color, s=90)
            plt.text(x+1.5, y+1.5, f"U{user['id']}\nΔ {delta:.1f} dB", fontsize=8, color=color)

        plt.title(f"Scenario {scenario_data.get('scenario_name','')} Setup")
        plt.xlabel('X (m)'); plt.ylabel('Y (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.plots_dir, f"scenario_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_iteration_progress(self, iteration_history: List[Dict[str, Any]], save_path: str = None) -> str:
        """Deprecated: legacy progress plot removed."""
        return None

    def plot_power_and_snr_evolution(self, iteration_history: List[Dict[str, Any]], save_path: str = None) -> Optional[str]:
        """Plot evolution of BS power and average final delta SNR per iteration."""
        if not iteration_history:
            return None
        iterations = list(range(1, len(iteration_history)+1))
        powers = [it.get("current_power_dBm", 0) for it in iteration_history]
        avg_deltas = [it.get("avg_final_delta_snr", 0) for it in iteration_history]

        fig, ax1 = plt.subplots(figsize=(10,6))
        color1 = '#1f77b4'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('BS Power (dBm)', color=color1)
        ax1.plot(iterations, powers, '-o', color=color1, label='Power (dBm)')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.25)

        ax2 = ax1.twinx()
        color2 = '#d62728'
        ax2.set_ylabel('Avg Final Δ SNR (dB)', color=color2)
        ax2.plot(iterations, avg_deltas, '-s', color=color2, label='Avg Final Δ SNR')
        ax2.tick_params(axis='y', labelcolor=color2)

        fig.suptitle('Power and Average Final Δ SNR Evolution')
        fig.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.plots_dir, f'power_snr_evolution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_power_efficiency(self, iteration_history: List[Dict[str, Any]], save_path: str = None) -> Optional[str]:
        """Plot power waste score and users satisfied over iterations to reflect efficiency."""
        if not iteration_history:
            return None
        iterations = list(range(1, len(iteration_history)+1))
        waste_scores = [it.get('evaluation', {}).get('power_waste_score', 0) for it in iteration_history]
        satisfied = [it.get('evaluation', {}).get('users_satisfied_after', 0) for it in iteration_history]

        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Power Waste Score', color='#9467bd')
        ax1.plot(iterations, waste_scores, '-o', color='#9467bd', label='Power Waste Score')
        ax1.tick_params(axis='y', labelcolor='#9467bd')
        ax1.grid(True, alpha=0.25)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Users Satisfied', color='#2ca02c')
        ax2.plot(iterations, satisfied, '-^', color='#2ca02c', label='Users Satisfied')
        ax2.tick_params(axis='y', labelcolor='#2ca02c')

        fig.suptitle('Power Efficiency & User Satisfaction')
        fig.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.plots_dir, f'power_efficiency_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def generate_final_report_plot(self, final_results: Dict[str, Any], save_path: str = None) -> str:
        """Generate comprehensive final report visualization."""
        # Implementation for final report plotting
        # This would create a detailed summary of the entire optimization session
        pass

    def plot_final_snr_comparison(self, scenario_data: Dict[str, Any], iteration_history: List[Dict[str, Any]], save_path: str = None) -> Optional[str]:
        """
        Plot per-user SNR comparison showing:
          - Required SNR
          - Initial (pre-optimization) achieved SNR
          - Final (post-optimization) SNR from last iteration's algorithm_results.final_snrs

        Returns path or None if insufficient data.
        """
        if not scenario_data or not iteration_history:
            return None

        last_iteration = iteration_history[-1]
        algorithm_results = last_iteration.get("algorithm_results", {})
        final_snrs = algorithm_results.get("final_snrs")
        if not final_snrs:
            return None

        try:
            users = scenario_data["scenario_data"]["users"]
        except Exception:
            return None

        user_ids = [u["id"] for u in users]
        required_snrs = [u.get("req_snr_dB", 0) for u in users]
        initial_snrs = [u.get("achieved_snr_dB", 0) for u in users]
        final_snrs_ordered = [final_snrs.get(uid, initial_snrs[i]) for i, uid in enumerate(user_ids)]

        x = np.arange(len(user_ids))
        width = 0.25

        plt.figure(figsize=(11, 6))
        plt.bar(x - width, required_snrs, width, label='Required', color='#4C72B0', alpha=0.8)
        plt.bar(x, initial_snrs, width, label='Initial', color='#55A868', alpha=0.8)
        plt.bar(x + width, final_snrs_ordered, width, label='Final', color='#C44E52', alpha=0.9)

        # Annotate improvements
        for i, (init, final) in enumerate(zip(initial_snrs, final_snrs_ordered)):
            imp_final = final - init
            plt.text(x[i] + width, final + 0.3, f"+{imp_final:.1f}dB", ha='center', va='bottom', fontsize=8)

        plt.xticks(x, [f'U{uid}' for uid in user_ids])
        plt.ylabel('SNR (dB)')
        plt.xlabel('Users')
        plt.title('Per-User SNR Comparison (Required vs Initial vs Final)')
        plt.grid(axis='y', alpha=0.25)
        plt.legend()
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.plots_dir, f'final_snr_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_algorithm_comparison(self, scenario_data: Dict[str, Any], comparison_results: Dict[str, Any], save_path: str = None, agent_final_snrs: Optional[Dict[int, float]] = None) -> Optional[str]:
        """Plot per-user Δ SNR (final - required) across algorithms with compact labels.

        Modes:
            1. Single-run (legacy): comparison_results maps algorithm -> absolute final SNRs.
               We convert to Δ SNR internally using scenario required SNRs.
            2. Multi-run (experimental): comparison_results contains '_multi_run': True with
               mean/std delta maps already computed for GD, Manifold, AO, Random.

        Agent final results are added as a single-run delta (no error bars).
        """
        if not comparison_results:
            return None

        users = scenario_data["scenario_data"]["users"]
        user_ids = [u["id"] for u in users]
        req_snrs = {u["id"]: u.get('req_snr_dB', 0.0) for u in users}

        multi_run_mode = isinstance(comparison_results, dict) and comparison_results.get('_multi_run')

        # Prepare data structures
        algo_order: List[str] = []
        mean_delta_map: Dict[str, Dict[int, float]] = {}
        std_delta_map: Dict[str, Dict[int, float]] = {}

        if multi_run_mode:
            mean_delta = comparison_results.get('mean_delta', {})
            std_delta = comparison_results.get('std_delta', {})
            labels_map = comparison_results.get('labels_map', {})
            # Preserve order GD, Manifold, AO
            for orig, short in labels_map.items():
                algo_order.append(short)
                mean_delta_map[short] = mean_delta.get(short, {})
                std_delta_map[short] = std_delta.get(short, {})
            # Random baseline
            algo_order.append('Random')
            mean_delta_map['Random'] = comparison_results.get('random_baseline_mean_delta', {})
            std_delta_map['Random'] = comparison_results.get('random_baseline_std_delta', {uid: 0.0 for uid in user_ids})
        else:
            # Legacy path: convert absolute final SNRs to delta
            for algo_full, per_user_abs in comparison_results.items():
                if algo_full == 'random_baseline':
                    short = 'Random'
                else:
                    if 'gradient_descent' in algo_full:
                        short = 'GD'
                    elif 'manifold' in algo_full:
                        short = 'Manifold'
                    elif 'alternating' in algo_full:
                        short = 'AO'
                    else:
                        short = algo_full[:8]
                algo_order.append(short)
                deltas = {uid: per_user_abs.get(uid, np.nan) - req_snrs[uid] for uid in user_ids}
                mean_delta_map[short] = deltas
                std_delta_map[short] = {uid: 0.0 for uid in user_ids}

        # Add agent final (single run delta)
        agent_label = 'Agent'
        agent_delta: Dict[int, float] = {}
        if agent_final_snrs:
            for uid in user_ids:
                agent_delta[uid] = agent_final_snrs.get(uid, np.nan) - req_snrs[uid]
            algo_order.append(agent_label)
            mean_delta_map[agent_label] = agent_delta
            std_delta_map[agent_label] = {uid: 0.0 for uid in user_ids}

        num_algos = len(algo_order)
        x = np.arange(len(user_ids))
        width = 0.12 if num_algos > 6 else 0.15

        # Try to use advanced visualization first
        try:
            from advanced_visualization import create_enhanced_algorithm_comparison
            save_path = create_enhanced_algorithm_comparison(
                user_ids, mean_delta_map, std_delta_map, algo_order, 
                self.plots_dir, multi_run_mode
            )
        except ImportError:
            # Fallback to original plotting
            plt.figure(figsize=(max(10, 2*len(user_ids)), 6))
            for i, algo in enumerate(algo_order):
                means = [mean_delta_map[algo].get(uid, np.nan) for uid in user_ids]
                stds = [std_delta_map[algo].get(uid, 0.0) for uid in user_ids]
                positions = x + (i - num_algos/2)*width + width/2
                if multi_run_mode and algo != 'Agent' and any(s > 1e-6 for s in stds):
                    plt.bar(positions, means, width, label=algo, alpha=0.85, yerr=stds, capsize=3)
                else:
                    plt.bar(positions, means, width, label=algo, alpha=0.85)

            plt.axhline(0, color='k', linewidth=1.0, linestyle='--', alpha=0.7)
            plt.xticks(x, [f'U{uid}' for uid in user_ids])
            plt.ylabel('Δ SNR (dB)')
            plt.xlabel('Users')
            title_suffix = '' if not multi_run_mode else f" (Mean ± Std over {comparison_results.get('runs', 1)} runs)"
            plt.title(f'Per-User Δ SNR: Algorithms vs Agent{title_suffix}')
            plt.grid(axis='y', alpha=0.25)
            plt.legend(ncol=3 if num_algos > 6 else 2, fontsize=8)
            plt.tight_layout()
            if save_path is None:
                save_path = os.path.join(self.plots_dir, f'algorithm_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return save_path


# Export all tools
__all__ = [
    'ScenarioTool', 'AlgorithmTool', 'PowerControlTool', 
    'MemoryTool', 'VisualizationTool'
]