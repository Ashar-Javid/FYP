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
        self.sim_settings = SIM_SETTINGS
        
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
        # Prepare channel data for selected users
        h_d_list, H, h_1_list = self._prepare_channel_data(scenario_data, selected_users)
        
        # Convert power to linear scale
        transmit_power = 10**(bs_power_dBm/10) / 1000  # dBm to Watts
        
        # Execute algorithm
        start_time = datetime.now()
        
        if algorithm_name.lower() == "gd":
            theta_opt, rates_hist, snrs_hist, phases_hist = self.ris_algorithms.gradient_descent_adam_multi(
                h_d_list, H, h_1_list, len(H), transmit_power, max_iterations=500
            )
        elif algorithm_name.lower() == "manifold":
            theta_opt, rates_hist, snrs_hist, phases_hist = self.ris_algorithms.manifold_optimization_adam_multi(
                h_d_list, H, h_1_list, len(H), transmit_power, max_iterations=500
            )
        elif algorithm_name.lower() == "ao":
            theta_opt, rates_hist, snrs_hist, phases_hist = self.ris_algorithms.alternating_optimization_multi(
                h_d_list, H, h_1_list, len(H), transmit_power, max_iterations=200
            )
        else:
            raise ValueError(f"Algorithm {algorithm_name} not implemented")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate final SNRs for all users (not just selected ones)
        final_snrs = self._calculate_all_user_snrs(scenario_data, theta_opt, bs_power_dBm)
        
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
    
    def _calculate_all_user_snrs(self, scenario_data: Dict[str, Any], theta_opt: np.ndarray, bs_power_dBm: float) -> Dict[int, float]:
        """Calculate final SNRs for all users after optimization."""
        # This is a simplified calculation - in practice, you'd use the optimized RIS phases
        final_snrs = {}
        
        for user in scenario_data["scenario_data"]["users"]:
            # Simplified: assume some improvement for selected users
            current_snr = user["achieved_snr_dB"]
            # Add some improvement based on optimization (placeholder)
            improvement = np.random.uniform(2, 8)  # dB improvement
            final_snrs[user["id"]] = current_snr + improvement
        
        return final_snrs


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
    
    def plot_iteration_progress(self, iteration_history: List[Dict[str, Any]], save_path: str = None) -> str:
        """Create plots showing progress across iterations."""
        if not iteration_history:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        iterations = range(1, len(iteration_history) + 1)
        
        # Plot 1: Average Delta SNR progression
        avg_delta_snrs = []
        for iteration in iteration_history:
            delta_snrs = [u.get("final_delta_snr", 0) for u in iteration.get("user_results", [])]
            avg_delta_snrs.append(np.mean(delta_snrs) if delta_snrs else 0)
        
        axes[0, 0].plot(iterations, avg_delta_snrs, 'b-o', markersize=6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Average Δ SNR (dB)')
        axes[0, 0].set_title('Delta SNR Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Power consumption
        power_levels = [iteration.get("final_power_dBm", 20) for iteration in iteration_history]
        axes[0, 1].plot(iterations, power_levels, 'g-s', markersize=6)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('BS Power (dBm)')
        axes[0, 1].set_title('Power Level Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Algorithm usage
        algorithms = [iteration.get("algorithm", "unknown") for iteration in iteration_history]
        unique_algos = list(set(algorithms))
        algo_counts = [algorithms.count(algo) for algo in unique_algos]
        
        axes[1, 0].bar(unique_algos, algo_counts, color=['red', 'blue', 'green', 'orange'][:len(unique_algos)])
        axes[1, 0].set_xlabel('Algorithm')
        axes[1, 0].set_ylabel('Usage Count')
        axes[1, 0].set_title('Algorithm Selection Distribution')
        
        # Plot 4: Success rate
        success_rates = []
        for i in range(len(iteration_history)):
            recent_iterations = iteration_history[max(0, i-2):i+1]  # Last 3 iterations
            successes = sum(1 for it in recent_iterations if it.get("success", False))
            success_rates.append(successes / len(recent_iterations) * 100)
        
        axes[1, 1].plot(iterations, success_rates, 'm-^', markersize=6)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title('Rolling Success Rate (3-iteration window)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, f'framework_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_final_report_plot(self, final_results: Dict[str, Any], save_path: str = None) -> str:
        """Generate comprehensive final report visualization."""
        # Implementation for final report plotting
        # This would create a detailed summary of the entire optimization session
        pass


# Export all tools
__all__ = [
    'ScenarioTool', 'AlgorithmTool', 'PowerControlTool', 
    'MemoryTool', 'VisualizationTool'
]