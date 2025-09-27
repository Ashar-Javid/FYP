import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import copy
from typing import Dict, List, Tuple

# Import from multiuserdatasetgenerator
sys.path.append('.')
from multiuserdatasetgenerator import RISAlgorithms, SIM_SETTINGS
from config import FRAMEWORK_CONFIG

class RISDatasetGenerator:
    """Simplified version focused on scenario generation for concept analysis"""
    
    def __init__(self):
        self.sim_settings = SIM_SETTINGS
        self.applications = FRAMEWORK_CONFIG.get("snr_calculation", {}).get("applications", {})
        
    def generate_random_case(self) -> Dict:
        """Generate a random case with varying users and channel conditions"""
        # Random number of users (3-6)
        num_users = np.random.randint(3, 7)
        
        # Random user placement within a reasonable range
        max_distance = 200
        users = []
        
        for user_id in range(1, num_users + 1):
            # Random position
            x = np.random.uniform(-max_distance, max_distance)
            y = np.random.uniform(-max_distance, max_distance)
            z = 1.5  # Standard user height
            
            # Random application
            app_name = np.random.choice(list(self.applications.keys()))
            
            # Random channel conditions
            los = np.random.choice([True, False])
            if los:
                fading = "Rician"
                k_factor = np.random.uniform(5, 15)  # K-factor for LoS
                csi = {"los": los, "fading": fading, "K_factor_dB": k_factor}
            else:
                fading = np.random.choice(["Rician", "Rayleigh"])
                if fading == "Rician":
                    k_factor = np.random.uniform(0, 8)  # Lower K-factor for NLoS
                    csi = {"los": los, "fading": fading, "K_factor_dB": k_factor}
                else:
                    csi = {"los": los, "fading": fading}
            
            # Calculate required SNR based on application and conditions
            base_snr = self.applications[app_name]["base_snr_dB"]
            # Add some variation based on channel conditions
            if not los:
                base_snr += np.random.uniform(2, 5)  # Higher requirement for NLoS
            
            user = {
                "id": user_id,
                "coord": [x, y, z],
                "app": app_name,
                "csi": csi,
                "required_snr_dB": base_snr
            }
            users.append(user)
        
        return {
            "num_users": num_users,
            "users": users
        }
    
    def calculate_received_snr(self, user: Dict, bs_power_linear: float, noise_power_linear: float) -> Tuple[float, complex, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate received SNR and channel components for a user"""
        # Distance calculations
        bs_coord = self.sim_settings["bs_coord"]
        ris_coord = self.sim_settings["ris_coord"]
        user_coord = user["coord"]
        
        # BS to user distance
        d_bs = np.sqrt(sum((a - b)**2 for a, b in zip(user_coord, bs_coord)))
        
        # RIS to user distance  
        d_ris = np.sqrt(sum((a - b)**2 for a, b in zip(user_coord, ris_coord)))
        
        # BS to RIS distance
        d_bs_ris = np.sqrt(sum((a - b)**2 for a, b in zip(bs_coord, ris_coord)))
        
        # Pathloss calculation
        def pathloss_db(distance):
            return self.sim_settings["PL0_dB"] + 10 * self.sim_settings["gamma"] * np.log10(distance)
        
        pl_bs = pathloss_db(d_bs)
        pl_ris = pathloss_db(d_ris) 
        pl_bs_ris = pathloss_db(d_bs_ris)
        
        # Channel generation based on CSI conditions
        csi = user["csi"]
        
        # Direct channel (BS to user)
        if csi["fading"] == "Rician" and "K_factor_dB" in csi:
            K_linear = 10**(csi["K_factor_dB"]/10)
            h_los = np.exp(1j * 2 * np.pi * np.random.rand())
            h_nlos = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
            h_d = np.sqrt(K_linear/(K_linear+1))*h_los + np.sqrt(1/(K_linear+1))*h_nlos
        else:  # Rayleigh
            h_d = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
        
        # Apply pathloss
        h_d *= 10**(-pl_bs/20)
        
        # RIS channels
        N = self.sim_settings["ris_elements"]
        
        # BS to RIS channel (H)
        H = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
        H *= 10**(-pl_bs_ris/20)
        
        # RIS to user channel (h_1)
        if csi["fading"] == "Rician" and "K_factor_dB" in csi:
            K_linear = 10**(csi["K_factor_dB"]/10)
            h_los_ris = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
            h_nlos_ris = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
            h_1 = np.sqrt(K_linear/(K_linear+1))*h_los_ris + np.sqrt(1/(K_linear+1))*h_nlos_ris
        else:  # Rayleigh
            h_1 = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
        
        h_1 *= 10**(-pl_ris/20)
        
        # Calculate received SNR with random RIS phases (for baseline)
        theta_random = np.random.uniform(0, 2*np.pi, N)
        e_random = np.exp(1j*theta_random)
        
        # Effective channel
        h_eff = h_d + np.dot(h_1, e_random * H)
        
        # Received power and SNR
        received_power = abs(h_eff)**2 * bs_power_linear
        snr_linear = received_power / noise_power_linear
        snr_db = 10 * np.log10(snr_linear + 1e-12)
        
        return snr_db, h_d, H, h_1, theta_random
    
    def select_users_for_optimization(self, case_data: Dict) -> Tuple[List[int], Dict, Dict]:
        """Select users that need optimization based on their SNR deficit"""
        bs_power_linear = 10**((self.sim_settings["default_bs_power_dBm"] - 30) / 10)
        noise_power_linear = 10**((self.sim_settings["noise_power_dBm"] - 30) / 10)
        
        users_needing_optimization = []
        user_channels = {}
        
        for user in case_data["users"]:
            # Calculate current SNR
            snr_db, h_d, H, h_1, theta_random = self.calculate_received_snr(user, bs_power_linear, noise_power_linear)
            
            # Check if user needs optimization
            required_snr = user["required_snr_dB"]
            if snr_db < required_snr:
                users_needing_optimization.append(user["id"])
            
            user_channels[user["id"]] = {
                "h_d": h_d,
                "H": H, 
                "h_1": h_1,
                "required_snr_dB": required_snr,
                "current_snr_dB": snr_db,
                "random_phases": theta_random  # Store the random phases used
            }
        
        return users_needing_optimization, user_channels, {}
    
    def run_algorithm_comparison(self, user_channels: Dict, selected_users: List[int], 
                               bs_power_linear: float, noise_power_linear: float) -> str:
        """Compare algorithms and return the best one"""
        if not selected_users:
            return "no_optimization_needed"
        
        algorithms = RISAlgorithms()
        algorithm_names = [
            'gradient_descent_adam_multi',
            'manifold_optimization_adam_multi', 
            'alternating_optimization_multi'
        ]
        
        # Prepare channel data for selected users only
        h_d_list = [user_channels[uid]["h_d"] for uid in selected_users]
        h_1_list = [user_channels[uid]["h_1"] for uid in selected_users]
        H = user_channels[selected_users[0]]["H"]  # Same for all users
        
        best_algorithm = None
        best_sum_rate = -np.inf
        
        for algo_name in algorithm_names:
            try:
                algo_method = getattr(algorithms, algo_name)
                theta, sum_rates, per_user_snrs, iterations = algo_method(
                    h_d_list, H, h_1_list,
                    self.sim_settings["ris_elements"],
                    bs_power_linear,
                    max_iterations=1000,  # Increased iterations
                    noise_power=noise_power_linear
                )
                
                final_sum_rate = sum_rates[-1] if sum_rates else -np.inf
                
                if final_sum_rate > best_sum_rate:
                    best_sum_rate = final_sum_rate
                    best_algorithm = algo_name
                    
            except Exception as e:
                print(f"Algorithm {algo_name} failed: {e}")
                continue
        
        return best_algorithm or algorithm_names[0]  # Fallback to first algorithm

class ConceptAnalyzer:
    def __init__(self):
        self.generator = RISDatasetGenerator()
        self.algorithms = RISAlgorithms()
        self.algorithm_names = ['gradient_descent_adam_multi', 'manifold_optimization_adam_multi', 'alternating_optimization_multi']
        
        # Create plots/Concept directory
        self.plots_dir = "plots/Concept"
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def generate_scenarios(self, num_scenarios: int = 10) -> List[Dict]:
        """Generate random scenarios using multiuserdatasetgenerator"""
        print(f"Generating {num_scenarios} scenarios...")
        scenarios = []
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(num_scenarios):
            case_data = self.generator.generate_random_case()
            scenarios.append(case_data)
            
        return scenarios
    
    def calculate_all_users_delta_snr(self, case_data: Dict, algorithm_name: str = None, 
                                    selected_users: List[int] = None) -> float:
        """Calculate average delta SNR for ALL users in scenario"""
        bs_power_linear = 10**((self.generator.sim_settings["default_bs_power_dBm"] - 30) / 10)
        noise_power_linear = 10**((self.generator.sim_settings["noise_power_dBm"] - 30) / 10)
        
        # Get all user channels and required SNRs
        all_user_channels = {}
        all_required_snrs = []
        all_random_phases = []  # Store random phases for each user
        
        for user in case_data["users"]:
            required_snr_dB = user["required_snr_dB"]
            all_required_snrs.append(required_snr_dB)
            
            # Calculate channels
            received_snr_dB, h_d, H, h_1, theta_random = self.generator.calculate_received_snr(
                user, bs_power_linear, noise_power_linear
            )
            
            all_user_channels[user["id"]] = {
                "h_d": h_d, "H": H, "h_1": h_1,
                "required_snr_dB": required_snr_dB
            }
            all_random_phases.append(theta_random)
        
        # If no algorithm specified, use the SAME random phases that were used during channel calculation
        if algorithm_name is None:
            achieved_snrs = []
            for i, user_id in enumerate(sorted(all_user_channels.keys())):
                channels = all_user_channels[user_id]
                # Use the same random phases that were used for this user's channel calculation
                e = np.exp(1j * all_random_phases[i])
                h_r = self.algorithms.compute_cascaded_channel(channels["H"], channels["h_1"])
                power = self.algorithms.compute_received_power(
                    channels["h_d"], h_r, e, bs_power_linear
                )
                snr_linear = power / noise_power_linear
                snr_dB = 10 * np.log10(snr_linear + 1e-12)
                achieved_snrs.append(snr_dB)
        else:
            # Use specified algorithm with ALL users for RIS optimization
            h_d_list = [all_user_channels[uid]["h_d"] for uid in sorted(all_user_channels.keys())]
            H_list = [all_user_channels[uid]["H"] for uid in sorted(all_user_channels.keys())]
            h_1_list = [all_user_channels[uid]["h_1"] for uid in sorted(all_user_channels.keys())]
            
            try:
                algo_method = getattr(self.algorithms, algorithm_name)
                theta, sum_rates, per_user_snrs, iterations = algo_method(
                    h_d_list, H_list[0], h_1_list, 
                    self.generator.sim_settings["ris_elements"],
                    bs_power_linear, 
                    max_iterations=1000,
                    noise_power=noise_power_linear
                )
                achieved_snrs = per_user_snrs[-1] if per_user_snrs else [0.0] * len(h_d_list)
            except Exception as e:
                print(f"Algorithm {algorithm_name} failed: {e}")
                achieved_snrs = [0.0] * len(all_user_channels)
        
        # Calculate delta SNRs and return average
        delta_snrs = [achieved - required for achieved, required in zip(achieved_snrs, all_required_snrs)]
        return np.mean(delta_snrs)
    
    def run_exhaustive_search(self, case_data: Dict) -> Tuple[str, List[int], float]:
        """Run exhaustive search to find best algorithm and user selection"""
        # First, get users that need optimization
        selected_users, user_channels, _ = self.generator.select_users_for_optimization(case_data)
        
        if not selected_users:
            return "no_optimization_needed", [], self.calculate_all_users_delta_snr(case_data)
        
        # Find best algorithm using existing method
        bs_power_linear = 10**((self.generator.sim_settings["default_bs_power_dBm"] - 30) / 10)
        noise_power_linear = 10**((self.generator.sim_settings["noise_power_dBm"] - 30) / 10)
        
        best_algorithm = self.generator.run_algorithm_comparison(
            user_channels, selected_users, bs_power_linear, noise_power_linear
        )
        
        # Calculate delta SNR for all users with best selection
        agent_delta_snr = self.calculate_all_users_delta_snr(
            case_data, best_algorithm, selected_users
        )
        
        return best_algorithm, selected_users, agent_delta_snr
    
    def analyze_scenarios(self, scenarios: List[Dict]) -> Dict:
        """Analyze all scenarios with different algorithms"""
        results = {
            'scenarios': [],
            'all_users_random': [],
            'all_users_gd': [],
            'all_users_manifold': [],
            'all_users_ao': [],
            'agent_results': []
        }
        
        for i, scenario in enumerate(scenarios):
            print(f"Analyzing scenario {i+1}/{len(scenarios)}...")
            
            # All users with random phases
            random_delta = self.calculate_all_users_delta_snr(scenario)
            
            # All users with each algorithm
            gd_delta = self.calculate_all_users_delta_snr(scenario, 'gradient_descent_adam_multi')
            manifold_delta = self.calculate_all_users_delta_snr(scenario, 'manifold_optimization_adam_multi')
            ao_delta = self.calculate_all_users_delta_snr(scenario, 'alternating_optimization_multi')
            
            # Agent (exhaustive search)
            best_algo, selected_users, agent_delta = self.run_exhaustive_search(scenario)
            
            results['scenarios'].append(f"Scenario {i+1}")
            results['all_users_random'].append(random_delta)
            results['all_users_gd'].append(gd_delta)
            results['all_users_manifold'].append(manifold_delta)
            results['all_users_ao'].append(ao_delta)
            results['agent_results'].append(agent_delta)
            
            # Store detailed scenario info
            scenario['analysis'] = {
                'best_algorithm': best_algo,
                'selected_users': selected_users,
                'delta_snrs': {
                    'random': random_delta,
                    'gd': gd_delta,
                    'manifold': manifold_delta,
                    'ao': ao_delta,
                    'agent': agent_delta
                }
            }
        
        return results
    
    def plot_scenario_layouts(self, scenarios: List[Dict]):
        """Plot all scenario layouts"""
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, scenario in enumerate(scenarios):
            ax = axes[i]
            
            # Plot BS and RIS
            bs_coord = self.generator.sim_settings["bs_coord"]
            ris_coord = self.generator.sim_settings["ris_coord"]
            
            ax.scatter(bs_coord[0], bs_coord[1], marker="^", color="red", s=100, label="BS")
            ax.scatter(ris_coord[0], ris_coord[1], marker="s", color="blue", s=100, label="RIS")
            
            # Plot users
            for user in scenario["users"]:
                x, y, _ = user["coord"]
                ax.scatter(x, y, marker="o", color="green", s=60)
                ax.text(x+5, y+5, f"U{user['id']}", fontsize=8)
            
            ax.set_title(f"Scenario {i+1} ({scenario['num_users']} users)")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-220, 220)
            ax.set_ylim(-220, 220)
            
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/scenario_layouts.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Scenario layouts saved to {self.plots_dir}/scenario_layouts.png")
    
    def plot_grouped_bar_charts(self, results: Dict):
        """Create grouped bar charts split into two plots"""
        # Split into two plots with 5 scenarios each
        n_scenarios = len(results['scenarios'])
        mid_point = n_scenarios // 2
        
        # Plot 1: First 5 scenarios
        self._create_bar_chart(results, 0, mid_point, "scenarios_1_5")
        
        # Plot 2: Last 5 scenarios  
        self._create_bar_chart(results, mid_point, n_scenarios, "scenarios_6_10")
    
    def _create_bar_chart(self, results: Dict, start_idx: int, end_idx: int, filename: str):
        """Create a single grouped bar chart"""
        scenarios = results['scenarios'][start_idx:end_idx]
        
        data = {
            'All Users + Random': results['all_users_random'][start_idx:end_idx],
            'All Users + GD': results['all_users_gd'][start_idx:end_idx],
            'All Users + Manifold': results['all_users_manifold'][start_idx:end_idx],
            'All Users + AO': results['all_users_ao'][start_idx:end_idx],
            'Agent (Best)': results['agent_results'][start_idx:end_idx]
        }
        
        x = np.arange(len(scenarios))
        width = 0.15
        multiplier = 0
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['gray', 'red', 'green', 'blue', 'orange']
        
        for i, (attribute, measurement) in enumerate(data.items()):
            offset = width * multiplier
            bars = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, measurement):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=9)
            
            multiplier += 1
        
        ax.set_xlabel('Scenarios', fontweight='bold')
        ax.set_ylabel('Average Delta SNR (dB)', fontweight='bold')
        ax.set_title(f'Average Delta SNR Comparison - {scenarios[0]} to {scenarios[-1]}', fontweight='bold')
        ax.set_xticks(x + width * 2, scenarios)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/{filename}.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Bar chart saved to {self.plots_dir}/{filename}.png")
    
    def save_results(self, scenarios: List[Dict], results: Dict):
        """Save detailed results to JSON"""
        output_data = {
            'scenarios': scenarios,
            'summary_results': results,
            'settings': self.generator.sim_settings
        }
        
        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, complex):
                return {'real': obj.real, 'imag': obj.imag}
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        output_data = convert_numpy(output_data)
        
        with open(f"{self.plots_dir}/concept_analysis_results.json", 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {self.plots_dir}/concept_analysis_results.json")

def main():
    """Main function to run the concept analysis"""
    print("=== Concept Analysis: RIS User Selection and Algorithm Comparison ===")
    print("Please ensure you are in the virtual environment (venv)")
    
    # Initialize analyzer
    analyzer = ConceptAnalyzer()
    
    # Generate 10 scenarios
    scenarios = analyzer.generate_scenarios(10)
    print(f"Generated {len(scenarios)} scenarios")
    
    # Plot scenario layouts
    analyzer.plot_scenario_layouts(scenarios)
    
    # Analyze all scenarios
    results = analyzer.analyze_scenarios(scenarios)
    
    # Create grouped bar charts
    analyzer.plot_grouped_bar_charts(results)
    
    # Save results
    analyzer.save_results(scenarios, results)
    
    # Print summary
    print("\n=== Summary Statistics ===")
    print(f"Average Delta SNR across all scenarios:")
    print(f"  Random: {np.mean(results['all_users_random']):.2f} dB")
    print(f"  GD: {np.mean(results['all_users_gd']):.2f} dB") 
    print(f"  Manifold: {np.mean(results['all_users_manifold']):.2f} dB")
    print(f"  AO: {np.mean(results['all_users_ao']):.2f} dB")
    print(f"  Agent: {np.mean(results['agent_results']):.2f} dB")
    
    improvement_over_random = np.mean(results['agent_results']) - np.mean(results['all_users_random'])
    print(f"\nAgent improvement over random: {improvement_over_random:.2f} dB")
    
    print("\nAnalysis completed successfully!")
    print(f"All plots saved in: {analyzer.plots_dir}/")

if __name__ == "__main__":
    main()