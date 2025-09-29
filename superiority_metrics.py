"""
Enhanced Metrics and Visualizations for Agentic System Superiority
================================================================

This module adds comprehensive fairness, efficiency, and intelligence metrics
specifically designed to showcase the agentic system's superiority.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import List, Dict, Any, Tuple
try:
    import seaborn as sns
    import pandas as pd
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn and pandas not available. Some visualizations will be simplified.")
from datetime import datetime

# Configure matplotlib to avoid font glyph warnings
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Use DejaVu Sans instead of Times New Roman
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
# Suppress font warnings
import warnings
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")

class AdvancedMetricsEngine:
    """Advanced metrics calculation and visualization for agentic superiority"""
    
    def __init__(self):
        # Publication-ready color scheme with high contrast and accessibility
        self.color_scheme = {
            # Agentic systems (warm colors showing intelligence)
            'gradient_descent_adam_multi': '#E74C3C',      # Red for GD
            'manifold_optimization_adam_multi': '#3498DB', # Blue for Manifold  
            'alternating_optimization_multi': '#2ECC71',   # Green for AO
            'random_baseline': '#95A5A6',                  # Gray for random
            
            # Agentic Framework (our intelligent system)
            'Agentic_Coordinator': '#8E44AD',    # Purple for our agentic system
            
            # Aliases for backward compatibility
            'Agent': '#8E44AD',          # Purple for agentic coordinator
            'Manifold': '#3498DB',       # Blue for manifold
            'GD': '#E74C3C',            # Red for GD
            'AO': '#2ECC71',            # Green for AO
            'Random': '#95A5A6'         # Gray for random
        }
        
        # Publication-ready algorithm names
        self.algorithm_names = {
            'gradient_descent_adam_multi': 'Gradient Descent\n(Adam)',
            'manifold_optimization_adam_multi': 'Manifold Optimization\n(Adam)', 
            'alternating_optimization_multi': 'Alternating\nOptimization',
            'random_baseline': 'Random\nBaseline',
            'Agentic_Coordinator': 'Agentic\nCoordinator',
            'Agent': 'Agentic\nCoordinator',
            'Manifold': 'Manifold\nOptimization',
            'GD': 'Gradient\nDescent',
            'AO': 'Alternating\nOptimization',
            'Random': 'Random\nBaseline'
        }
        
        # Matplotlib style settings for publication quality
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Computer Modern Roman'],
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2.0,
            'patch.linewidth': 1.2
        })
    
    def get_display_name(self, algorithm_key: str) -> str:
        """Get publication-ready display name for algorithm"""
        return self.algorithm_names.get(algorithm_key, algorithm_key.replace('_', ' ').title())
    
    def get_algorithm_color(self, algorithm_key: str) -> str:
        """Get color for algorithm with fallback"""
        return self.color_scheme.get(algorithm_key, '#34495E')  # Dark blue-gray fallback
    
    def calculate_jains_fairness_index(self, delta_snrs: List[float]) -> float:
        """Calculate Jain's Fairness Index (0-1, higher = more fair)"""
        if not delta_snrs:
            return 0.0
        
        # Convert to satisfaction scores (0-1)
        satisfactions = [max(0, min(1, (d + 10) / 20)) for d in delta_snrs]  # Normalize to 0-1
        
        sum_x = sum(satisfactions)
        sum_x_squared = sum(x**2 for x in satisfactions)
        n = len(satisfactions)
        
        if sum_x_squared == 0:
            return 1.0
        
        jfi = (sum_x**2) / (n * sum_x_squared)
        return jfi
    
    def calculate_service_quality_disparity(self, delta_snrs: List[float]) -> float:
        """Calculate Service Quality Disparity (lower = more fair)"""
        if not delta_snrs:
            return 0.0
        return max(delta_snrs) - min(delta_snrs)
    
    def calculate_energy_efficiency_per_satisfied_user(self, delta_snrs: List[float], power_dbm: float) -> float:
        """Calculate Energy Efficiency per Satisfied User (higher = better)"""
        satisfied_users = sum(1 for d in delta_snrs if d >= 0)
        power_linear = 10**((power_dbm - 30) / 10)  # Convert to watts
        
        if power_linear == 0:
            return 0.0
        return satisfied_users / power_linear
    
    def calculate_power_waste_index(self, delta_snrs: List[float]) -> float:
        """Calculate Power Waste Index (lower = less waste)"""
        if not delta_snrs:
            return 0.0
        
        # Power waste = excess beyond 2dB safety margin
        excess_power = sum(max(0, delta - 2) for delta in delta_snrs)
        return excess_power / len(delta_snrs)
    

    
    def create_fairness_comparison_plot(self, results_dict: Dict[str, Dict], save_path: str = None) -> str:
        """Create publication-ready comprehensive fairness comparison plot"""
        methods = list(results_dict.keys())
        display_names = [self.get_display_name(m) for m in methods]
        
        # Calculate all fairness metrics
        jfi_scores = []
        sqd_scores = []
        variance_scores = []
        
        for method in methods:
            delta_snrs = results_dict[method]['delta_snrs']
            
            jfi = self.calculate_jains_fairness_index(delta_snrs)
            sqd = self.calculate_service_quality_disparity(delta_snrs)
            variance = np.var(delta_snrs)
            
            jfi_scores.append(jfi)
            sqd_scores.append(sqd)
            variance_scores.append(variance)
        
        # Create subplot figure with publication styling
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Resource Allocation Fairness Analysis:\nComparative Performance of Optimization Algorithms', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Publication-ready colors with high contrast
        colors = [self.get_algorithm_color(m) for m in methods]
        
        # 1. Jain's Fairness Index (higher = better)
        bars1 = axes[0,0].bar(display_names, jfi_scores, color=colors, alpha=0.85, 
                             edgecolor='white', linewidth=2.5)
        
        # Highlight best performer with star and improved annotation
        best_jfi_idx = np.argmax(jfi_scores)
        if best_jfi_idx < len(bars1):
            bars1[best_jfi_idx].set_edgecolor('#FFD700')
            bars1[best_jfi_idx].set_linewidth(4)
            axes[0,0].annotate(f'◆ {jfi_scores[best_jfi_idx]:.3f}', 
                              xy=(best_jfi_idx, jfi_scores[best_jfi_idx]), 
                              xytext=(0, 15), textcoords='offset points',
                              ha='center', va='bottom', fontweight='bold', fontsize=13,
                              bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFD700', 
                                       alpha=0.9, edgecolor='black', linewidth=1),
                              color='black')
        
        axes[0,0].set_title("Jain's Fairness Index\n(1.0 = Perfect Fairness)", fontweight='bold', pad=15)
        axes[0,0].set_ylabel('Fairness Index', fontweight='bold')
        axes[0,0].grid(axis='y', alpha=0.4, linestyle='--')
        axes[0,0].set_ylim(0, 1.05)
        axes[0,0].tick_params(axis='x', rotation=15)
        
        # 2. Service Quality Disparity (lower = better)
        bars2 = axes[0,1].bar(display_names, sqd_scores, color=colors, alpha=0.85, 
                             edgecolor='white', linewidth=2.5)
        
        # Highlight best performer (lowest SQD)
        best_sqd_idx = np.argmin(sqd_scores)
        if best_sqd_idx < len(bars2):
            bars2[best_sqd_idx].set_edgecolor('#FFD700')
            bars2[best_sqd_idx].set_linewidth(4)
            axes[0,1].annotate(f'Best: {sqd_scores[best_sqd_idx]:.1f} dB', 
                              xy=(best_sqd_idx, sqd_scores[best_sqd_idx]), 
                              xytext=(0, 15), textcoords='offset points',
                              ha='center', va='bottom', fontweight='bold', fontsize=13,
                              bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFD700', 
                                       alpha=0.9, edgecolor='black', linewidth=1),
                              color='black')
        
        axes[0,1].set_title('Service Quality Disparity\n(0 dB = Perfect Equity)', fontweight='bold', pad=15)
        axes[0,1].set_ylabel('Quality Disparity (dB)', fontweight='bold')
        axes[0,1].grid(axis='y', alpha=0.4, linestyle='--')
        axes[0,1].tick_params(axis='x', rotation=15)
        
        # 3. Performance Distribution Analysis (violin plot for publication quality)
        delta_data = [results_dict[method]['delta_snrs'] for method in methods]
        
        # Create violin plot for better distribution visualization
        parts = axes[1,0].violinplot(delta_data, positions=range(len(methods)), 
                                    showmeans=True, showmedians=True)
        
        # Color the violins with publication colors
        for i, (part, color) in enumerate(zip(parts['bodies'], colors)):
            part.set_facecolor(color)
            part.set_alpha(0.7)
            part.set_edgecolor('black')
            part.set_linewidth(1.2)
        
        # Style the violin plot elements
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(2)
        
        axes[1,0].set_xticks(range(len(methods)))
        axes[1,0].set_xticklabels(display_names, rotation=15)
        axes[1,0].set_title('Performance Distribution Analysis\n(Narrower = More Consistent)', fontweight='bold', pad=15)
        axes[1,0].set_ylabel('Delta SNR (dB)', fontweight='bold')
        axes[1,0].grid(axis='y', alpha=0.4, linestyle='--')
        axes[1,0].axhline(y=0, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8, 
                         label='QoS Threshold')
        axes[1,0].legend(frameon=True, fancybox=True, shadow=True)
        
        # 4. Performance Trade-off Analysis (Satisfaction vs Fairness)
        satisfaction_rates = []
        for method in methods:
            delta_snrs = results_dict[method]['delta_snrs']
            sat_rate = sum(1 for d in delta_snrs if d >= 0) / len(delta_snrs)
            satisfaction_rates.append(sat_rate)
        
        # Create enhanced scatter plot with publication styling
        for i, (method, disp_name) in enumerate(zip(methods, display_names)):
            axes[1,1].scatter(satisfaction_rates[i], jfi_scores[i], 
                             c=colors[i], s=300, alpha=0.8, 
                             edgecolors='white', linewidth=3,
                             label=disp_name, zorder=5)
        
        # Add trend line if possible
        try:
            # Check if we have enough points and they're not all the same
            if len(satisfaction_rates) >= 2 and len(set(satisfaction_rates)) > 1:
                with np.errstate(all='ignore'):  # Suppress polyfit warnings
                    z = np.polyfit(satisfaction_rates, jfi_scores, 1)
                    if np.isfinite(z).all():  # Only plot if coefficients are valid
                        p = np.poly1d(z)
                        x_trend = np.linspace(min(satisfaction_rates), max(satisfaction_rates), 100)
                        axes[1,1].plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=1.5, 
                                      label='Trend', zorder=1)
        except:
            pass  # Skip trend line if not possible
        
        # Highlight the optimal region (top-right quadrant)
        axes[1,1].axhspan(0.7, 1.0, 0.7, 1.0, alpha=0.1, color='green', 
                         label='Optimal Region')
        
        axes[1,1].set_xlabel('User Satisfaction Rate', fontweight='bold')
        axes[1,1].set_ylabel("Jain's Fairness Index", fontweight='bold')
        axes[1,1].set_title('Optimization Trade-off Analysis\n(Top-Right = Optimal Performance)', 
                           fontweight='bold', pad=15)
        axes[1,1].grid(True, alpha=0.4, linestyle='--')
        axes[1,1].set_xlim(0, 1.05)
        axes[1,1].set_ylim(0, 1.05)
        axes[1,1].legend(frameon=True, fancybox=True, shadow=True, 
                        bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout(pad=3.0)
        
        # Add publication metadata
        plt.figtext(0.02, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                                f'Algorithms: {len(methods)} | Users: {len(results_dict[methods[0]]["delta_snrs"])}',
                   fontsize=9, alpha=0.7, style='italic')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', metadata={'Title': 'Resource Allocation Fairness Analysis',
                                                 'Subject': 'Agentic Multi-User RIS Optimization',
                                                 'Creator': 'Advanced Metrics Engine'})
            plt.close()  # Free memory
            return save_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fairness_superiority_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', metadata={'Title': 'Resource Allocation Fairness Analysis',
                                                 'Subject': 'Agentic Multi-User RIS Optimization',
                                                 'Creator': 'Advanced Metrics Engine'})
            plt.close()  # Free memory
            return filename
    
    def create_power_efficiency_analysis(self, results_dict: Dict[str, Dict], save_path: str = None) -> str:
        """Create publication-ready comprehensive power efficiency analysis"""
        methods = list(results_dict.keys())
        display_names = [self.get_display_name(m) for m in methods]
        
        # Calculate efficiency metrics
        eepsu_scores = []
        pwi_scores = []
        power_consumptions = []
        satisfaction_rates = []
        
        for method in methods:
            delta_snrs = results_dict[method]['delta_snrs']
            power_dbm = results_dict[method]['power_dbm']
            
            eepsu = self.calculate_energy_efficiency_per_satisfied_user(delta_snrs, power_dbm)
            pwi = self.calculate_power_waste_index(delta_snrs)
            power_linear = 10**((power_dbm - 30) / 10)  # Convert to watts
            sat_rate = sum(1 for d in delta_snrs if d >= 0) / len(delta_snrs)
            
            eepsu_scores.append(eepsu)
            pwi_scores.append(pwi)
            power_consumptions.append(power_linear)
            satisfaction_rates.append(sat_rate)
        
        # Create publication-ready figure with enhanced layout
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('Energy Efficiency Analysis:\nComparative Performance of Multi-User RIS Optimization Algorithms', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Create enhanced subplot layout
        axes = []
        axes.append(fig.add_subplot(221))  # EEPSU
        axes.append(fig.add_subplot(222))  # PWI
        axes.append(fig.add_subplot(223))  # Trade-off
        axes.append(fig.add_subplot(224, projection='polar'))  # Radar chart
        
        colors = [self.get_algorithm_color(m) for m in methods]
        axes.append(fig.add_subplot(223))  # Trade-off
        axes.append(fig.add_subplot(224, projection='polar'))  # Radar chart
        
        colors = [self.get_algorithm_color(m) for m in methods]
        
        # 1. Energy Efficiency per Satisfied User (higher = better)
        bars1 = axes[0].bar(display_names, eepsu_scores, color=colors, alpha=0.85, 
                           edgecolor='white', linewidth=2.5)
        
        best_eepsu_idx = np.argmax(eepsu_scores)
        if best_eepsu_idx < len(bars1):
            bars1[best_eepsu_idx].set_edgecolor('#FFD700')
            bars1[best_eepsu_idx].set_linewidth(4)
            axes[0].annotate(f'Best: {eepsu_scores[best_eepsu_idx]:.2f}', 
                              xy=(best_eepsu_idx, eepsu_scores[best_eepsu_idx]), 
                              xytext=(0, 15), textcoords='offset points',
                              ha='center', va='bottom', fontweight='bold', fontsize=12,
                              bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFD700', 
                                       alpha=0.9, edgecolor='black', linewidth=1),
                              color='black')
        
        axes[0].set_title('Energy Efficiency per Satisfied User\n(Satisfied Users per Watt)', 
                         fontweight='bold', pad=15)
        axes[0].set_ylabel('EEPSU (users/watt)', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.4, linestyle='--')
        axes[0].tick_params(axis='x', rotation=15)
        
        # 2. Power Waste Index (lower = better)
        bars2 = axes[1].bar(display_names, pwi_scores, color=colors, alpha=0.85, 
                           edgecolor='white', linewidth=2.5)
        
        best_pwi_idx = np.argmin(pwi_scores)
        if best_pwi_idx < len(bars2):
            bars2[best_pwi_idx].set_edgecolor('#FFD700')
            bars2[best_pwi_idx].set_linewidth(4)
            axes[1].annotate(f'Best: {pwi_scores[best_pwi_idx]:.1f} dB', 
                              xy=(best_pwi_idx, pwi_scores[best_pwi_idx]), 
                              xytext=(0, 15), textcoords='offset points',
                              ha='center', va='bottom', fontweight='bold', fontsize=12,
                              bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFD700', 
                                       alpha=0.9, edgecolor='black', linewidth=1),
                              color='black')
        
        axes[1].set_title('Power Waste Index\n(0 dB = No Waste)', fontweight='bold', pad=15)
        axes[1].set_ylabel('Power Waste (dB)', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.4, linestyle='--')
        axes[1].tick_params(axis='x', rotation=15)
        
        # 3. Power-Performance Trade-off Analysis
        # Create enhanced scatter plot with publication styling
        for i, (method, disp_name) in enumerate(zip(methods, display_names)):
            axes[2].scatter(power_consumptions[i], satisfaction_rates[i], 
                           c=colors[i], s=350, alpha=0.8, 
                           edgecolors='white', linewidth=3,
                           label=disp_name, zorder=5)
        
        # Add Pareto frontier analysis
        try:
            # Calculate efficiency metric (satisfaction per watt)
            efficiency_scores = [sat/power for sat, power in zip(satisfaction_rates, power_consumptions)]
            best_efficiency_idx = np.argmax(efficiency_scores)
            
            # Highlight most efficient algorithm
            axes[2].scatter(power_consumptions[best_efficiency_idx], satisfaction_rates[best_efficiency_idx],
                           s=500, facecolors='none', edgecolors='gold', linewidth=4, 
                           zorder=10, alpha=0.8)
            axes[2].annotate('Most Efficient', 
                           (power_consumptions[best_efficiency_idx], satisfaction_rates[best_efficiency_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           fontweight='bold', fontsize=11, color='#B8860B',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFD700', alpha=0.8))
        except:
            pass  # Skip if calculation fails
        
        # Add ideal region indicator
        min_power = min(power_consumptions)
        max_sat = max(satisfaction_rates)
        axes[2].axvspan(0, min_power * 1.1, 0.8, 1.0, alpha=0.1, color='green', 
                       label='Optimal Region')
        
        axes[2].set_xlabel('Power Consumption (Watts)', fontweight='bold')
        axes[2].set_ylabel('User Satisfaction Rate', fontweight='bold')
        axes[2].set_title('Power-Performance Efficiency Analysis\n(Top-Left = Optimal)', 
                         fontweight='bold', pad=15)
        axes[2].grid(True, alpha=0.4, linestyle='--')
        axes[2].legend(frameon=True, fancybox=True, shadow=True, 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Multi-Dimensional Performance Radar
        metrics_names = ['Energy\nEfficiency', 'Low Power\nWaste', 'User\nSatisfaction', 'Power\nConservation']
        
        # Normalize metrics for radar chart (publication-ready)
        normalized_data = {}
        display_mapping = {}
        
        for i, (method, disp_name) in enumerate(zip(methods, display_names)):
            # Normalize each metric to 0-1 scale with improved formulations
            norm_eepsu = min(1.0, eepsu_scores[i] / max(eepsu_scores)) if max(eepsu_scores) > 0 else 0
            norm_pwi = 1 - min(1.0, pwi_scores[i] / max(pwi_scores)) if max(pwi_scores) > 0 else 1  # Inverted
            norm_sat = satisfaction_rates[i]
            norm_power = 1 - min(1.0, power_consumptions[i] / max(power_consumptions)) if max(power_consumptions) > 0 else 1  # Inverted
            
            normalized_data[method] = [norm_eepsu, norm_pwi, norm_sat, norm_power]
            display_mapping[method] = disp_name
        
        # Create publication-ready radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax_radar = axes[3]
        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_thetagrids(np.degrees(angles[:-1]), metrics_names, fontsize=11)
        
        # Plot each algorithm with enhanced styling
        for method, values in normalized_data.items():
            values += values[:1]  # Complete the circle
            color = self.get_algorithm_color(method)
            disp_name = display_mapping[method]
            
            ax_radar.plot(angles, values, 'o-', linewidth=3, label=disp_name, 
                         color=color, markersize=6, alpha=0.8)
            ax_radar.fill(angles, values, alpha=0.15, color=color)
        
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Multi-Dimensional Performance Profile\n(Larger Area = Superior Performance)', 
                          fontweight='bold', pad=25)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), 
                       frameon=True, fancybox=True, shadow=True)
        ax_radar.grid(True, alpha=0.3)
        ax_radar.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        
        # Add publication metadata
        plt.figtext(0.02, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                                f'Algorithms: {len(methods)} | Power Range: {min(power_consumptions):.1f}-{max(power_consumptions):.1f}W',
                   fontsize=9, alpha=0.7, style='italic')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', metadata={'Title': 'Energy Efficiency Analysis',
                                                 'Subject': 'Multi-User RIS Optimization',
                                                 'Creator': 'Advanced Metrics Engine'})
            plt.close()  # Free memory
            return save_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"power_efficiency_superiority_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', metadata={'Title': 'Energy Efficiency Analysis',
                                                 'Subject': 'Multi-User RIS Optimization',
                                                 'Creator': 'Advanced Metrics Engine'})
            plt.close()  # Free memory
            return filename
    


# Convenience functions for easy integration
def create_comprehensive_superiority_analysis(results_dict: Dict[str, Dict], output_dir: str = "plots") -> List[str]:
    """Create all superiority analysis plots"""
    engine = AdvancedMetricsEngine()
    
    plots = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all analysis plots
    fairness_plot = engine.create_fairness_comparison_plot(
        results_dict, f"{output_dir}/fairness_superiority_{timestamp}.png"
    )
    plots.append(fairness_plot)
    
    efficiency_plot = engine.create_power_efficiency_analysis(
        results_dict, f"{output_dir}/power_efficiency_superiority_{timestamp}.png"
    )
    plots.append(efficiency_plot)
    

    
    return plots