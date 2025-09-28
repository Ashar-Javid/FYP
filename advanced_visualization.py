"""
Advanced Visualization Module for RIS Optimization Framework
===========================================================

This module provides enhanced plotting capabilities for the RIS optimization framework,
with intelligent highlighting of best-performing methods and comprehensive visualization
of algorithm comparisons and performance metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import math

class RISVisualizationEngine:
    """Advanced visualization engine for RIS optimization results"""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = output_dir
        self.color_scheme = {
            'Agent': '#FF6B35',      # Bright orange for our agent
            'Manifold': '#FF6B35',   # Same for manifold
            'GD': '#4A90E2',         # Blue for GD
            'AO': '#7ED321',         # Green for AO
            'Random': '#9B9B9B'      # Gray for random
        }
        
    def _get_method_color(self, method_label: str) -> Tuple[str, str]:
        """Get color scheme for a method"""
        base_color = self.color_scheme.get(method_label, '#9B9B9B')
        # Darker edge color
        edge_color = self._darken_color(base_color)
        return base_color, edge_color
    
    def _darken_color(self, hex_color: str, factor: float = 0.8) -> str:
        """Darken a hex color by a factor"""
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Darken
        darkened = tuple(int(c * factor) for c in rgb)
        # Convert back to hex
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
    
    def _find_best_performer(self, values: List[float], metric_type: str) -> int:
        """Find index of best performing method"""
        valid_values = [(i, v) for i, v in enumerate(values) if not np.isnan(v) and np.isfinite(v)]
        if not valid_values:
            return -1
        
        if metric_type in ['rwsw', 'psd1']:  # Higher is better
            return max(valid_values, key=lambda x: x[1])[0]
        else:  # Lower is better (psd2)
            return min(valid_values, key=lambda x: x[1])[0]
    
    def _emphasize_bar_differences(self, bars, values: List[float], method_labels: List[str], 
                                 best_idx: int) -> None:
        """Emphasize differences between bars"""
        for i, (bar, value) in enumerate(zip(bars, values)):
            if i == best_idx:
                # Best performer: thick red border + higher alpha
                bar.set_linewidth(3)
                bar.set_edgecolor('#FF0000')
                bar.set_alpha(0.95)
            elif method_labels[i] in ['Agent']:
                # Our methods: medium emphasis
                bar.set_linewidth(2)
                bar.set_edgecolor('#CC5429')
                bar.set_alpha(0.85)
            else:
                # Other methods: standard appearance
                bar.set_linewidth(1.5)
                bar.set_alpha(0.75)
    
    def create_enhanced_metrics_plots(self, rwsw_vals: List[float], psd1_vals: List[float], 
                                    psd2_vals_dBm: List[float], method_labels: List[str]) -> Dict[str, str]:
        """Create enhanced metric plots with best performer highlighting"""
        
        # Prepare colors
        colors = []
        edge_colors = []
        for label in method_labels:
            color, edge = self._get_method_color(label)
            colors.append(color)
            edge_colors.append(edge)
        
        # Find best performers for each metric
        best_rwsw_idx = self._find_best_performer(rwsw_vals, 'rwsw')
        best_psd1_idx = self._find_best_performer(psd1_vals, 'psd1')
        best_psd2_idx = self._find_best_performer(psd2_vals_dBm, 'psd2')
        
        x = np.arange(len(method_labels))
        
        # 1. Enhanced RWS/W Plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(x, rwsw_vals, color=colors, alpha=0.85, edgecolor=edge_colors, linewidth=2)
        
        # Emphasize differences and highlight best performer
        self._emphasize_bar_differences(bars, rwsw_vals, method_labels, best_rwsw_idx)
        
        # Add annotation only for best performer
        if best_rwsw_idx >= 0:
            plt.annotate(f'{rwsw_vals[best_rwsw_idx]:.3f}', 
                        xy=(best_rwsw_idx, rwsw_vals[best_rwsw_idx]), 
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='gold', alpha=0.9, edgecolor='orange'))
        
        # Enhanced styling
        plt.xticks(x, method_labels, rotation=0, fontsize=12, fontweight='bold')
        plt.ylabel('RWS per Watt (arb)', fontsize=14, fontweight='bold')
        plt.title('Rank-Weighted Satisfaction per Watt\n(Higher is Better)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.gca().set_axisbelow(True)
        plt.ylim(0, max(rwsw_vals) * 1.15)
        plt.tight_layout()
        
        rwsw_path = os.path.join(self.output_dir, f"enhanced_rwsw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(rwsw_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Enhanced PSD Part 1 Plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(x, psd1_vals, color=colors, alpha=0.85, edgecolor=edge_colors, linewidth=2)
        
        self._emphasize_bar_differences(bars, psd1_vals, method_labels, best_psd1_idx)
        
        # Add annotation only for best performer
        if best_psd1_idx >= 0:
            plt.annotate(f'{psd1_vals[best_psd1_idx]:.3f}', 
                        xy=(best_psd1_idx, psd1_vals[best_psd1_idx]), 
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='gold', alpha=0.9, edgecolor='orange'))
        
        plt.xticks(x, method_labels, rotation=0, fontsize=12, fontweight='bold')
        plt.ylabel('Satisfaction Core S̃ (arb)', fontsize=14, fontweight='bold')
        plt.title('PSD Part 1: Satisfaction at Fixed Power\n(Higher is Better)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.gca().set_axisbelow(True)
        
        if any(v < 0 for v in psd1_vals):
            plt.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        psd1_path = os.path.join(self.output_dir, f"enhanced_psd1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(psd1_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Enhanced PSD Part 2 Plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(x, psd2_vals_dBm, color=colors, alpha=0.85, edgecolor=edge_colors, linewidth=2)
        
        self._emphasize_bar_differences(bars, psd2_vals_dBm, method_labels, best_psd2_idx)
        
        # Add annotation only for best performer (lowest value)
        if best_psd2_idx >= 0:
            plt.annotate(f'{psd2_vals_dBm[best_psd2_idx]:.1f} dBm', 
                        xy=(best_psd2_idx, psd2_vals_dBm[best_psd2_idx]), 
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9, edgecolor='green'))
        
        plt.xticks(x, method_labels, rotation=0, fontsize=12, fontweight='bold')
        plt.ylabel('Required Power (dBm)', fontsize=14, fontweight='bold')
        plt.title('PSD Part 2: Power to Reach Target Satisfaction\n(Lower is Better)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.gca().set_axisbelow(True)
        plt.tight_layout()
        
        psd2_path = os.path.join(self.output_dir, f"enhanced_psd2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(psd2_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Combined comparison plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
        
        # RWS/W subplot
        bars1 = ax1.bar(x, rwsw_vals, color=colors, alpha=0.85, edgecolor=edge_colors, linewidth=2)
        self._emphasize_bar_differences(bars1, rwsw_vals, method_labels, best_rwsw_idx)
        ax1.set_xticks(x)
        ax1.set_xticklabels(method_labels, rotation=45, ha='right')
        ax1.set_ylabel('RWS/W (arb)', fontweight='bold')
        ax1.set_title('RWS per Watt\n(Higher Better)', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max(rwsw_vals) * 1.1)
        
        # PSD Part 1 subplot
        bars2 = ax2.bar(x, psd1_vals, color=colors, alpha=0.85, edgecolor=edge_colors, linewidth=2)
        self._emphasize_bar_differences(bars2, psd1_vals, method_labels, best_psd1_idx)
        ax2.set_xticks(x)
        ax2.set_xticklabels(method_labels, rotation=45, ha='right')
        ax2.set_ylabel('Satisfaction S̃ (arb)', fontweight='bold')
        ax2.set_title('PSD Part 1\n(Higher Better)', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        if any(v < 0 for v in psd1_vals):
            ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # PSD Part 2 subplot
        bars3 = ax3.bar(x, psd2_vals_dBm, color=colors, alpha=0.85, edgecolor=edge_colors, linewidth=2)
        self._emphasize_bar_differences(bars3, psd2_vals_dBm, method_labels, best_psd2_idx)
        ax3.set_xticks(x)
        ax3.set_xticklabels(method_labels, rotation=45, ha='right')
        ax3.set_ylabel('Required Power (dBm)', fontweight='bold')
        ax3.set_title('PSD Part 2\n(Lower Better)', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.suptitle('Complete Performance Comparison: All Metrics', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        combined_path = os.path.join(self.output_dir, f"enhanced_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'enhanced_rwsw': rwsw_path,
            'enhanced_psd1': psd1_path, 
            'enhanced_psd2': psd2_path,
            'enhanced_combined': combined_path
        }
    
    def create_enhanced_algorithm_comparison(self, user_ids: List[int], 
                                           mean_delta_map: Dict[str, Dict[int, float]], 
                                           std_delta_map: Dict[str, Dict[int, float]], 
                                           algo_order: List[str],
                                           multi_run_mode: bool = False) -> str:
        """Create enhanced algorithm comparison plot"""
        
        num_algos = len(algo_order)
        x = np.arange(len(user_ids))
        width = 0.12 if num_algos > 6 else 0.15
        
        plt.figure(figsize=(max(14, 3*len(user_ids)), 8))
        
        # Find best performing algorithm (most users satisfied)
        satisfaction_rates = {}
        for algo in algo_order:
            means = [mean_delta_map[algo].get(uid, np.nan) for uid in user_ids]
            satisfied = sum(1 for m in means if not np.isnan(m) and m >= 0)
            total = len([m for m in means if not np.isnan(m)])
            satisfaction_rates[algo] = satisfied / total if total > 0 else 0
        
        best_algo = max(satisfaction_rates.keys(), key=lambda k: satisfaction_rates[k])
        
        for i, algo in enumerate(algo_order):
            means = [mean_delta_map[algo].get(uid, np.nan) for uid in user_ids]
            stds = [std_delta_map[algo].get(uid, 0.0) for uid in user_ids]
            positions = x + (i - num_algos/2)*width + width/2
            
            # Get color for this algorithm
            color, edge_color = self._get_method_color(algo)
            alpha = 0.95 if algo == best_algo else (0.85 if algo in ['Agent'] else 0.75)
            
            if multi_run_mode and algo not in ['Agent'] and any(s > 1e-6 for s in stds):
                bars = plt.bar(positions, means, width, label=algo, alpha=alpha, color=color,
                              yerr=stds, capsize=3, edgecolor=edge_color, linewidth=1)
            else:
                bars = plt.bar(positions, means, width, label=algo, alpha=alpha, color=color,
                              edgecolor=edge_color, linewidth=1)
            
            # Emphasize best algorithm
            if algo == best_algo:
                for bar in bars:
                    bar.set_linewidth(3)
                    bar.set_edgecolor('#FF0000')
        
        # Add horizontal line at zero (satisfaction threshold)
        plt.axhline(0, color='red', linewidth=2, linestyle='--', alpha=0.8, 
                    label='Satisfaction Threshold')
        
        # Enhanced styling
        plt.xticks(x, [f'User {uid}' for uid in user_ids], fontsize=12, fontweight='bold')
        plt.ylabel('Δ SNR (dB)', fontsize=14, fontweight='bold')
        plt.xlabel('Users', fontsize=14, fontweight='bold')
        
        title_suffix = '' if not multi_run_mode else f" (Mean ± Std)"
        plt.title(f'Algorithm Comparison: Per-User Performance{title_suffix}\n(Above 0 = Satisfied Users)', 
                  fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.gca().set_axisbelow(True)
        
        # Enhanced legend
        plt.legend(ncol=min(4, num_algos+1), fontsize=11, loc='upper left', 
                  bbox_to_anchor=(0, 1), framealpha=0.9)
        
        # Add performance summary for best algorithm
        if best_algo in satisfaction_rates:
            rate = satisfaction_rates[best_algo] * 100
            plt.text(0.02, 0.98, f'Best: {best_algo} ({rate:.1f}% users satisfied)', 
                    transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='gold', alpha=0.8))
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'enhanced_algorithm_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path


# Convenience functions for backward compatibility
def create_enhanced_metrics_plots(rwsw_vals: List[float], psd1_vals: List[float], 
                                psd2_vals_dBm: List[float], method_labels: List[str], 
                                output_dir: str) -> Dict[str, str]:
    """Convenience function for creating enhanced metric plots"""
    engine = RISVisualizationEngine(output_dir)
    return engine.create_enhanced_metrics_plots(rwsw_vals, psd1_vals, psd2_vals_dBm, method_labels)


def create_enhanced_algorithm_comparison(user_ids: List[int], 
                                       mean_delta_map: Dict[str, Dict[int, float]], 
                                       std_delta_map: Dict[str, Dict[int, float]], 
                                       algo_order: List[str], output_dir: str,
                                       multi_run_mode: bool = False) -> str:
    """Convenience function for creating enhanced algorithm comparison"""
    engine = RISVisualizationEngine(output_dir)
    return engine.create_enhanced_algorithm_comparison(user_ids, mean_delta_map, std_delta_map, 
                                                     algo_order, multi_run_mode)