"""Metrics evaluation and plotting (isolated module)
Computes and plots:
- RWS/W: Rank-Weighted Satisfaction per Watt
- PSD: Power-Satisfaction Dominance (two-part: fixed-power satisfaction lift and fixed-satisfaction power saving)

This module is optional and controlled via FRAMEWORK_CONFIG["metrics_eval"].
It reuses ScenarioTool, AlgorithmTool, and VisualizationTool without changing their APIs.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, DefaultDict
import os
import math
from datetime import datetime
import numpy as np

from config import FRAMEWORK_CONFIG
from tools import ScenarioTool, AlgorithmTool
from ris_framework import RISOptimizationFramework
from scenario import SIM_SETTINGS as _GLOBAL_SIM_SETTINGS
import random

# ---------------- Metric core ---------------- #

def _rank_weights(n: int, gamma: float) -> np.ndarray:
    ranks = np.arange(1, n+1)
    w = np.exp(-gamma * ranks)
    return w / (w.sum() + 1e-12)


def _phi_asym_log_component(delta: np.ndarray, alpha: float, beta: float, kappa: float) -> np.ndarray:
    # Component-level log transform on magnitudes to amplify tail differences while keeping sign asymmetry
    pos = np.maximum(0.0, delta)
    neg = np.maximum(0.0, -delta)
    pos_t = np.log1p(kappa * pos)
    neg_t = np.log1p(kappa * neg)
    return -alpha * neg_t + beta * pos_t


def _phi_asym_linear(delta: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    pos = np.maximum(0.0, delta)
    neg = np.maximum(0.0, -delta)
    return -alpha * neg + beta * pos


def compute_satisfaction_core(deltas: List[float], *, alpha: float, beta: float, gamma: float,
                              kappa: float, use_log: bool) -> float:
    x = np.array([d if np.isfinite(d) else 0.0 for d in deltas], dtype=float)
    # rank by ascending (worst first)
    order = np.argsort(x)
    x_sorted = x[order]
    w = _rank_weights(len(x_sorted), gamma)
    if use_log:
        phi = _phi_asym_log_component(x_sorted, alpha, beta, kappa)
    else:
        phi = _phi_asym_linear(x_sorted, alpha, beta)
    return float((w * phi).sum())


def compute_rws_per_watt(deltas: List[float], power_watts: float, params: Dict[str, Any]) -> float:
    alpha = float(params.get('alpha', 1.0))
    beta = float(params.get('beta', 0.1))
    gamma = float(params.get('gamma', math.log(2)))
    kappa = float(params.get('kappa', 1.0))
    epsW = float(params.get('epsilon_watts', 1e-3))
    use_log = bool(FRAMEWORK_CONFIG.get('metrics_eval', {}).get('log_component_level', True))
    S_tilde = compute_satisfaction_core(deltas, alpha=alpha, beta=beta, gamma=gamma, kappa=kappa, use_log=use_log)
    return S_tilde / max(power_watts, epsW)


def compute_psd_fixed_power(S_tilde_A: float, S_tilde_B: float) -> float:
    return S_tilde_A - S_tilde_B


def estimate_power_for_target(core_fn, target_S: float, nominal_power_W: float, power_grid_W: List[float]) -> Optional[float]:
    # Find minimal power meeting or exceeding target_S (using given grid)
    feasible = [(p, core_fn(p)) for p in power_grid_W]
    feasible = [(p, s) for (p, s) in feasible if np.isfinite(s)]
    if not feasible:
        return None
    # monotonicity not guaranteed; pick minimal p with s >= target
    feasible_sorted = sorted(feasible, key=lambda t: t[0])
    for p, s in feasible_sorted:
        if s >= target_S:
            return p
    return None


# ---------------- Evaluation harness ---------------- #

def _to_watts(power_dBm: float) -> float:
    return 10 ** (power_dBm/10.0) / 1000.0


def _power_grid_watts(nominal_power_dBm: float, offsets_dB: List[float]) -> List[float]:
    return [_to_watts(nominal_power_dBm + off) for off in offsets_dB]


def _extract_deltas(final_snrs: Dict[int, float], required_map: Dict[int, float]) -> List[float]:
    uids = list(required_map.keys())
    return [float(final_snrs.get(uid, np.nan) - required_map[uid]) for uid in uids]


def run_metrics_eval(per_session_plots_dir: str):
    cfg = FRAMEWORK_CONFIG.get('metrics_eval', {})
    if not cfg.get('enabled', False):
        return None

    iters = int(cfg.get('iterations', 10))
    m = int(cfg.get('num_scenarios', 5))
    mode = cfg.get('scenario_mode', 'list')
    scenarios = list(cfg.get('scenarios', []))
    params = cfg.get('metric_params', {})
    power_offsets = list(cfg.get('power_sweep_dB', [-6, -3, 0, 3]))

    scen_tool = ScenarioTool()
    algo_tool = AlgorithmTool()

    # Select scenarios
    available = list(scen_tool.available_scenarios.keys())
    if mode == 'random':
        rng = np.random.default_rng()
        chosen = list(rng.choice(available, size=min(m, len(available)), replace=False))
    else:
        # ordered selection from provided list, fallback to available
        pool = [s for s in scenarios if s in available] or available
        chosen = pool[:min(m, len(pool))]

    # Storage for aggregation per scenario
    rwsw_by_scenario: Dict[str, Dict[str, List[float]]] = {}  # scen -> method -> [values]
    S_atP_by_scenario: Dict[str, Dict[str, List[float]]] = {}
    P_atS_by_scenario: Dict[str, Dict[str, List[float]]] = {}

    # Methods to compare
    methods = [
        ('Our', 'manifold_optimization_adam_multi'),
        ('gradient_descent_adam_multi', 'GD'),
        ('manifold_optimization_adam_multi', 'Manifold'),
        ('alternating_optimization_adam_multi', 'AO')
    ]
    random_label = 'Random'

    # Iterate scenarios and iterations
    for scen in chosen:
        # Static info (does not depend on RNG)
        nominal_power_dBm = scen_tool.sim_settings['default_bs_power_dBm']
        nominal_power_W = _to_watts(nominal_power_dBm)
        static_users = scen_tool.available_scenarios[scen]['users']
        users_all_static = [u['id'] for u in static_users]
        req_map = {u['id']: u.get('req_snr_dB', 0.0) for u in static_users}
        # init containers for this scenario
        rwsw_by_scenario.setdefault(scen, {})
        S_atP_by_scenario.setdefault(scen, {})
        P_atS_by_scenario.setdefault(scen, {})

        base_seed = int(cfg.get('base_seed', 20250927))
        for it in range(1, iters+1):
            # Deterministic seed per (scenario, iteration)
            scen_hash = sum(ord(c) for c in scen) & 0xFFFFFFFF
            per_iter_seed = (int(base_seed) ^ scen_hash ^ (it * 1009)) & 0xFFFFFFFF

            # Run methods
            final_maps: Dict[str, Dict[int, float]] = {}

            # Our: run the agentic framework to convergence on this scenario (isolated sub-session)
            try:
                # Set RNGs
                np.random.seed(per_iter_seed)
                random.seed(per_iter_seed)
                # Temporarily override SIM_SETTINGS seed for scenario generation inside tools
                prev_seed = _GLOBAL_SIM_SETTINGS.get('seed', None)
                _GLOBAL_SIM_SETTINGS['seed'] = int(per_iter_seed)
                # temporarily disable metrics eval to avoid recursive triggering within the sub-session
                metrics_cfg = FRAMEWORK_CONFIG.get('metrics_eval', {})
                prev_enabled = bool(metrics_cfg.get('enabled', False))
                prev_default_scen = FRAMEWORK_CONFIG.get('default_scenario', '')
                try:
                    metrics_cfg['enabled'] = False
                    FRAMEWORK_CONFIG['default_scenario'] = scen
                    fw = RISOptimizationFramework()
                    sub_results = fw.run_optimization_session()
                    hist = sub_results.get('iteration_history', [])
                    if hist:
                        last_algo = hist[-1].get('algorithm_results', {})
                        final_maps['Our'] = last_algo.get('final_snrs', {})
                    else:
                        final_maps['Our'] = {uid: np.nan for uid in users_all_static}
                finally:
                    metrics_cfg['enabled'] = prev_enabled
                    FRAMEWORK_CONFIG['default_scenario'] = prev_default_scen
                    if prev_seed is not None:
                        _GLOBAL_SIM_SETTINGS['seed'] = prev_seed
                    else:
                        _GLOBAL_SIM_SETTINGS.pop('seed', None)
            except Exception:
                final_maps['Our'] = {uid: np.nan for uid in users_all_static}

            # Build a fresh scenario_data for this iteration for other methods
            np.random.seed(per_iter_seed)
            random.seed(per_iter_seed)
            scenario_data_it = scen_tool.get_scenario(scen)
            users_all = [u['id'] for u in scenario_data_it['scenario_data']['users']]

            # Other algorithms at fixed power on all users
            for label, algo_full in [('GD','gradient_descent_adam_multi'), ('Manifold','manifold_optimization_adam_multi'), ('AO','alternating_optimization_adam_multi')]:
                try:
                    # Ensure same RNG state for each method to share the same scenario realization
                    np.random.seed(per_iter_seed)
                    random.seed(per_iter_seed)
                    res = algo_tool.execute_algorithm(algo_full, scenario_data_it, users_all, nominal_power_dBm)
                    final_maps[label] = res.get('final_snrs', {})
                except Exception:
                    final_maps[label] = {uid: np.nan for uid in users_all}
            # Random baseline
            np.random.seed(per_iter_seed)
            final_maps[random_label] = {u['id']: u['achieved_snr_dB'] + float(np.random.uniform(0,3)) for u in scenario_data_it['scenario_data']['users']}

            # Compute independent metrics for each method
            # RWS/W and S at fixed power
            S_scores: Dict[str, float] = {}
            for label, fmap in final_maps.items():
                deltas = _extract_deltas(fmap, req_map)
                rwsw_val = compute_rws_per_watt(deltas, nominal_power_W, params)
                rwsw_by_scenario[scen].setdefault(label, []).append(rwsw_val)
                S_scores[label] = compute_satisfaction_core(deltas, alpha=params.get('alpha',1.0), beta=params.get('beta',0.1),
                                                            gamma=params.get('gamma', math.log(2)), kappa=params.get('kappa',1.0),
                                                            use_log=FRAMEWORK_CONFIG.get('metrics_eval', {}).get('log_component_level', True))
                S_atP_by_scenario[scen].setdefault(label, []).append(S_scores[label])

            # PSD Part 2: minimal power to reach a common target S*
            target_mode = FRAMEWORK_CONFIG.get('metrics_eval', {}).get('psd_target', 'best')
            if target_mode == 'median':
                S_target = float(np.nanmedian(list(S_scores.values())))
            elif target_mode == 'fixed_value':
                S_target = float(FRAMEWORK_CONFIG.get('metrics_eval', {}).get('psd_target_value', 0.0))
            else:
                # 'best' of methods at nominal power
                S_target = float(np.nanmax(list(S_scores.values())))

            grid_W = _power_grid_watts(nominal_power_dBm, power_offsets)

            def core_at_power(base_deltas: List[float], pW: float) -> float:
                scale_dB = 10.0 * math.log10(max(pW, 1e-9) / max(nominal_power_W, 1e-9))
                deltas_scaled = [d + scale_dB for d in base_deltas]
                return compute_satisfaction_core(deltas_scaled, alpha=params.get('alpha',1.0), beta=params.get('beta',0.1),
                                                 gamma=params.get('gamma', math.log(2)), kappa=params.get('kappa',1.0),
                                                 use_log=FRAMEWORK_CONFIG.get('metrics_eval', {}).get('log_component_level', True))

            for label, fmap in final_maps.items():
                base_deltas = _extract_deltas(fmap, req_map)
                def fn(pW: float, bd=base_deltas):
                    return core_at_power(bd, pW)
                p_needed_W = estimate_power_for_target(lambda p: fn(p), S_target, nominal_power_W, grid_W)
                if p_needed_W is None or not np.isfinite(p_needed_W) or p_needed_W <= 0:
                    p_needed_W = float('nan')
                P_atS_by_scenario[scen].setdefault(label, []).append(p_needed_W)

    # --- Plotting ---
    import matplotlib.pyplot as plt

    # Aggregate per scenario -> then aggregate across scenarios: single point per method
    scen_order = list(rwsw_by_scenario.keys())
    method_labels = sorted({lbl for scen in rwsw_by_scenario for lbl in rwsw_by_scenario[scen].keys()})
    # Exclude Random baseline from final plots
    method_labels = [lbl for lbl in method_labels if lbl.lower() != 'random']

    # Compute across-scenario means
    def across_scenarios_mean(metric_dict: Dict[str, Dict[str, List[float]]], label: str) -> float:
        vals = []
        for s in scen_order:
            v = float(np.nanmean(metric_dict.get(s, {}).get(label, [np.nan])))
            if np.isfinite(v):
                vals.append(v)
        return float(np.nanmean(vals)) if vals else float('nan')

    # RWS/W: single bar plot (one bar per method)
    rwsw_vals = [across_scenarios_mean(rwsw_by_scenario, lbl) for lbl in method_labels]
    plt.figure(figsize=(10,6))
    x = np.arange(len(method_labels))
    plt.bar(x, rwsw_vals, color='tab:blue', alpha=0.8)
    plt.xticks(x, method_labels, rotation=0)
    plt.ylabel('RWS per Watt (arb)')
    plt.title('Final Metric: Rank-Weighted Satisfaction per Watt (across scenarios)')
    plt.tight_layout()
    rwsw_path = os.path.join(per_session_plots_dir, f"rwsw_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(rwsw_path, dpi=150)
    plt.close()

    # PSD Part 1: S̃ at fixed power: single bar plot
    psd1_vals = [across_scenarios_mean(S_atP_by_scenario, lbl) for lbl in method_labels]
    plt.figure(figsize=(10,6))
    x = np.arange(len(method_labels))
    plt.bar(x, psd1_vals, color='tab:green', alpha=0.8)
    plt.xticks(x, method_labels, rotation=0)
    plt.ylabel('Satisfaction core S̃ (arb)')
    plt.title('Final Metric: PSD Part 1 (S̃ at nominal power)')
    plt.tight_layout()
    psd_part1_path = os.path.join(per_session_plots_dir, f"psd_part1_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(psd_part1_path, dpi=150)
    plt.close()

    # PSD Part 2: minimal power to reach common target S* -> convert to dBm; single bar plot
    def w_to_dBm(pW: float) -> float:
        if not np.isfinite(pW) or pW <= 0:
            return float('nan')
        return 10.0 * math.log10(pW * 1000.0)

    psd2_vals_dBm = []
    for lbl in method_labels:
        ys_W = [float(np.nanmean(P_atS_by_scenario.get(s, {}).get(lbl, [np.nan]))) for s in scen_order]
        ys_W = [v for v in ys_W if np.isfinite(v)]
        mean_W = float(np.nanmean(ys_W)) if ys_W else float('nan')
        psd2_vals_dBm.append(w_to_dBm(mean_W))

    plt.figure(figsize=(10,6))
    x = np.arange(len(method_labels))
    plt.bar(x, psd2_vals_dBm, color='tab:orange', alpha=0.8)
    plt.xticks(x, method_labels, rotation=0)
    plt.ylabel('Required power (dBm)')
    plt.title('Final Metric: PSD Part 2 (Power to reach S*)')
    plt.tight_layout()
    psd_part2_path = os.path.join(per_session_plots_dir, f"psd_part2_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(psd_part2_path, dpi=150)
    plt.close()

    return {
        'scenarios': scen_order,
        'rwsw_plot': rwsw_path,
        'psd_part1_plot': psd_part1_path,
        'psd_part2_plot': psd_part2_path
    }
