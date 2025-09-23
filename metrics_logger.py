"""Metrics Logger
Writes per-iteration summarized metrics to a CSV for research analysis.
"""
from __future__ import annotations
import os
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime
from config import FRAMEWORK_CONFIG, ensure_directories

DEFAULT_FIELDS = [
    "session_id",
    "iteration",
    "timestamp",
    "selected_users",
    "algorithm",
    "power_dBm",
    "power_change_dB",
    "success",
    "overall_score",
    "avg_delta_improvement",
    "worst_user_improvement",
    "users_satisfied_before",
    "users_satisfied_after",
    "satisfaction_improvement",
    "power_waste_score",
    "fairness_before",
    "fairness_after",
    "fairness_improvement",
    "execution_time_s"
]

class MetricsLogger:
    def __init__(self, session_id: str, filename: Optional[str] = None):
        ensure_directories()
        self.session_id = session_id
        metrics_dir = FRAMEWORK_CONFIG["metrics_dir"]
        if filename is None:
            filename = f"metrics_{session_id}.csv"
        self.path = os.path.join(metrics_dir, filename)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(DEFAULT_FIELDS)

    def log_iteration(self, iteration_result: Dict[str, Any]):
        # Extract values with safe defaults
        coord = iteration_result.get("coordinator_decision", {})
        eval_data = iteration_result.get("evaluation", {})
        metrics = {k: eval_data.get(k) for k in [
            "avg_delta_improvement","worst_user_improvement","users_satisfied_before","users_satisfied_after",
            "satisfaction_improvement","power_waste_score","fairness_before","fairness_after","fairness_improvement"
        ]}
        row = [
            self.session_id,
            iteration_result.get("iteration"),
            iteration_result.get("timestamp"),
            ";".join(str(u) for u in coord.get("selected_users", [])),
            coord.get("selected_algorithm"),
            iteration_result.get("current_power_dBm"),
            coord.get("power_change_dB"),
            iteration_result.get("success"),
            eval_data.get("overall_score"),
            metrics.get("avg_delta_improvement"),
            metrics.get("worst_user_improvement"),
            metrics.get("users_satisfied_before"),
            metrics.get("users_satisfied_after"),
            metrics.get("satisfaction_improvement"),
            metrics.get("power_waste_score"),
            metrics.get("fairness_before"),
            metrics.get("fairness_after"),
            metrics.get("fairness_improvement"),
            iteration_result.get("execution_time")
        ]
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

__all__ = ["MetricsLogger"]
