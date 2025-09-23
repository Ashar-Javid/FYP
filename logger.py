"""Unified Logging Module
Combines LLM response logging and metrics CSV logging into a single interface.

Features:
- Thread-safe log writing
- File rotation for text logs
- Metrics CSV writer with dynamic header management
- Simple facade functions for framework use
"""
from __future__ import annotations
import os
import csv
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from config import FRAMEWORK_CONFIG, ensure_directories

_lock = threading.Lock()

# Rotation settings for text logs
DEFAULT_MAX_SIZE = 512 * 1024  # 512 KB
MAX_ROTATIONS = 5
LLM_LOG_FILENAME = "framework.log"  # unified text log (includes LLM responses + other events)

# Default metrics fields (can be extended at runtime)
DEFAULT_METRICS_FIELDS = [
    "session_id","iteration","timestamp","selected_users","algorithm","power_dBm","power_change_dB",
    "success","overall_score","avg_delta_improvement","worst_user_improvement","users_satisfied_before",
    "users_satisfied_after","satisfaction_improvement","power_waste_score","fairness_before","fairness_after",
    "fairness_improvement","execution_time_s"
]

# --- Sanitization & rotation utilities --- #

def _sanitize(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    return ''.join(ch if ch.isprintable() or ch in '\n\r\t' else '?' for ch in text)

def _rotate_if_needed(path: str, max_size: int = DEFAULT_MAX_SIZE):
    if not os.path.exists(path):
        return
    if os.path.getsize(path) < max_size:
        return
    for i in range(MAX_ROTATIONS, 0, -1):
        older = f"{path}.{i}"
        if i == MAX_ROTATIONS and os.path.exists(older):
            try:
                os.remove(older)
            except OSError:
                pass
        prev = f"{path}.{i-1}" if i > 1 else path
        if os.path.exists(prev):
            try:
                os.replace(prev, older)
            except OSError:
                pass

# --- Text / event logging (LLM responses etc.) --- #

def log_event(message: str, *, session_id: Optional[str] = None):
    """Generic event logger (writes to unified framework.log)."""
    ensure_directories()
    logs_dir = FRAMEWORK_CONFIG["logs_dir"]
    log_path = os.path.join(logs_dir, LLM_LOG_FILENAME)
    timestamp = datetime.utcnow().isoformat()
    entry = f"[{timestamp}] session={session_id or 'NA'} {message.strip()}\n"
    with _lock:
        _rotate_if_needed(log_path)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(entry)


def log_llm_response(agent: str, phase: str, prompt: str, response: str, session_id: Optional[str] = None):
    """Structured LLM response logging (compatible with previous API)."""
    safe_prompt = _sanitize(prompt)[:2000]
    safe_response = _sanitize(response)
    divider = '-' * 40
    message = (f"agent={agent} phase={phase}\nPROMPT: {safe_prompt}\nRESPONSE: {safe_response}\n{divider}")
    log_event(message, session_id=session_id)

# --- Metrics CSV logging --- #

class MetricsCSVLogger:
    """CSV-based metrics logger. Maintains a header and appends iteration rows."""
    def __init__(self, session_id: str, filename: Optional[str] = None, fields: Optional[List[str]] = None):
        ensure_directories()
        self.session_id = session_id
        self.fields = fields or DEFAULT_METRICS_FIELDS
        metrics_dir = FRAMEWORK_CONFIG["metrics_dir"]
        if filename is None:
            filename = f"metrics_{session_id}.csv"
        self.path = os.path.join(metrics_dir, filename)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.fields)

    def log_iteration(self, iteration_result: Dict[str, Any]):
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
            csv.writer(f).writerow(row)

# Backwards compatibility alias
MetricsLogger = MetricsCSVLogger

__all__ = [
    'log_llm_response','log_event','MetricsCSVLogger','MetricsLogger'
]
