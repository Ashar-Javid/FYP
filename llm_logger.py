"""LLM Response Logging Utility
Logs raw (sanitized) LLM responses to a rolling log file for post-analysis.
"""
from __future__ import annotations
import os
import time
import threading
from datetime import datetime
from typing import Optional
from config import FRAMEWORK_CONFIG, ensure_directories

_lock = threading.Lock()

DEFAULT_MAX_SIZE = 512 * 1024  # 512 KB per log file before rotation
MAX_ROTATIONS = 5


def _sanitize(text: str) -> str:
    """Basic sanitization: collapse whitespace and strip control chars."""
    if not isinstance(text, str):
        text = str(text)
    # Replace problematic control characters except newlines
    return ''.join(ch if ch.isprintable() or ch in '\n\r\t' else '?' for ch in text)


def _rotate_if_needed(path: str, max_size: int = DEFAULT_MAX_SIZE):
    if not os.path.exists(path):
        return
    if os.path.getsize(path) < max_size:
        return
    # Rotate: existing.log -> existing.log.1 ... up to MAX_ROTATIONS
    for i in range(MAX_ROTATIONS, 0, -1):
        older = f"{path}.{i}"
        if i == MAX_ROTATIONS and os.path.exists(older):
            os.remove(older)
        prev = f"{path}.{i-1}" if i > 1 else path
        if os.path.exists(prev):
            os.rename(prev, older)


def log_llm_response(agent: str, phase: str, prompt: str, response: str, session_id: Optional[str] = None):
    """Append a raw LLM response to the rolling log file.

    Parameters:
        agent: 'coordinator' or 'evaluator'
        phase: logical phase label (e.g., 'decision', 'evaluation')
        prompt: original prompt (truncated for length)
        response: raw model output
        session_id: optional session identifier for correlation
    """
    ensure_directories()
    logs_dir = FRAMEWORK_CONFIG["logs_dir"]
    log_path = os.path.join(logs_dir, "llm_responses.log")

    safe_prompt = _sanitize(prompt)[:2000]
    safe_response = _sanitize(response)

    timestamp = datetime.utcnow().isoformat()
    header = f"[{timestamp}] session={session_id or 'NA'} agent={agent} phase={phase}\n"
    divider = "-" * 40 + "\n"

    entry = (
        header +
        f"PROMPT: {safe_prompt}\n" +
        f"RESPONSE: {safe_response}\n" +
        divider
    )

    with _lock:
        _rotate_if_needed(log_path)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(entry)

__all__ = ["log_llm_response"]
