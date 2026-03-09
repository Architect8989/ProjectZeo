"""
core/learning/preference_generator.py

Formats AgentQ preference pairs into DPO training datasets.

Reads pairs from AgentQ store, converts them to the HuggingFace TRL
DPO format (prompt / chosen / rejected), and exports as JSONL for
grpo_trainer.py to consume.

Reference:
    Rafailov et al., NeurIPS 2023 — "Direct Preference Optimization"
    TRL DPOTrainer: https://huggingface.co/docs/trl/dpo_trainer
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

_OUTPUT_DIR  = os.path.expanduser(os.environ.get("PROJECTZEO_DPO_DATA_DIR", "~/.projectzeo/dpo_data"))
_MIN_QUALITY = float(os.environ.get("PROJECTZEO_DPO_MIN_QUALITY", "0.3"))
_MAX_EXPORT  = int(os.environ.get("PROJECTZEO_DPO_MAX_EXPORT", "1000"))


def _steps_to_text(steps: List[Dict[str, Any]]) -> str:
    parts = []
    for step in steps:
        obs    = step.get("observation", "")[:200]
        action = step.get("action", {})
        op     = action.get("operation", "unknown")
        target = action.get("label") or action.get("text") or action.get("coordinate", "")
        reward = step.get("reward", 0.0)
        parts.append(f"[{op}] {target} (r={reward:.2f})")
        if obs:
            parts.append(f"  obs: {obs}")
    return "\n".join(parts)


def _build_dpo_record(pair) -> Dict[str, Any]:
    prompt = (
        f"Task: {pair.objective}\n"
        f"Application: {pair.app_name}\n\n"
        "Complete this task step by step."
    )
    chosen_text   = _steps_to_text(pair.chosen)
    rejected_text = _steps_to_text(pair.rejected)
    return {
        "prompt":   prompt,
        "chosen":   chosen_text,
        "rejected": rejected_text,
        "quality":  pair.quality,
        "pair_id":  pair.pair_id,
    }


class PreferenceGenerator:

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self._dir  = output_dir or _OUTPUT_DIR
        self._lock = threading.Lock()
        os.makedirs(self._dir, exist_ok=True)

    def generate_dataset(
        self,
        source: Optional[Any] = None,
        min_quality: float = _MIN_QUALITY,
        max_pairs: int = _MAX_EXPORT,
    ) -> str:
        """
        Generate a JSONL DPO dataset from AgentQ pairs.

        Args:
            source:      AgentQ instance (defaults to singleton)
            min_quality: minimum preference pair quality to include
            max_pairs:   maximum pairs in output

        Returns:
            Path to the generated JSONL file.
        """
        from core.learning.agent_q import get_agent_q
        aq = source or get_agent_q()

        pairs = aq.load_pairs(limit=max_pairs * 2)
        pairs = [p for p in pairs if p.quality >= min_quality]
        pairs.sort(key=lambda p: p.quality, reverse=True)
        pairs = pairs[:max_pairs]

        if not pairs:
            _logger.info("[PrefGen] No pairs meeting quality threshold — skipping dataset generation.")
            return ""

        ts   = int(time.time())
        path = os.path.join(self._dir, f"dpo_dataset_{ts}.jsonl")
        tmp  = path + ".tmp"

        with open(tmp, "w", encoding="utf-8") as f:
            for pair in pairs:
                record = _build_dpo_record(pair)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        os.replace(tmp, path)
        _logger.info("[PrefGen] Generated DPO dataset: %s (%d pairs)", path, len(pairs))
        return path

    def latest_dataset(self) -> Optional[str]:
        try:
            files = sorted(
                (e for e in os.scandir(self._dir) if e.name.startswith("dpo_dataset_") and e.name.endswith(".jsonl")),
                key=lambda e: e.stat().st_mtime,
                reverse=True,
            )
            return files[0].path if files else None
        except Exception:
            return None

    def dataset_stats(self) -> Dict[str, Any]:
        try:
            files = [e for e in os.scandir(self._dir) if e.name.endswith(".jsonl")]
            total = sum(1 for fn in files for _ in open(os.path.join(self._dir, fn.name), encoding="utf-8"))
        except Exception:
            total = 0
        return {
            "dataset_files": len(files) if "files" in dir() else 0,
            "total_records":  total,
            "output_dir":     self._dir,
        }


_instance: Optional[PreferenceGenerator] = None
_lock = threading.Lock()


def get_preference_generator() -> PreferenceGenerator:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PreferenceGenerator()
    return _instance
