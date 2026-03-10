"""
core/learning/nightly_consolidation.py

Nightly consolidation scheduler — runs after task hours to:
  1. Consolidate episodic memory into semantic memory (CORE loop)
  2. Trigger DPO dataset generation from AgentQ pairs
  3. Schedule GRPO training pass on queued tasks
  4. Prune stale knowledge vault entries

Runs as a background daemon thread. Can be triggered manually via
NightlyConsolidator.run_now() for testing.

Env vars:
  PROJECTZEO_CONSOLIDATION_ENABLED  1/0       (default: 1)
  PROJECTZEO_CONSOLIDATION_HOUR     0-23      (default: 3 = 3 AM)
  PROJECTZEO_CONSOLIDATION_MIN_PAIRS          (default: 10)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

_ENABLED      = os.environ.get("PROJECTZEO_CONSOLIDATION_ENABLED", "1").strip() != "0"
_HOUR         = int(os.environ.get("PROJECTZEO_CONSOLIDATION_HOUR", "3"))
_MIN_PAIRS    = int(os.environ.get("PROJECTZEO_CONSOLIDATION_MIN_PAIRS", "10"))
_POLL_SECS    = 60.0


class ConsolidationResult:
    def __init__(self) -> None:
        self.dpo_pairs_found:    int = 0
        self.dpo_dataset_path:   str = ""
        self.grpo_dataset_path:  str = ""
        self.memories_pruned:    int = 0
        self.errors:             List[str] = []
        self.started_at:         float = time.time()
        self.finished_at:        float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


class NightlyConsolidator:

    def __init__(
        self,
        knowledge_vault=None,
        episodic_synthesizer=None,
        on_complete: Optional[Callable[[ConsolidationResult], None]] = None,
    ) -> None:
        self._vault          = knowledge_vault
        self._synthesizer    = episodic_synthesizer
        self._on_complete    = on_complete
        self._thread: Optional[threading.Thread] = None
        self._stop           = threading.Event()
        self._last_run_date  = ""
        self._lock           = threading.Lock()
        self._running        = False

    def start(self) -> None:
        if not _ENABLED:
            _logger.info("[Consolidation] Disabled.")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            name="nightly_consolidation",
            daemon=True,
        )
        self._thread.start()
        _logger.info("[Consolidation] Scheduler started. runs at %02d:00 daily.", _HOUR)

    def stop(self) -> None:
        self._stop.set()

    def run_now(self) -> ConsolidationResult:
        with self._lock:
            if self._running:
                _logger.info("[Consolidation] Already running.")
                return ConsolidationResult()
            self._running = True
        try:
            return self._run_consolidation()
        finally:
            with self._lock:
                self._running = False

    # -------------------------------------------------------------------------
    # Scheduler
    # -------------------------------------------------------------------------

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            now  = datetime.now(tz=timezone.utc)
            date = now.strftime("%Y-%m-%d")
            if now.hour == _HOUR and date != self._last_run_date:
                with self._lock:
                    if self._running:
                        self._stop.wait(timeout=_POLL_SECS)
                        continue
                    self._running = True
                try:
                    result = self._run_consolidation()
                    self._last_run_date = date
                    if self._on_complete:
                        try:
                            self._on_complete(result)
                        except Exception:
                            pass
                except Exception as e:
                    _logger.error("[Consolidation] Error during scheduled run: %s", e)
                finally:
                    with self._lock:
                        self._running = False
            self._stop.wait(timeout=_POLL_SECS)

    # -------------------------------------------------------------------------
    # Consolidation steps
    # -------------------------------------------------------------------------

    def _run_consolidation(self) -> ConsolidationResult:
        result = ConsolidationResult()
        _logger.info("[Consolidation] Starting consolidation pass.")

        # Step 1: Episodic → semantic synthesis
        self._step_episodic_synthesis(result)

        # Step 2: DPO dataset generation
        self._step_dpo_generation(result)

        # Step 3: GRPO training pass (if enabled)
        self._step_grpo_pass(result)

        # Step 4: Knowledge vault pruning
        self._step_vault_pruning(result)

        # Step 5: Grounding trainer batch flush (Blueprint §9.5 — UI-AGILE)
        # Writes accumulated click-level grounding data to JSONL batches for
        # offline vision model fine-tuning. This is the nightly integration
        # point for the UI-AGILE continuous grounding reward loop.
        self._step_grounding_flush(result)

        result.finished_at = time.time()
        _logger.info(
            "[Consolidation] Done in %.1fs. dpo_pairs=%d grpo=%s pruned=%d grounding_flushed=%s errors=%d",
            result.duration_s, result.dpo_pairs_found,
            bool(result.grpo_dataset_path), result.memories_pruned,
            getattr(result, "grounding_batch_path", None),
            len(result.errors),
        )
        return result

    def _step_episodic_synthesis(self, result: ConsolidationResult) -> None:
        if self._synthesizer is None:
            return
        try:
            self._synthesizer.synthesize_all()
            _logger.debug("[Consolidation] Episodic synthesis complete.")
        except Exception as e:
            result.errors.append(f"episodic_synthesis: {e}")
            _logger.warning("[Consolidation] Episodic synthesis failed: %s", e)

    def _step_dpo_generation(self, result: ConsolidationResult) -> None:
        try:
            from core.learning.agent_q import get_agent_q
            from core.learning.preference_generator import get_preference_generator

            aq    = get_agent_q()
            stats = aq.get_stats()
            pairs = stats.get("pairs_on_disk", 0)
            result.dpo_pairs_found = pairs

            if pairs < _MIN_PAIRS:
                _logger.info("[Consolidation] Only %d pairs on disk (need %d) — skipping DPO.", pairs, _MIN_PAIRS)
                return

            pg   = get_preference_generator()
            path = pg.generate_dataset(source=aq)
            result.dpo_dataset_path = path
            _logger.info("[Consolidation] DPO dataset: %s", path)

        except Exception as e:
            result.errors.append(f"dpo_generation: {e}")
            _logger.warning("[Consolidation] DPO generation failed: %s", e)

    def _step_grpo_pass(self, result: ConsolidationResult) -> None:
        try:
            from core.learning.grpo_trainer import get_grpo_trainer
            trainer = get_grpo_trainer()

            if not result.dpo_dataset_path:
                return

            import json as _json
            tasks: List[Dict[str, Any]] = []
            with open(result.dpo_dataset_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = _json.loads(line)
                        tasks.append({"prompt": rec.get("prompt", ""), "task_id": rec.get("pair_id", "")})
                    except Exception:
                        pass

            if not tasks:
                return

            path = trainer.run_training_pass(tasks)
            result.grpo_dataset_path = path

        except Exception as e:
            result.errors.append(f"grpo_pass: {e}")
            _logger.warning("[Consolidation] GRPO pass failed: %s", e)

    def _step_vault_pruning(self, result: ConsolidationResult) -> None:
        if self._vault is None:
            return
        try:
            pruned = 0
            if hasattr(self._vault, "prune_stale"):
                pruned = self._vault.prune_stale(max_age_days=30)
            result.memories_pruned = pruned
            _logger.debug("[Consolidation] Vault pruning: %d entries removed.", pruned)
        except Exception as e:
            result.errors.append(f"vault_pruning: {e}")
            _logger.warning("[Consolidation] Vault pruning failed: %s", e)

    def _step_grounding_flush(self, result: ConsolidationResult) -> None:
        """
        Step 5: Flush accumulated grounding trainer data (Blueprint §9.5).
        Writes UI-AGILE training batches so the vision model can be fine-tuned
        on grounding errors discovered during the previous day's task execution.
        """
        try:
            from core.learning.grounding_trainer import get_global_grounding_trainer
            gt = get_global_grounding_trainer()
            stats = gt.get_stats()
            pending = stats.get("pending_steps", 0)
            if pending == 0:
                _logger.debug("[Consolidation] Grounding trainer: no pending steps to flush.")
                return
            batch_path = gt.flush_batch()
            if batch_path:
                result.grounding_batch_path = batch_path  # type: ignore[attr-defined]
                _logger.info(
                    "[Consolidation] Grounding batch flushed: %d steps → %s",
                    pending, batch_path,
                )
        except Exception as e:
            result.errors.append(f"grounding_flush: {e}")
            _logger.warning("[Consolidation] Grounding flush failed: %s", e)


_instance: Optional[NightlyConsolidator] = None
_instance_lock = threading.Lock()


def get_consolidator(**kwargs) -> NightlyConsolidator:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NightlyConsolidator(**kwargs)
    return _instance
