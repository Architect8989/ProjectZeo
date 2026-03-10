from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set

_logger = logging.getLogger(__name__)

_SCAFFOLD_PATH_PATTERNS: List[str] = [
    "scaffold", "execution_plan", "plan.json", "milestones.json",
    "steps.json", "task_plan", "objective.json", "gii_plan",
    "authority_policy", "constitution.md", "authority_constitution",
]

_WRITE_OPS: FrozenSet[str] = frozenset({
    "write", "file_create", "file_write", "file_modify", "command",
})

_OVERWRITE_COMMAND_PATTERNS: List[re.Pattern] = [
    re.compile(r"\btee\s+.*(?:scaffold|plan|milestone|objective)", re.IGNORECASE),
    re.compile(r"\becho\s+.*>\s*.*(?:scaffold|plan|milestone|objective)", re.IGNORECASE),
    re.compile(r"\bcp\s+.*(?:scaffold|plan|milestone|objective)", re.IGNORECASE),
    re.compile(r"\bsed\s+-i.*(?:scaffold|plan|milestone|objective)", re.IGNORECASE),
]

@dataclass
class AuditResult:
    decision: str
    reason: str = ""
    action_key: str = ""

class ScaffoldAudit:

    def __init__(
        self,
        scaffold_paths: Optional[List[str]] = None,
        *,
        journal=None,
        extra_protected_dirs: Optional[List[str]] = None,
    ) -> None:
        self._scaffold_paths: Set[str] = set()
        self._scaffold_hashes: Dict[str, str] = {}
        self._journal = journal
        self._armed: bool = False
        self._lock = threading.Lock()
        self._block_count: int = 0

        for p in (scaffold_paths or []):
            self.register_scaffold_path(p)

        _pz_home = os.path.expanduser("~/.projectzeo")
        _protected_dirs: List[str] = [_pz_home] + (extra_protected_dirs or [])
        self._protected_dirs: List[str] = [
            os.path.abspath(d) for d in _protected_dirs if d
        ]

    def register_scaffold_path(self, path: str) -> None:
        abs_path = os.path.abspath(path)
        with self._lock:
            self._scaffold_paths.add(abs_path)
            try:
                if os.path.isfile(abs_path):
                    self._scaffold_hashes[abs_path] = self._hash_file(abs_path)
            except OSError:
                pass
        _logger.debug("[ScaffoldAudit] Registered scaffold path: %s", abs_path)

    def arm(self) -> None:
        with self._lock:
            self._armed = True
        _logger.info("[ScaffoldAudit] ARMED — scaffold modification audit active.")

    def disarm(self) -> None:
        with self._lock:
            self._armed = False
        _logger.info(
            "[ScaffoldAudit] DISARMED — audit complete. Total blocks: %d.", self._block_count
        )

    def check_action(self, action: Dict[str, Any]) -> AuditResult:
        with self._lock:
            armed = self._armed
        if not armed:
            return AuditResult(decision="ALLOW")

        op = str(action.get("operation", "")).lower()
        if op not in _WRITE_OPS:
            return AuditResult(decision="ALLOW")

        target_path = action.get("path") or action.get("file") or ""
        if target_path:
            abs_target = os.path.abspath(str(target_path))
            with self._lock:
                if abs_target in self._scaffold_paths:
                    return self._block(action, f"Direct scaffold file write: {abs_target}")
            target_lower = str(target_path).lower()
            for pattern in _SCAFFOLD_PATH_PATTERNS:
                if pattern in target_lower:
                    return self._block(action, f"Scaffold-like path write: {target_path}")
            for pdir in self._protected_dirs:
                if abs_target.startswith(pdir):
                    return self._block(action, f"Write to protected dir: {pdir}")

        if op == "command":
            cmd = str(action.get("command", ""))
            for pattern in _OVERWRITE_COMMAND_PATTERNS:
                if pattern.search(cmd):
                    return self._block(action, f"Command matches scaffold-overwrite pattern: {cmd[:100]}")

        content = str(action.get("content") or action.get("text") or "")
        if content and len(content) > 50:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            with self._lock:
                for path, known_hash in self._scaffold_hashes.items():
                    if content_hash == known_hash:
                        return self._block(
                            action,
                            f"Action content matches scaffold hash for: {path}"
                        )

        return AuditResult(decision="ALLOW")

    def verify_scaffold_integrity(self) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        with self._lock:
            snapshot = dict(self._scaffold_hashes)
        for path, original_hash in snapshot.items():
            try:
                current_hash = self._hash_file(path)
                ok = current_hash == original_hash
                if not ok:
                    _logger.critical(
                        "[ScaffoldAudit] INTEGRITY VIOLATION: %s has been modified!", path
                    )
                    if self._journal:
                        try:
                            self._journal.record({
                                "event": "scaffold_integrity_violation",
                                "path": path,
                                "original_hash": original_hash[:16],
                                "current_hash": current_hash[:16],
                            })
                        except Exception:
                            pass
                results[path] = ok
            except OSError:
                results[path] = False
        return results

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "armed": self._armed,
                "scaffold_paths_registered": len(self._scaffold_paths),
                "block_count": self._block_count,
                "protected_dirs": self._protected_dirs,
            }

    def _block(self, action: Dict[str, Any], reason: str) -> AuditResult:
        with self._lock:
            self._block_count += 1
        _logger.critical(
            "[ScaffoldAudit] BLOCK: scaffold modification attempt detected! "
            "op=%s reason=%s",
            action.get("operation"), reason,
        )
        if self._journal:
            try:
                self._journal.record({
                    "event": "scaffold_modification_blocked",
                    "operation": action.get("operation"),
                    "reason": reason,
                    "action_snippet": {
                        k: str(v)[:100] for k, v in action.items()
                        if k in ("operation", "path", "command", "content")
                    },
                })
            except Exception:
                pass
        return AuditResult(
            decision="BLOCK",
            reason=f"[ScaffoldAudit] {reason}",
        )

    @staticmethod
    def _hash_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

def build_scaffold_audit_from_plan(
    plan,
    journal=None,
) -> ScaffoldAudit:
    paths: List[str] = []
    plan_path = getattr(plan, "_source_path", None) or getattr(plan, "path", None)
    if plan_path and isinstance(plan_path, str):
        paths.append(plan_path)
    _config_dir = os.path.expanduser("~/.projectzeo")
    for fname in ("restore_ledger.json", "reflexion.db", "semantic.db", "chunks.json"):
        paths.append(os.path.join(_config_dir, fname))
    return ScaffoldAudit(scaffold_paths=paths, journal=journal)
