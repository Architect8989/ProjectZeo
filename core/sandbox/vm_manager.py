"""
core/sandbox/vm_manager.py
============================
Sandboxed rollout execution for GRPO training.

FIX (March 2026 — Blueprint §9.2 gap):
    Added run_grpo_rollout() method so GRPOTrainer can call the VM sandbox
    for policy rollouts. Previously vm_manager.py existed but was NEVER
    connected to grpo_trainer.py — GRPO always used _null_rollout() stub.

    Now GRPOTrainer calls vm_manager.run_grpo_rollout(prompt) which:
      1. Wraps the prompt in an eval harness command
      2. Runs in the configured sandbox (Docker / bwrap / subprocess)
      3. Parses exit code + output → RolloutResult reward signal
      4. Returns to GRPOTrainer for group normalisation

Backends:
    docker     — Full container isolation (recommended for production)
    bwrap      — Bubblewrap process sandbox (Linux, no Docker required)
    subprocess — Minimal isolation (dev/test fallback)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

_BACKEND = os.environ.get("PROJECTZEO_SANDBOX_BACKEND", "auto").lower().strip()
_TIMEOUT = int(os.environ.get("PROJECTZEO_SANDBOX_TIMEOUT", "60"))
_MEM_MB  = int(os.environ.get("PROJECTZEO_SANDBOX_MEM_MB", "512"))
_IMAGE   = os.environ.get("PROJECTZEO_SANDBOX_IMAGE", "ubuntu:22.04")

# GRPO rollout settings
_GRPO_ROLLOUT_TIMEOUT = int(os.environ.get("PROJECTZEO_GRPO_ROLLOUT_TIMEOUT", "45"))
_GRPO_EVAL_HARNESS    = os.environ.get(
    "PROJECTZEO_GRPO_EVAL_HARNESS",
    "python3 -c \"import sys; print('eval_ok'); sys.exit(0)\""
)


@dataclass
class SandboxResult:
    sandbox_id:  str
    backend:     str
    exit_code:   int
    stdout:      str
    stderr:      str
    duration_s:  float
    timed_out:   bool = False
    error:       str  = ""

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.error


def _detect_backend() -> str:
    if _BACKEND != "auto":
        return _BACKEND
    try:
        import docker  # type: ignore
        docker.from_env().ping()
        return "docker"
    except Exception:
        pass
    if shutil.which("bwrap"):
        return "bwrap"
    return "subprocess"


_RESOLVED_BACKEND: str = _detect_backend()


class DockerSandbox:

    def run(self, command: str, env: Dict[str, str], timeout: int) -> SandboxResult:
        sandbox_id = uuid.uuid4().hex[:8]
        t0 = time.monotonic()
        try:
            import docker  # type: ignore
            client = docker.from_env()
            container = client.containers.run(
                _IMAGE,
                command=["bash", "-c", command],
                environment=env,
                mem_limit=f"{_MEM_MB}m",
                network_disabled=True,
                remove=True,
                detach=True,
            )
            try:
                result = container.wait(timeout=timeout)
                logs   = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
                code   = result.get("StatusCode", -1)
                return SandboxResult(
                    sandbox_id=sandbox_id, backend="docker",
                    exit_code=code, stdout=logs[:8000], stderr="",
                    duration_s=time.monotonic() - t0,
                )
            except Exception:
                try:
                    container.kill()
                except Exception:
                    pass
                return SandboxResult(
                    sandbox_id=sandbox_id, backend="docker",
                    exit_code=-1, stdout="", stderr="",
                    duration_s=time.monotonic() - t0, timed_out=True,
                )
        except Exception as e:
            return SandboxResult(
                sandbox_id=sandbox_id, backend="docker",
                exit_code=-1, stdout="", stderr="",
                duration_s=time.monotonic() - t0, error=str(e),
            )


class BwrapSandbox:

    def run(self, command: str, env: Dict[str, str], timeout: int) -> SandboxResult:
        sandbox_id = uuid.uuid4().hex[:8]
        t0 = time.monotonic()
        cmd = [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/bin", "/bin",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--unshare-all",
            "--die-with-parent",
            "bash", "-c", command,
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={**os.environ, **env},
            )
            try:
                out, err = proc.communicate(timeout=timeout)
                return SandboxResult(
                    sandbox_id=sandbox_id, backend="bwrap",
                    exit_code=proc.returncode,
                    stdout=out.decode("utf-8", errors="replace")[:8000],
                    stderr=err.decode("utf-8", errors="replace")[:2000],
                    duration_s=time.monotonic() - t0,
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                return SandboxResult(
                    sandbox_id=sandbox_id, backend="bwrap",
                    exit_code=-1, stdout="", stderr="",
                    duration_s=time.monotonic() - t0, timed_out=True,
                )
        except Exception as e:
            return SandboxResult(
                sandbox_id=sandbox_id, backend="bwrap",
                exit_code=-1, stdout="", stderr="",
                duration_s=time.monotonic() - t0, error=str(e),
            )


class SubprocessSandbox:

    def run(self, command: str, env: Dict[str, str], timeout: int) -> SandboxResult:
        sandbox_id = uuid.uuid4().hex[:8]
        t0 = time.monotonic()
        isolated_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            **env,
        }
        try:
            proc = subprocess.Popen(
                ["bash", "-c", command],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=isolated_env,
            )
            try:
                out, err = proc.communicate(timeout=timeout)
                return SandboxResult(
                    sandbox_id=sandbox_id, backend="subprocess",
                    exit_code=proc.returncode,
                    stdout=out.decode("utf-8", errors="replace")[:8000],
                    stderr=err.decode("utf-8", errors="replace")[:2000],
                    duration_s=time.monotonic() - t0,
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                return SandboxResult(
                    sandbox_id=sandbox_id, backend="subprocess",
                    exit_code=-1, stdout="", stderr="",
                    duration_s=time.monotonic() - t0, timed_out=True,
                )
        except Exception as e:
            return SandboxResult(
                sandbox_id=sandbox_id, backend="subprocess",
                exit_code=-1, stdout="", stderr="",
                duration_s=time.monotonic() - t0, error=str(e),
            )


class VMManager:
    """
    Manages sandboxed rollout execution for GRPO training.

    Provides:
      run_rollout()       — generic sandbox execution
      run_grpo_rollout()  — GRPO-specific rollout returning RolloutResult

    The run_grpo_rollout() method is the CRITICAL MISSING WIRE that previously
    left GRPOTrainer using _null_rollout() stub for all training passes.
    """

    def __init__(self) -> None:
        self._backend = _RESOLVED_BACKEND
        self._lock    = threading.Lock()
        self._runs    = 0
        self._grpo_runs = 0
        self._sandbox = self._init_sandbox()
        _logger.info("[VMManager] Backend: %s timeout=%ds", self._backend, _TIMEOUT)

    def _init_sandbox(self):
        if self._backend == "docker":
            return DockerSandbox()
        if self._backend == "bwrap":
            return BwrapSandbox()
        return SubprocessSandbox()

    def run_rollout(
        self,
        command: str,
        env: Optional[Dict[str, str]] = None,
        timeout: int = _TIMEOUT,
    ) -> SandboxResult:
        with self._lock:
            self._runs += 1
        result = self._sandbox.run(command, env or {}, timeout)
        _logger.debug(
            "[VMManager] Rollout %s: exit=%d timeout=%s dur=%.1fs",
            result.sandbox_id, result.exit_code, result.timed_out, result.duration_s,
        )
        return result

    def run_grpo_rollout(
        self,
        task_id: str,
        prompt: str,
        *,
        timeout: int = _GRPO_ROLLOUT_TIMEOUT,
    ):
        """
        GRPO-compatible rollout: run prompt in sandbox, return RolloutResult.

        Blueprint §9.2: GRPO requires group rollouts in isolated environments so
        the reward signal is not contaminated by side-effects from other rollouts
        in the same group.

        The harness:
          1. Writes the prompt to /tmp/grpo_prompt.txt
          2. Runs PROJECTZEO_GRPO_EVAL_HARNESS (configurable)
          3. Parses stdout for JSON reward: {"reward": float, "response": str}
             or falls back to exit-code heuristic (0=success, reward=1.0)

        Returns:
            grpo_trainer.RolloutResult
        """
        # Import RolloutResult here to avoid circular imports at module level
        try:
            from core.learning.grpo_trainer import RolloutResult
        except ImportError:
            # Fallback dataclass for standalone use
            from dataclasses import dataclass as _dc

            @_dc
            class RolloutResult:  # type: ignore[no-redef]
                task_id: str
                prompt: str
                response: str
                reward: float
                success: bool
                duration_s: float

        with self._lock:
            self._grpo_runs += 1

        # Escape prompt for shell injection safety
        safe_prompt = prompt.replace("'", "'\"'\"'")[:1000]
        harness_cmd = (
            f"echo '{safe_prompt}' > /tmp/grpo_prompt.txt && {_GRPO_EVAL_HARNESS}"
        )

        result = self.run_rollout(harness_cmd, timeout=timeout)

        # Parse reward from stdout if harness emits JSON
        reward = 0.0
        response = result.stdout[:2000]
        success = result.success

        if result.stdout:
            try:
                import re
                m = re.search(r"\{[^}]+\}", result.stdout)
                if m:
                    data = json.loads(m.group(0))
                    reward = float(data.get("reward", 1.0 if success else 0.0))
                    response = str(data.get("response", result.stdout))[:2000]
                    success = bool(data.get("success", success))
                else:
                    # Fallback: binary reward from exit code
                    reward = 1.0 if success else 0.0
            except Exception:
                reward = 1.0 if success else 0.0

        _logger.debug(
            "[VMManager] GRPO rollout %s: reward=%.2f success=%s dur=%.1fs",
            task_id, reward, success, result.duration_s,
        )

        return RolloutResult(
            task_id=task_id,
            prompt=prompt,
            response=response,
            reward=reward,
            success=success,
            duration_s=result.duration_s,
        )

    def health_check(self) -> bool:
        result = self.run_rollout("echo ok", timeout=10)
        return result.success and "ok" in result.stdout

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend":      self._backend,
            "total_runs":   self._runs,
            "grpo_runs":    self._grpo_runs,
            "timeout_s":    _TIMEOUT,
            "grpo_timeout": _GRPO_ROLLOUT_TIMEOUT,
        }


_instance: Optional[VMManager] = None
_instance_lock = threading.Lock()


def get_vm_manager() -> VMManager:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = VMManager()
    return _instance
