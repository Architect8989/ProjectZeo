from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_BACKEND = os.environ.get("PROJECTZEO_SANDBOX_BACKEND", "auto").lower().strip()
_TIMEOUT = int(os.environ.get("PROJECTZEO_SANDBOX_TIMEOUT", "60"))
_MEM_MB  = int(os.environ.get("PROJECTZEO_SANDBOX_MEM_MB", "512"))
_IMAGE   = os.environ.get("PROJECTZEO_SANDBOX_IMAGE", "ubuntu:22.04")


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
                    sandbox_id=sandbox_id,
                    backend="docker",
                    exit_code=code,
                    stdout=logs[:8000],
                    stderr="",
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

    Selects the strongest available backend and provides a uniform
    run_rollout() interface for GRPOTrainer.
    """

    def __init__(self) -> None:
        self._backend = _RESOLVED_BACKEND
        self._lock    = threading.Lock()
        self._runs    = 0
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

    def health_check(self) -> bool:
        result = self.run_rollout("echo ok", timeout=10)
        return result.success and "ok" in result.stdout

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend":     self._backend,
            "total_runs":  self._runs,
            "timeout_s":   _TIMEOUT,
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
