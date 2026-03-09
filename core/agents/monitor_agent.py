"""
core/agents/monitor_agent.py
==============================
Monitor Agent — Lightweight Background Event Watcher.

Blueprint §15.4 — Multi-Agent Orchestration

Role: Monitor Agent (lightweight, background, parallel)
    - Polls AT-SPI + process output continuously
    - Watches for unexpected dialogs, error messages, prompts
    - Interrupts Primary Agent when attention required (asyncio.Event)
    - Model: Claude Haiku (cheap, fast) or local 7B

Why single-agent is a ceiling (Blueprint §15.1):
    During a long-running command (npm install, pip install, Blender render),
    the agent must EITHER block (miss UI events) OR poll at 1Hz (current).
    Monitor Agent watches in parallel → Primary Agent can block without missing events.

Architecture:
    MonitorAgent runs as a background thread, continuously:
    1. Polling AT-SPI for new window/dialog events
    2. Checking subprocess output for error patterns
    3. Evaluating any new unexpected UI elements
    4. Setting asyncio.Event to interrupt Primary Agent when needed

Integration:
    - gii_loop.py → MonitorAgent.start() at task begin; stop() at end
    - gii_controller.py → interrupt_event.wait() during long operations
    - atspi_bridge.py → MonitorAgent.on_atspi_event() callback
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────

_POLL_INTERVAL_SECONDS  = float(os.environ.get("PROJECTZEO_MONITOR_POLL", "0.5"))
_MAX_EVENT_QUEUE        = int(os.environ.get("PROJECTZEO_MONITOR_QUEUE", "50"))
_DIALOG_TIMEOUT_SECONDS = float(os.environ.get("PROJECTZEO_MONITOR_DIALOG_TIMEOUT", "30.0"))

# Patterns that suggest a critical dialog appeared
_CRITICAL_DIALOG_PATTERNS: frozenset = frozenset({
    r"error",
    r"warning",
    r"permission",
    r"confirm",
    r"delete",
    r"remove",
    r"uninstall",
    r"restart",
    r"reboot",
    r"failed",
    r"cannot",
    r"unable",
    r"denied",
    r"authentication",
    r"password",
    r"sudo",
    r"admin",
    r"overwrite",
    r"replace",
    r"are you sure",
    r"yes.*no",
    r"ok.*cancel",
})

_PROGRESS_PATTERNS: frozenset = frozenset({
    r"\d+%",
    r"downloading",
    r"installing",
    r"compiling",
    r"building",
    r"loading",
    r"please wait",
    r"in progress",
})


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

class MonitorEventType(str, Enum):
    UNEXPECTED_DIALOG     = "unexpected_dialog"
    ERROR_DETECTED        = "error_detected"
    PROCESS_COMPLETE      = "process_complete"
    PERMISSION_PROMPT     = "permission_prompt"
    WINDOW_APPEARED       = "window_appeared"
    WINDOW_CLOSED         = "window_closed"
    PROGRESS_UPDATE       = "progress_update"
    FOCUS_CHANGED         = "focus_changed"
    ATSPI_EVENT           = "atspi_event"


@dataclass
class MonitorEvent:
    """An event detected by the Monitor Agent."""
    event_type:     MonitorEventType
    description:    str
    severity:       str          # "info" | "warn" | "critical"
    source:         str          # "atspi" | "process" | "vision" | "heuristic"
    raw_data:       Dict[str, Any] = field(default_factory=dict)
    detected_at:    float = field(default_factory=time.time)
    requires_interrupt: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type":   self.event_type.value,
            "description":  self.description[:300],
            "severity":     self.severity,
            "source":       self.source,
            "detected_at":  self.detected_at,
            "requires_interrupt": self.requires_interrupt,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MonitorAgent
# ─────────────────────────────────────────────────────────────────────────────

class MonitorAgent:
    """
    Lightweight background agent that watches for unexpected events.

    The primary agent can call await interrupt_event.wait() inside long
    operations. MonitorAgent sets this event when something unexpected happens.

    Usage:
        monitor = MonitorAgent()
        monitor.start()

        # In primary agent long-running operation:
        async def run_long_command():
            proc = await asyncio.create_subprocess_shell(cmd)
            monitor_task = asyncio.create_task(monitor.interrupt_event.wait())
            proc_task = asyncio.create_task(proc.wait())
            done, _ = await asyncio.wait(
                [monitor_task, proc_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if monitor_task in done:
                # Unexpected event — handle it
                events = monitor.drain_events()
                ...

        monitor.stop()
    """

    def __init__(
        self,
        *,
        atspi_bridge: Optional[Any] = None,
        vision_fn: Optional[Callable] = None,
        on_event_callback: Optional[Callable[[MonitorEvent], None]] = None,
        poll_interval: float = _POLL_INTERVAL_SECONDS,
    ) -> None:
        self._atspi_bridge = atspi_bridge
        self._vision_fn    = vision_fn
        self._on_event     = on_event_callback
        self._poll_interval = poll_interval

        self._events: List[MonitorEvent] = []
        self._events_lock = threading.Lock()
        self._known_windows: Set[str] = set()

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Cross-thread interrupt signal (can be polled by async code)
        self._interrupt_flag = threading.Event()

        # Process output monitoring
        self._watched_processes: Dict[int, Any] = {}  # pid → process

        _logger.info("[MonitorAgent] Initialised. poll_interval=%.1fs", poll_interval)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._interrupt_flag.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="monitor_agent",
        )
        self._thread.start()
        _logger.info("[MonitorAgent] Started.")

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        _logger.info("[MonitorAgent] Stopped.")

    @property
    def interrupt_event(self) -> threading.Event:
        """Threading event set when a critical event requires primary agent attention."""
        return self._interrupt_flag

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag after primary agent has handled the event."""
        self._interrupt_flag.clear()

    # =========================================================================
    # Event interface
    # =========================================================================

    def on_atspi_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Callback for AT-SPI events (Blueprint §5.3).
        Called by atspi_bridge.py when UI state changes.
        """
        event = self._classify_atspi_event(event_type, event_data)
        if event:
            self._emit_event(event)

    def watch_process(self, pid: int, process: Any) -> None:
        """Register a subprocess for output monitoring."""
        with self._events_lock:
            self._watched_processes[pid] = process

    def unwatch_process(self, pid: int) -> None:
        """Remove a subprocess from monitoring."""
        with self._events_lock:
            self._watched_processes.pop(pid, None)

    def drain_events(self, clear: bool = True) -> List[MonitorEvent]:
        """Return all buffered events, optionally clearing the buffer."""
        with self._events_lock:
            events = list(self._events)
            if clear:
                self._events.clear()
        return events

    def get_critical_events(self) -> List[MonitorEvent]:
        """Return only critical/warn events without clearing."""
        with self._events_lock:
            return [e for e in self._events if e.severity in ("critical", "warn")]

    def update_world_context(self, world_snapshot: Dict[str, Any]) -> None:
        """
        Update context from current world state — detects new windows/dialogs.
        Called by gii_loop.py after each observation cycle.
        """
        entities = world_snapshot.get("entities", []) or []
        current_windows: Set[str] = set()

        for ent in entities:
            if isinstance(ent, dict):
                role  = str(ent.get("role") or ent.get("type") or "").lower()
                label = str(ent.get("label") or ent.get("name") or "").lower()
                if role in ("window", "dialog", "frame", "alert"):
                    current_windows.add(f"{role}:{label}")

        # Detect new windows
        new_windows = current_windows - self._known_windows
        closed_windows = self._known_windows - current_windows

        for w in new_windows:
            role, _, label = w.partition(":")
            severity = self._assess_window_severity(label)
            event = MonitorEvent(
                event_type=MonitorEventType.WINDOW_APPEARED,
                description=f"New {role} appeared: {label[:100]}",
                severity=severity,
                source="heuristic",
                raw_data={"window_key": w},
                requires_interrupt=(severity in ("critical", "warn")),
            )
            self._emit_event(event)

        for w in closed_windows:
            self._emit_event(MonitorEvent(
                event_type=MonitorEventType.WINDOW_CLOSED,
                description=f"Window closed: {w}",
                severity="info",
                source="heuristic",
            ))

        self._known_windows = current_windows

    def summarize_for_primary_agent(self) -> str:
        """
        Format recent events as a text block to inject into primary agent context.
        """
        events = self.get_critical_events()
        if not events:
            return ""
        lines = ["=== MONITOR AGENT ALERTS ==="]
        for e in events[-5:]:
            age = time.time() - e.detected_at
            lines.append(
                f"[{e.severity.upper()}] {e.event_type.value}: {e.description} ({age:.0f}s ago)"
            )
        lines.append("=== END ALERTS ===")
        return "\n".join(lines)

    # =========================================================================
    # Private — monitor loop
    # =========================================================================

    def _monitor_loop(self) -> None:
        """Main background monitoring loop."""
        _logger.debug("[MonitorAgent] Monitor loop started.")
        while not self._stop_event.is_set():
            try:
                self._check_processes()
            except Exception as exc:
                _logger.debug("[MonitorAgent] Monitor iteration error: %s", exc)
            self._stop_event.wait(timeout=self._poll_interval)
        _logger.debug("[MonitorAgent] Monitor loop stopped.")

    def _check_processes(self) -> None:
        """Check watched processes for completion or error output."""
        import subprocess
        with self._events_lock:
            pids = list(self._watched_processes.keys())

        for pid in pids:
            try:
                import psutil
                if not psutil.pid_exists(pid):
                    self._emit_event(MonitorEvent(
                        event_type=MonitorEventType.PROCESS_COMPLETE,
                        description=f"Watched process {pid} completed",
                        severity="info",
                        source="process",
                        requires_interrupt=True,
                    ))
                    with self._events_lock:
                        self._watched_processes.pop(pid, None)
            except ImportError:
                pass
            except Exception:
                pass

    # =========================================================================
    # Private — event classification
    # =========================================================================

    def _classify_atspi_event(
        self, event_type: str, event_data: Dict[str, Any]
    ) -> Optional[MonitorEvent]:
        """
        Classify an AT-SPI event into a MonitorEvent.
        Blueprint §5.3 AT-SPI event table.
        """
        text = str(event_data.get("title") or event_data.get("name") or "").lower()

        if "state-changed:showing" in event_type or "children-changed:add" in event_type:
            severity = self._assess_window_severity(text)
            return MonitorEvent(
                event_type=MonitorEventType.WINDOW_APPEARED,
                description=f"AT-SPI new element: {text[:100]}",
                severity=severity,
                source="atspi",
                raw_data=event_data,
                requires_interrupt=(severity == "critical"),
            )

        if "window:activate" in event_type:
            return MonitorEvent(
                event_type=MonitorEventType.FOCUS_CHANGED,
                description=f"Window focus: {text[:80]}",
                severity="info",
                source="atspi",
                raw_data=event_data,
            )

        return None

    def _assess_window_severity(self, text: str) -> str:
        """Classify a window/dialog label as info/warn/critical."""
        text_lower = text.lower()
        for pattern in _CRITICAL_DIALOG_PATTERNS:
            if re.search(pattern, text_lower):
                return "critical" if any(
                    w in text_lower for w in ["delete", "remove", "wipe", "format", "uninstall", "destroy"]
                ) else "warn"
        return "info"

    def _emit_event(self, event: MonitorEvent) -> None:
        """Add event to buffer and potentially trigger interrupt."""
        with self._events_lock:
            self._events.append(event)
            # Keep bounded
            if len(self._events) > _MAX_EVENT_QUEUE:
                self._events = self._events[-_MAX_EVENT_QUEUE:]

        if event.requires_interrupt or event.severity == "critical":
            self._interrupt_flag.set()
            _logger.info(
                "[MonitorAgent] INTERRUPT: %s — %s",
                event.event_type.value, event.description[:100],
            )
        elif event.severity == "warn":
            _logger.info(
                "[MonitorAgent] WARN: %s — %s",
                event.event_type.value, event.description[:100],
            )

        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass
