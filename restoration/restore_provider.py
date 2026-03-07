from __future__ import annotations

import time
import threading
import json
import os
import subprocess
import sys
from typing import Optional

from restoration.snapshot_types import (
    RestorationSnapshot,
    levenshtein_distance,
    title_match as _title_match_shared,
)
from restoration.snapshot_provider import SnapshotProvider
from core.mode_controller import ModeController, SystemMode




_PLAYWRIGHT_AVAILABLE: bool = False
try:
    from playwright.sync_api import sync_playwright as _sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _sync_playwright = None  # type: ignore[assignment]

class RestorationError(RuntimeError):
    """Raised when restoration or post-restore verification fails."""
    pass


class RestoreProvider:
   

    CURSOR_TOLERANCE_PX = 5

    
    POST_ACTION_DELAY: float = 0.50     # was 0.25s
    MAX_VERIFY_ATTEMPTS: int = 10       # was 5 → total window: 10 × 0.50 = 5.0s

    MAX_LEDGER_ENTRIES = 10_000
    
    MAX_TITLE_DISTANCE = 5

    _RESTORE_LEDGER_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "memory",
        "restore_ledger.json",
    )

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        os_backend,
        mode_controller: ModeController,
        snapshot_provider: SnapshotProvider,
        authority_state=None,  # HIGH-7 FIX: wired from main.py so duplicate-restore warning flag is set
    ) -> None:
        self._os = os_backend
        self._mode = mode_controller
        self._snapshot_provider = snapshot_provider
        self._lock = threading.Lock()
        
        self._authority_state = authority_state

        # Ledger availability — disabled on read-only filesystems without raising
        self._ledger_available: bool = True
        try:
            os.makedirs(os.path.dirname(self._RESTORE_LEDGER_PATH), exist_ok=True)
        except (PermissionError, OSError) as _ledger_dir_err:
            self._ledger_available = False
            print(
                f"[RestoreProvider] WARNING: Cannot create ledger directory "
                f"({os.path.dirname(self._RESTORE_LEDGER_PATH)!r}): "
                f"{_ledger_dir_err}. "
                "Duplicate-restore protection is DISABLED for this session. "
                "This is expected on read-only filesystems (containers, NFS, CI).",
                file=sys.stderr,
            )

        self._completed_snapshots: dict = (
            self._load_ledger() if self._ledger_available else {}
        )

        # HIGH-3 FIX: Browser session state (captured at snapshot time, restored on restore)
        self._browser_session: dict = {}  # {url, scroll_x, scroll_y, tab_index, ...}
        self._browser_capture_lock = threading.Lock()

    # =========================================================================
    # LEDGER
    # =========================================================================

    def _load_ledger(self) -> dict:
        
        if not os.path.exists(self._RESTORE_LEDGER_PATH):
            return {}

        try:
            with open(self._RESTORE_LEDGER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                # New format: {snapshot_id: float_timestamp}
                return {
                    str(k): float(v)
                    for k, v in list(data.items())[: self.MAX_LEDGER_ENTRIES]
                    if isinstance(v, (int, float))
                }

            if isinstance(data, list):
                # Old format: ["snapshot_id", ...] — upgrade in memory
                return {str(x): 0.0 for x in data[: self.MAX_LEDGER_ENTRIES]}

            raise RestorationError("Restore ledger corrupted: unexpected format")

        except RestorationError:
            raise
        except Exception as e:
            raise RestorationError(f"Restore ledger load failed: {e}") from e

    def _persist_ledger(self) -> None:
        
        if not self._ledger_available:
            return

        tmp_path = self._RESTORE_LEDGER_PATH + ".tmp"

        try:
            # SI-4 / H9: Evict oldest entries (lowest timestamps) to stay under cap
            if len(self._completed_snapshots) > self.MAX_LEDGER_ENTRIES:
                sorted_by_ts = sorted(
                    self._completed_snapshots.items(),
                    key=lambda kv: kv[1],           # sort ascending by timestamp
                )
                self._completed_snapshots = dict(
                    sorted_by_ts[-self.MAX_LEDGER_ENTRIES:]  # keep newest
                )

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._completed_snapshots,
                    f,
                    separators=(",", ":"),
                )
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, self._RESTORE_LEDGER_PATH)

        except Exception as e:
            # Non-fatal: ledger persistence failure logs but does not abort restoration
            print(
                f"[RestoreProvider] WARNING: Ledger persist failed: {e}. "
                "Duplicate-restore protection may be incomplete for this session.",
                file=sys.stderr,
            )

    

    def capture_browser_session(self) -> dict:
        
        if not _PLAYWRIGHT_AVAILABLE:
            return {"captured": False, "reason": "playwright not installed"}

        result = {
            "captured": False,
            "tabs": [],
            "active_tab_index": 0,
            "timestamp": time.time(),
        }

        try:
            with _sync_playwright() as pw:
                # Try to connect to an existing Chromium-based browser
                # Requires browser to be launched with --remote-debugging-port=9222
                _cdp_url = os.environ.get(
                    "PROJECTZEO_CDP_URL", "http://localhost:9222"
                ).strip()
                try:
                    browser = pw.chromium.connect_over_cdp(_cdp_url, timeout=3000)
                    contexts = browser.contexts
                    if not contexts:
                        return result

                    context = contexts[0]
                    pages = context.pages
                    if not pages:
                        return result

                    tabs = []
                    for page in pages:
                        try:
                            url = page.url
                            title = page.title()
                            scroll_y = page.evaluate("window.scrollY") if url.startswith("http") else 0
                            scroll_x = page.evaluate("window.scrollX") if url.startswith("http") else 0
                            tabs.append({
                                "url": url,
                                "title": title,
                                "scroll_x": scroll_x,
                                "scroll_y": scroll_y,
                            })
                        except Exception:
                            tabs.append({"url": "unknown", "title": "", "scroll_x": 0, "scroll_y": 0})

                    result["tabs"] = tabs
                    result["active_tab_index"] = 0  # first tab is active in CDP
                    result["captured"] = True
                    browser.close()

                except Exception as cdp_exc:
                    # CDP connection failed — browser may not be running with remote debugging
                    print(
                        f"[RestoreProvider] Browser session capture: CDP connect failed "
                        f"({cdp_exc!s:.80}). "
                        "To enable browser capture, launch browser with: "
                        "--remote-debugging-port=9222",
                        file=sys.stderr,
                    )
                    result["reason"] = f"CDP unavailable: {cdp_exc!s:.60}"

        except Exception as exc:
            result["reason"] = f"Playwright error: {exc!s:.80}"
            print(
                f"[RestoreProvider] Browser session capture failed: {exc!s:.80}",
                file=sys.stderr,
            )

        with self._browser_capture_lock:
            self._browser_session = dict(result)

        return result

    def restore_browser_session(self, session_state: dict) -> bool:
        
        if not _PLAYWRIGHT_AVAILABLE:
            return False

        if not session_state or not session_state.get("captured"):
            return False

        tabs = session_state.get("tabs", [])
        if not tabs:
            return True

        try:
            with _sync_playwright() as pw:
                _cdp_url = os.environ.get(
                    "PROJECTZEO_CDP_URL", "http://localhost:9222"
                ).strip()
                try:
                    browser = pw.chromium.connect_over_cdp(_cdp_url, timeout=3000)
                    contexts = browser.contexts
                    if not contexts:
                        return False

                    context = contexts[0]

                    # Close any tabs opened during the task
                    current_pages = context.pages
                    for page in current_pages[len(tabs):]:
                        try:
                            page.close()
                        except Exception:
                            pass

                    # Navigate existing/new tabs to snapshot URLs
                    for i, tab in enumerate(tabs):
                        url = tab.get("url", "")
                        if not url or url in ("about:blank", "about:newtab"):
                            continue
                        try:
                            if i < len(current_pages):
                                page = current_pages[i]
                            else:
                                page = context.new_page()

                            if page.url != url:
                                page.goto(url, timeout=10000, wait_until="domcontentloaded")

                            # Restore scroll position
                            sx = tab.get("scroll_x", 0)
                            sy = tab.get("scroll_y", 0)
                            if sx or sy:
                                page.evaluate(f"window.scrollTo({sx}, {sy})")

                        except Exception as page_exc:
                            print(
                                f"[RestoreProvider] Browser tab {i} restore failed: {page_exc!s:.80}",
                                file=sys.stderr,
                            )

                    browser.close()
                    print(
                        f"[RestoreProvider] Browser session restored: {len(tabs)} tab(s).",
                        file=sys.stderr,
                    )
                    return True

                except Exception as cdp_exc:
                    print(
                        f"[RestoreProvider] Browser restore: CDP connect failed: {cdp_exc!s:.80}",
                        file=sys.stderr,
                    )
                    return False

        except Exception as exc:
            print(
                f"[RestoreProvider] Browser restore failed: {exc!s:.80}",
                file=sys.stderr,
            )
            return False

    def restore_snapshot(self, snapshot_id: str) -> None:
        
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise RestorationError("Invalid snapshot_id: must be a non-empty string")

        snapshot = self._snapshot_provider.get_snapshot(snapshot_id)
        if snapshot is None:
            raise RestorationError(f"Snapshot not found in provider: {snapshot_id!r}")

        self.restore(snapshot)

    # =========================================================================
    # CORE RESTORE
    # =========================================================================

    def restore(self, snapshot: RestorationSnapshot) -> None:
        
        if not isinstance(snapshot, RestorationSnapshot):
            raise RestorationError(
                f"restore() requires RestorationSnapshot, got {type(snapshot).__name__}"
            )

        snapshot_id = snapshot.snapshot_id

        # --- LOCK: idempotency check only ---
        with self._lock:
            if snapshot_id in self._completed_snapshots:
                import sys as _sys
                print(
                    f"[RestoreProvider] WARNING BUG-06: Idempotency guard fired "
                    f"for snapshot_id={snapshot_id!r} — this snapshot was already "
                    "marked completed.  If this is a replan fallback, the workspace "
                    "may NOT have been restored to the pre-task state.  "
                    "Inspect the authority log and verify workspace manually.",
                    file=_sys.stderr,
                )
                _auth = getattr(self, "_authority_state", None)
                if _auth is not None and hasattr(_auth, "verification_warning"):
                    try:
                        _auth.verification_warning = True
                    except Exception:
                        pass
                return

        

        if self._mode.mode is not SystemMode.RESTORING:
            raise RestorationError(
                f"restore() attempted in mode {self._mode.mode!r}; "
                "only RESTORING mode is permitted (fail-closed)."
            )

        # Stop all automated input before manipulating the workspace
        try:
            self._os.stop_automated_input()
            self._os.force_release_all(reason="restoration")
            self._os.mark_automation_inactive()
        except Exception as e:
            raise RestorationError(
                f"Automation shutdown failed before restore: {e}"
            ) from e

        # Best-effort workspace restoration
        self._restore_application(snapshot)
        self._restore_window(snapshot)
        self._restore_cursor(snapshot)

        # HIGH-3 FIX: Restore browser session if captured at snapshot time
        _browser_state = snapshot.metadata.get("browser_session")
        if _browser_state and _browser_state.get("captured"):
            try:
                self.restore_browser_session(_browser_state)
            except Exception as _br_err:
                print(
                    f"[RestoreProvider] Browser session restore failed (non-fatal): {_br_err}",
                    file=sys.stderr,
                )

        # Hard verification — raises RestorationError on failure
        # AUDIT FIX: On failure, guarantee a last-resort cursor-only restore
        # and set RESTORATION_INCOMPLETE flag before re-raising.
        try:
            self._verify(snapshot)
        except RestorationError as _verify_err:
            self._handle_restore_failure(snapshot, str(_verify_err))
            raise

        self._report_unrestored_processes(snapshot)

        # --- LOCK: ledger write only ---
        with self._lock:
            
            if snapshot_id not in self._completed_snapshots:
                
                self._completed_snapshots[snapshot_id] = time.monotonic()
                self._persist_ledger()

    

    def _handle_restore_failure(
        self, snapshot: RestorationSnapshot, error_reason: str
    ) -> None:
        """
        AUDIT MEDIUM FIX: Guaranteed last-resort actions on any restore failure.

        On full restore failure or verification timeout:
        1. Attempt cursor-only restoration unconditionally (cursor at safe position).
        2. Write RESTORATION_INCOMPLETE flag to authority_state.
        3. Log prominently so operator is aware before accepting next task.
        """
        # Step 1: Last-resort cursor-only restoration
        try:
            self._os.set_cursor_position({"x": snapshot.cursor.x, "y": snapshot.cursor.y})
            print(
                f"[RestoreProvider] LAST-RESORT: cursor restored to "
                f"({snapshot.cursor.x}, {snapshot.cursor.y}) after full restore failure.",
                file=sys.stderr,
            )
        except Exception as cursor_err:
            # Absolute fallback: move cursor to screen center
            try:
                _w, _h = self._os.screen_size()
                self._os.set_cursor_position({"x": _w // 2, "y": _h // 2})
                print(
                    f"[RestoreProvider] LAST-RESORT: cursor moved to screen center "
                    f"({_w // 2}, {_h // 2}) — original position restore failed: {cursor_err}",
                    file=sys.stderr,
                )
            except Exception as center_err:
                print(
                    f"[RestoreProvider] LAST-RESORT cursor restore completely failed: {center_err}",
                    file=sys.stderr,
                )

        # Step 2: Set RESTORATION_INCOMPLETE flag on authority_state
        try:
            auth = getattr(self, "_authority_state", None)
            if auth is not None:
                if hasattr(auth, "restoration_incomplete"):
                    auth.restoration_incomplete = True
                elif hasattr(auth, "__dict__"):
                    auth.__dict__["restoration_incomplete"] = True
        except Exception:
            pass

        # Step 3: Write an incomplete-restoration marker to disk for operator
        try:
            import pathlib as _pl
            _project_root = _pl.Path(__file__).resolve().parent.parent
            _flag_path = _project_root / "temp" / "RESTORATION_INCOMPLETE"
            _flag_path.parent.mkdir(parents=True, exist_ok=True)
            _ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            _snap_id = getattr(snapshot, 'snapshot_id', 'unknown')
            _flag_text = (
                "RESTORATION INCOMPLETE\n"
                + f"Reason: {error_reason}\n"
                + f"Snapshot: {_snap_id}\n"
                + f"Time: {_ts}\n"
                + "Operator must acknowledge this file before the next task runs.\n"
            )
            _flag_path.write_text(_flag_text,
                encoding="utf-8",
            )
            print(
                f"[RestoreProvider] RESTORATION_INCOMPLETE flag written to: {_flag_path}. "
                "DELETE this file to acknowledge and allow the next task.",
                file=sys.stderr,
            )
        except Exception as flag_err:
            print(
                f"[RestoreProvider] Could not write RESTORATION_INCOMPLETE flag: {flag_err}",
                file=sys.stderr,
            )

    def _restore_application(self, snapshot: RestorationSnapshot) -> None:
        
        if snapshot.application.process_name == "__bare_desktop__":
            return  # Bare desktop: no application to activate

        try:
            self._os.activate_application(
                {"title": snapshot.application.process_name}
            )
        except OSError as _app_err:
            # GAP-02: Log the specific failure reason before continuing
            print(
                f"[RestoreProvider] WARNING: _restore_application() — "
                f"activate_application({snapshot.application.process_name!r}) "
                f"raised OSError: {_app_err}. "
                "Target application may have been closed during task execution. "
                "Continuing with best-effort restoration (cursor will still be restored).",
                file=sys.stderr,
            )
            # Best-effort: continue; _verify() will detect and report the mismatch

        time.sleep(self.POST_ACTION_DELAY)

    def _restore_window(self, snapshot: RestorationSnapshot) -> None:
        """
        AUDIT MEDIUM FIX: Log every title mismatch to journal before continuing.
        Previously, title mismatches were silently ignored — the system appeared
        to succeed even when window focus was wrong.
        """
        window_id = getattr(snapshot.focus, "window_id", None)
        if not isinstance(window_id, str) or not window_id.strip():
            return

        if window_id == "__bare_desktop__":
            return

        try:
            self._os.focus_window({"title": window_id})
        except OSError as _focus_err:
            print(
                f"[RestoreProvider] WARNING: _restore_window() — "
                f"focus_window({window_id!r}) raised OSError: {_focus_err}. "
                "Window may have been closed during task execution. Continuing.",
                file=sys.stderr,
            )

        time.sleep(self.POST_ACTION_DELAY)

        # AUDIT FIX: Verify focus and log mismatch for operator awareness
        try:
            current_window = self._os.get_focused_window()
            if isinstance(current_window, dict) and isinstance(current_window.get("title"), str):
                expected_norm = " ".join(window_id.lower().strip().split())
                actual_norm   = " ".join(current_window["title"].lower().strip().split())
                _dist = levenshtein_distance(expected_norm, actual_norm)
                if _dist > self.MAX_TITLE_DISTANCE:
                    # AUDIT FIX: Log title mismatch prominently
                    print(
                        f"[RestoreProvider] TITLE MISMATCH after _restore_window: "
                        f"expected={window_id!r} actual={current_window['title']!r} "
                        f"levenshtein={_dist} (threshold={self.MAX_TITLE_DISTANCE}). "
                        "Window focus may be incorrect. Operator should verify.",
                        file=sys.stderr,
                    )
                    # For critical applications require exact match
                    _CRITICAL_APP_PATTERNS = (
                        "firefox", "chromium", "chrome", "code", "code-oss",
                        "cursor", "vscode", "sublime", "atom",
                    )
                    expected_lower = window_id.lower()
                    if any(p in expected_lower for p in _CRITICAL_APP_PATTERNS):
                        print(
                            f"[RestoreProvider] CRITICAL APP MISMATCH: {window_id!r} is a "
                            "critical application. Exact title match required. "
                            "Restoration may be incomplete.",
                            file=sys.stderr,
                        )
        except Exception:
            pass  # Title check is best-effort — never block restoration

    def _restore_cursor(self, snapshot: RestorationSnapshot) -> None:
        
        self._os.set_cursor_position(
            {"x": snapshot.cursor.x, "y": snapshot.cursor.y}
        )
        time.sleep(self.POST_ACTION_DELAY)

    

    def _verify(self, snapshot: RestorationSnapshot) -> None:
        
        if self._mode.mode is not SystemMode.RESTORING:
            raise RestorationError(
                "Verification attempted outside RESTORING mode — "
                "this indicates a mode-controller bug or external mode mutation."
            )

        _verify_cursor_failures = 0
        _verify_window_failures = 0
        _verify_app_failures = 0

        for attempt in range(1, self.MAX_VERIFY_ATTEMPTS + 1):

            cursor = self._os.get_cursor_position()
            current_window = self._os.get_focused_window()
            current_app = self._os.get_active_application()

            _cursor_ok = self._validate_cursor(cursor, snapshot)
            _window_ok = self._validate_window(current_window, snapshot)
            _app_ok = self._validate_application(current_app, snapshot)

            if _cursor_ok and _window_ok and _app_ok:
                return  # All checks passed — restoration verified

            # Increment per-dimension failure counters for diagnostics
            if not _cursor_ok:
                _verify_cursor_failures += 1
            if not _window_ok:
                _verify_window_failures += 1
            if not _app_ok:
                _verify_app_failures += 1

            # Wait before retrying — gives WM time to propagate focus events
            if attempt < self.MAX_VERIFY_ATTEMPTS:
                time.sleep(self.POST_ACTION_DELAY)

        # All attempts exhausted — raise with full diagnostic context
        raise RestorationError(
            f"Post-restore verification failed after {self.MAX_VERIFY_ATTEMPTS} attempts "
            f"(window={self.MAX_VERIFY_ATTEMPTS * self.POST_ACTION_DELAY:.1f}s total). "
            f"Failures: cursor={_verify_cursor_failures}, "
            f"window={_verify_window_failures}, "
            f"app={_verify_app_failures}. "
            f"Expected: cursor=({snapshot.cursor.x},{snapshot.cursor.y}) "
            f"±{self.CURSOR_TOLERANCE_PX}px, "
            f"window={snapshot.focus.window_id!r}, "
            f"app={snapshot.application.process_name!r}. "
            "Possible causes: "
            "(1) Window manager did not propagate focus within the verification window — "
            f"increase POST_ACTION_DELAY (currently {self.POST_ACTION_DELAY}s) or "
            f"MAX_VERIFY_ATTEMPTS (currently {self.MAX_VERIFY_ATTEMPTS}). "
            "(2) Target application was closed during task execution (check window failures). "
            "(3) Display server is unresponsive (check cursor failures)."
        )

    # =========================================================================
    # VALIDATION HELPERS
    # =========================================================================

    def _validate_cursor(self, cursor, snapshot: RestorationSnapshot) -> bool:
        """True iff cursor is within CURSOR_TOLERANCE_PX of the expected position."""
        if not isinstance(cursor, dict):
            return False

        try:
            cx = int(cursor["x"])
            cy = int(cursor["y"])
        except (KeyError, TypeError, ValueError):
            return False

        return (
            abs(cx - snapshot.cursor.x) <= self.CURSOR_TOLERANCE_PX
            and abs(cy - snapshot.cursor.y) <= self.CURSOR_TOLERANCE_PX
        )

    def _normalize(self, text: str) -> str:
        """Normalise a window/app title for fuzzy comparison."""
        return " ".join(text.lower().strip().split())

    def _levenshtein(self, a: str, b: str) -> int:
        
        return levenshtein_distance(a, b)

    def _strict_match(self, expected: str, actual: str) -> bool:
        
        if not expected or not actual:
            return False
        return _title_match_shared(expected, actual, max_distance=self.MAX_TITLE_DISTANCE)
        # RTB-02 FIX: removed dead code `shared = expected_tokens & actual_tokens`
        # (undefined names; unreachable after the return above)

    def _validate_window(self, current_window, snapshot: RestorationSnapshot) -> bool:
        """True iff the currently focused window matches the snapshot's window title."""
        # Bare desktop: no window expected — always passes
        if snapshot.focus.window_id == "__bare_desktop__":
            return True

        
        if snapshot.focus.window_id == "__wayland_unknown__  # Wayland: window title unavailable — cursor-only restore":
            return True

        if (
            not isinstance(current_window, dict)
            or not isinstance(current_window.get("title"), str)
        ):
            return False

        expected = self._normalize(snapshot.focus.window_id)
        actual = self._normalize(current_window["title"])
        return self._strict_match(expected, actual)

    def _report_unrestored_processes(self, snapshot: RestorationSnapshot) -> None:
        
        baseline_names = snapshot.metadata.get("extended", {}).get("processes")
        if not baseline_names:
            # Census was not captured at snapshot time (psutil unavailable,
            # degraded mode, or legacy snapshot).  Cannot compute diff.
            print(
                "[RestoreProvider] DEBUG IH-6: No process census in snapshot "
                "metadata['extended']['processes'] — unrestored-process diff "
                "skipped. Ensure psutil is installed for census capture.",
                file=sys.stderr,
            )
            return

        try:
            # Collect current process NAMES using the same strategy as snapshot
            # creation (psutil first, /proc fallback for Linux).
            current_names: list = []
            try:
                import psutil as _psutil
                current_names = sorted(
                    {p.name() for p in _psutil.process_iter(["name"]) if p.name()}
                )
            except ImportError:
                # psutil not installed — try /proc on Linux
                try:
                    import os as _os2
                    _names = set()
                    for _pid_str in _os2.listdir("/proc"):
                        if _pid_str.isdigit():
                            _comm = f"/proc/{_pid_str}/comm"
                            try:
                                with open(_comm, "r") as _f:
                                    _names.add(_f.read().strip())
                            except OSError:
                                pass
                    current_names = sorted(_names)
                except Exception:
                    print(
                        "[RestoreProvider] DEBUG IH-6: Cannot enumerate current "
                        "process names — unrestored-process diff skipped.",
                        file=sys.stderr,
                    )
                    return
            except Exception:
                print(
                    "[RestoreProvider] DEBUG IH-6: psutil.process_iter() failed "
                    "— unrestored-process diff skipped.",
                    file=sys.stderr,
                )
                return

            baseline_set = set(baseline_names)
            current_set = set(current_names)
            new_names = sorted(current_set - baseline_set)

            if not new_names:
                print(
                    f"[RestoreProvider] IH-6: Process census diff: 0 new process "
                    f"names since snapshot {snapshot.snapshot_id!r}. "
                    "No unrestored processes detected.",
                    file=sys.stderr,
                )
                return

            # Resolve new names to PIDs for actionable reporting.
            name_pids: dict = {}
            try:
                import psutil as _psutil2
                for _proc in _psutil2.process_iter(["pid", "name"]):
                    try:
                        _n = _proc.name()
                        if _n in new_names:
                            name_pids.setdefault(_n, []).append(_proc.pid)
                    except Exception:
                        pass
            except Exception:
                pass

            name_summary = ", ".join(
                f"{name}(pids={name_pids.get(name, ['?'])})" for name in new_names
            )

            print(
                f"[RestoreProvider] WARNING IH-6: {len(new_names)} process name(s) "
                f"present after restoration that were NOT present at snapshot time: "
                f"{name_summary}. "
                "Attempting SIGTERM on detected processes.",
                file=sys.stderr,
            )

            
            _TERMINATION_WHITELIST: frozenset = frozenset({
                # Core OS daemons that must never be killed
                "systemd", "init", "kernel", "kthreadd", "dbus-daemon",
                "NetworkManager", "pulseaudio", "pipewire",
                # Session components the desktop depends on
                "gnome-shell", "xorg", "x11", "wayland", "weston",
                # The agent's own supporting processes
                "ollama", "ydotoold",
            })

            _skip_term = (
                os.environ.get("PROJECTZEO_SKIP_PROCESS_TERM", "").strip().lower()
                in ("1", "true", "yes")
            )

            if not _skip_term:
                import signal as _signal
                import subprocess as _subprocess
                import shutil as _shutil
                _sigterm_pids: dict = {}
                for _name in new_names:
                    if _name in _TERMINATION_WHITELIST:
                        print(
                            f"[RestoreProvider] IH-6: Skipping SIGTERM for "
                            f"{_name!r} (in termination whitelist).",
                            file=sys.stderr,
                        )
                        continue
                    _pids = name_pids.get(_name, [])
                    for _pid in _pids:
                        try:
                            os.kill(_pid, _signal.SIGTERM)
                            _sigterm_pids.setdefault(_name, []).append(_pid)
                            print(
                                f"[RestoreProvider] IH-6: SIGTERM sent to "
                                f"{_name!r} (pid={_pid}).",
                                file=sys.stderr,
                            )
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            print(
                                f"[RestoreProvider] IH-6: SIGTERM denied for "
                                f"{_name!r} (pid={_pid}) — insufficient permission. "
                                "Process persists.",
                                file=sys.stderr,
                            )
                        except Exception as _sig_err:
                            print(
                                f"[RestoreProvider] IH-6: SIGTERM failed for "
                                f"{_name!r} (pid={_pid}): {_sig_err}.",
                                file=sys.stderr,
                            )

                if _sigterm_pids:
                    time.sleep(5.0)
                    for _name, _pids in _sigterm_pids.items():
                        for _pid in _pids:
                            try:
                                os.kill(_pid, 0)
                                os.kill(_pid, _signal.SIGKILL)
                                print(
                                    f"[RestoreProvider] IH-6: SIGKILL sent to "
                                    f"{_name!r} (pid={_pid}) — still alive after SIGTERM.",
                                    file=sys.stderr,
                                )
                            except (ProcessLookupError, PermissionError):
                                pass
                            except Exception as _kill_err:
                                print(
                                    f"[RestoreProvider] IH-6: SIGKILL failed for "
                                    f"{_name!r} (pid={_pid}): {_kill_err}.",
                                    file=sys.stderr,
                                )

                if _shutil.which("xclip"):
                    try:
                        _subprocess.run(
                            ["xclip", "-i", "/dev/null", "-selection", "clipboard"],
                            shell=False, capture_output=True, timeout=3,
                        )
                    except Exception:
                        pass
                elif _shutil.which("xsel"):
                    try:
                        _subprocess.run(
                            ["xsel", "--clear", "--clipboard"],
                            shell=False, capture_output=True, timeout=3,
                        )
                    except Exception:
                        pass
            else:
                print(
                    "[RestoreProvider] IH-6: PROJECTZEO_SKIP_PROCESS_TERM=1 — "
                    "process termination skipped. "
                    f"Unrestored processes: {name_summary}",
                    file=sys.stderr,
                )

            try:
                snapshot.metadata["unrestored_process_names"] = new_names
                snapshot.metadata["unrestored_process_name_pids"] = name_pids
            except Exception:
                pass

        except Exception as _diff_err:
            print(
                f"[RestoreProvider] DEBUG IH-6: Process census diff failed "
                f"unexpectedly: {_diff_err}. Restoration is still complete.",
                file=sys.stderr,
            )

    def _validate_application(
        self, current_app, snapshot: RestorationSnapshot
    ) -> bool:
        """True iff the currently active application matches the snapshot's app name."""
        # Bare desktop: no application expected — always passes
        if snapshot.application.process_name == "__bare_desktop__":
            return True

        if (
            not isinstance(current_app, dict)
            or not isinstance(current_app.get("title"), str)
        ):
            return False

        expected = self._normalize(snapshot.application.process_name)
        actual = self._normalize(current_app["title"])
        return self._strict_match(expected, actual)
