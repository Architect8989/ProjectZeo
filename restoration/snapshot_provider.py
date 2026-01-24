from __future__ import annotations

import time
from typing import Dict, Any

from restoration.snapshot_types import (
    CursorState,
    FocusState,
    ApplicationState,
    RestorationSnapshot,
)

# Existing system components (authoritative)
from observer.screenpipe_adapter import ScreenpipeAdapter
from observer.observer_core import ObserverCore


class SnapshotProviderError(RuntimeError):
    pass


class SnapshotProvider:
    """
    Concrete snapshot provider.

    This is NOT optional.
    If this fails, SOC must never run.
    """

    def __init__(
        self,
        *,
        observer: ObserverCore,
        screenpipe: ScreenpipeAdapter,
        os_backend,
    ):
        """
        os_backend MUST provide:
          - get_cursor_position() -> (x, y)
          - get_focused_window() -> dict {id, title}
          - get_active_application() -> dict {process_name, pid}
          - get_execution_mode() -> str
        """
        self._observer = observer
        self._screenpipe = screenpipe
        self._os = os_backend

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def capture_pre_hijack_snapshot(self) -> RestorationSnapshot:
        """
        Capture and validate pre-hijack snapshot.

        This method is a hard gate.
        Any exception here must abort execution.
        """

        # 1. Enforce execution mode
        execution_mode = self._os.get_execution_mode()
        if execution_mode != "OBSERVER":
            raise SnapshotProviderError(
                f"Snapshot capture attempted in mode '{execution_mode}'. "
                "Snapshots MUST be captured in OBSERVER mode."
            )

        # 2. Enforce live vision (already hardened upstream, rechecked here)
        screen_state = self._screenpipe.read()
        if not screen_state.get("available") or screen_state.get("blind"):
            raise SnapshotProviderError(
                "Screenpipe vision unavailable or blind during snapshot capture"
            )

        # 3. Pull OS-authoritative state
        try:
            cursor_x, cursor_y = self._os.get_cursor_position()
            focused_window = self._os.get_focused_window()
            active_app = self._os.get_active_application()
        except Exception as e:
            raise SnapshotProviderError(
                f"Failed to retrieve OS state: {e}"
            ) from e

        # 4. Build state objects (with validation)
        cursor_state = CursorState(
            x=int(cursor_x),
            y=int(cursor_y),
        )

        focus_state = FocusState(
            window_id=str(focused_window.get("id")),
            title=focused_window.get("title"),
        )

        application_state = ApplicationState(
            process_name=str(active_app.get("process_name")),
            pid=active_app.get("pid"),
        )

        # 5. Bind visual evidence (Screenpipe)
        metadata: Dict[str, Any] = {
            "screenpipe": {
                "frame_ts": screen_state.get("frame_ts"),
                "screen_text_hash": screen_state.get("screen_text_hash"),
                "captured_at": time.time(),
            }
        }

        # 6. Create immutable snapshot (contract-enforced)
        snapshot = RestorationSnapshot.create(
            cursor=cursor_state,
            focus=focus_state,
            application=application_state,
            execution_mode=execution_mode,
            metadata=metadata,
        )

        # 7. Notify observer (witness continuity)
        try:
            self._observer.attach_ui_snapshot(snapshot.to_dict())
        except Exception:
            # Observer attachment must never block snapshot
            pass

        return snapshot
