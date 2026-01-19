import time
from typing import Dict, Optional


class ScreenpipeAdapter:
    """
    Read-only adapter for Screenpipe outputs.

    This class NEVER:
    - starts Screenpipe
    - controls Screenpipe
    - interacts with the OS
    - blocks the main loop

    It only attempts to read observation data if available.
    """

    def __init__(self):
        self.last_read_ts: Optional[float] = None

        # Normalized observation structure (stable contract)
        self.state: Dict[str, object] = {
            "screen_text": None,
            "screen_text_hash": None,
            "frame_ts": None,
            "available": False,
        }

        print("[SCREENPIPE] Adapter initialized (passive)")

    def read(self) -> Dict[str, object]:
        """
        Attempt to read latest observation from Screenpipe.

        Safe behavior:
        - If Screenpipe is not running or no data exists,
          returns a stable empty state.
        - Never raises exceptions to caller.
        """
        try:
            # Placeholder for future Screenpipe integration
            # Example (later):
            # data = self._read_from_screenpipe_api()

            # For now: no data source
            self.state["available"] = False
            self.state["frame_ts"] = None
            self.state["screen_text"] = None
            self.state["screen_text_hash"] = None

            self.last_read_ts = time.time()

        except Exception as e:
            # Absolute safety: observer must never crash
            self.state["available"] = False

        return dict(self.state)

    def is_available(self) -> bool:
        """
        Indicates whether Screenpipe data is currently available.
        """
        return bool(self.state.get("available", False))
