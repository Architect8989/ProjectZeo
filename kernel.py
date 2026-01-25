# kernel.py

import os
import time
import subprocess
from datetime import datetime

# ------------------------
# IMPORT REAL MODULES
# ------------------------

from observer.screenpipe_adapter import capture_frame
from observer.perception_engine import PerceptionEngine

from audit.journal import Journal

from restoration.snapshot_provider import take_snapshot
from restoration.restore_provider import restore_snapshot

# ------------------------
# CONFIG
# ------------------------

FRAMES_DIR = "frames"
INTENT_PREFIX = "[INTENT]"

os.makedirs(FRAMES_DIR, exist_ok=True)

journal = Journal()
perception = PerceptionEngine()

print("[KERNEL] Booted")
print("[KERNEL] Observer mode")

# ------------------------
# MAIN LOOP
# ------------------------

while True:

    # ------------------------
    # OBSERVER MODE
    # ------------------------

    frame = capture_frame()

    if frame is None:
        time.sleep(1)
        continue

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    frame_path = os.path.join(FRAMES_DIR, f"{ts}.png")
    frame.save(frame_path)

    journal.write({
        "component": "observer",
        "frame": frame_path
    })

    # ------------------------
    # PERCEPTION (INTENT)
    # ------------------------

    state = perception.extract(frame_path)
    texts = state.get("text_blocks", [])

    intent = None
    for t in texts:
        if t.startswith(INTENT_PREFIX):
            intent = t.replace(INTENT_PREFIX, "").strip()
            break

    # ------------------------
    # EXECUTOR MODE
    # ------------------------

    if intent:

        print("[KERNEL] Intent detected:", intent)

        journal.write({
            "component": "kernel",
            "event": "intent_detected",
            "task": intent
        })

        # ------------------------
        # SNAPSHOT
        # ------------------------

        snapshot_id = take_snapshot()

        journal.write({
            "component": "kernel",
            "event": "snapshot_taken",
            "snapshot_id": snapshot_id
        })

        # ------------------------
        # RUN SOC
        # ------------------------

        print("[KERNEL] Launching SOC")

        soc = subprocess.Popen(
            [
                "python",
                "operate/main.py",
                "--task",
                intent,
                "--screenshot",
                frame_path
            ]
        )

        soc.wait()

        journal.write({
            "component": "kernel",
            "event": "soc_finished"
        })

        # ------------------------
        # RESTORE
        # ------------------------

        print("[KERNEL] Restoring screen")

        restore_snapshot(snapshot_id)

        journal.write({
            "component": "kernel",
            "event": "restored",
            "snapshot_id": snapshot_id
        })

        print("[KERNEL] Back to observer mode")

    time.sleep(1)
