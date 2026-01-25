# kernel.py

import os
import time
import subprocess
from datetime import datetime

from observer.screenpipe_adapter import capture_frame
from observer.perception_engine import PerceptionEngine

from restoration.snapshot_provider import take_snapshot
from restoration.restore_provider import restore_snapshot

from audit.journal import Journal

FRAMES_DIR = "frames"
SOC_TRIGGER_TEXT = "SOC READY"   # text visible on screen when user wants SOC

os.makedirs(FRAMES_DIR, exist_ok=True)

journal = Journal()
perception = PerceptionEngine()

print("[KERNEL] Booted")
print("[KERNEL] Observer mode")

while True:

    frame = capture_frame()

    if frame is None:
        time.sleep(1)
        continue

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    frame_path = os.path.join(FRAMES_DIR, f"{ts}.png")
    frame.save(frame_path)

    state = perception.extract(frame_path)
    texts = state.get("text_blocks", [])

    trigger = False
    for t in texts:
        if SOC_TRIGGER_TEXT in t:
            trigger = True
            break

    if trigger:

        journal.write({"event": "soc_triggered"})

        snapshot_id = take_snapshot()

        soc = subprocess.Popen(
            ["python", "operate/main.py"]
        )

        soc.wait()

        restore_snapshot(snapshot_id)

        journal.write({"event": "soc_finished"})

        print("[KERNEL] Returned to observer")

    time.sleep(1)
