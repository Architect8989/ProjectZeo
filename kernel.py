# kernel.py

import os
import time
import subprocess
from datetime import datetime

from observer.screenpipe_adapter import ScreenpipeAdapter
from observer.perception_engine import PerceptionEngine

from restoration.snapshot_provider import SnapshotProvider
from restoration.restore_provider import RestoreProvider

from operate.utils.operating_system import OperatingSystem
from audit.journal import Journal

FRAMES_DIR = "frames"
SOC_TRIGGER_TEXT = "SOC READY"   # text visible on screen when user wants SOC

os.makedirs(FRAMES_DIR, exist_ok=True)

journal = Journal()
perception = PerceptionEngine()
screenpipe = ScreenpipeAdapter()

os_backend = OperatingSystem()
snapshot_provider = SnapshotProvider(observer=None, screenpipe=screenpipe)
restore_provider = RestoreProvider(os_backend)

print("[KERNEL] Booted")
print("[KERNEL] Observer mode")

while True:
    try:
        # ---- SCREEN CAPTURE (REAL API) ----
        screen_state = screenpipe.read()

        if not screen_state or not screen_state.get("available"):
            time.sleep(1)
            continue

        frame = screen_state.get("frame")
        if frame is None:
            time.sleep(1)
            continue

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        frame_path = os.path.join(FRAMES_DIR, f"{ts}.png")
        frame.save(frame_path)

        # ---- PERCEPTION (REAL API) ----
        ui_snapshot = perception.process(screen_state)

        texts = []
        for block in ui_snapshot.text_blocks:
            texts.append(block.text)

        trigger = any(SOC_TRIGGER_TEXT in t for t in texts)

        if trigger:
            journal.write({"event": "soc_triggered"})

            snapshot_id = snapshot_provider.take_snapshot()

            soc = subprocess.Popen(
                ["python", "operate/main.py"]
            )

            soc.wait()

            restore_provider.restore_snapshot(snapshot_id)

            journal.write({"event": "soc_finished"})
            print("[KERNEL] Returned to observer")

    except Exception as e:
        print(f"[KERNEL] Error: {e}")

    time.sleep(1)
