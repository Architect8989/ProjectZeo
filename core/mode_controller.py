class ModeController:
    OBSERVER = "OBSERVER"
    ARMED = "ARMED"
    EXECUTING = "EXECUTING"

    def __init__(self):
        self.mode = self.OBSERVER
