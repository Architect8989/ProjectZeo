# core/control/kernel_controller.py

from enum import Enum, auto
import time
from typing import Optional, Any

import operate.main as main  # runtime surface

from core.schemas.action_schema import validate_actions


class KernelState(Enum):
    OBSERVER = auto()
    ARMED = auto()
    PLANNING = auto()
    EXECUTING = auto()
    VERIFYING = auto()
    RESTORING = auto()
    ERROR = auto()


class KernelController:
    """
    Governing state machine for the entire system.

    - Owns all state transitions
    - Calls into existing operate.main entrypoints
    - No business logic inside kernel
    """

    TICK_INTERVAL = 0.05  # 20Hz control loop

    def __init__(self):
        self.state: KernelState = KernelState.OBSERVER
        self.current_intent: Optional[str] = None
        self.current_plan: Optional[Any] = None

    # -----------------------
    # Public Entry
    # -----------------------

    def start(self):
        while True:
            try:
                self._step()
            except Exception as e:
                print("[KERNEL] Unhandled error:", e)
                self._transition(KernelState.ERROR)

            time.sleep(self.TICK_INTERVAL)

    # -----------------------
    # Core Loop
    # -----------------------

    def _step(self):
        if self.state == KernelState.OBSERVER:
            self._observer()

        elif self.state == KernelState.ARMED:
            self._armed()

        elif self.state == KernelState.PLANNING:
            self._planning()

        elif self.state == KernelState.EXECUTING:
            self._executing()

        elif self.state == KernelState.VERIFYING:
            self._verifying()

        elif self.state == KernelState.RESTORING:
            self._restoring()

        elif self.state == KernelState.ERROR:
            self._error()

    # -----------------------
    # State Handlers
    # -----------------------

    def _observer(self):
        main.start_observer()

        intent = main.check_for_user_intent()
        if intent:
            self.current_intent = intent
            self._transition(KernelState.ARMED)

    def _armed(self):
        main.arm_system()
        self._transition(KernelState.PLANNING)

    def _planning(self):
        self.current_plan = main.generate_plan(self.current_intent)

        # SCHEMA ENFORCEMENT
        if not validate_actions(self.current_plan):
            print("[KERNEL] Invalid plan schema. Replanning.")
            self._transition(KernelState.PLANNING)
            return

        self._transition(KernelState.EXECUTING)

    def _executing(self):
        # RE-VALIDATE BEFORE EXECUTION
        if not validate_actions(self.current_plan):
            print("[KERNEL] Plan corrupted before execution. Replanning.")
            self._transition(KernelState.PLANNING)
            return

        main.run_soc(self.current_plan)
        self._transition(KernelState.VERIFYING)

    def _verifying(self):
        success = main.verify_task()

        if success:
            self._transition(KernelState.RESTORING)
        else:
            self._transition(KernelState.EXECUTING)

    def _restoring(self):
        main.restore_screen()
        self.current_intent = None
        self.current_plan = None
        self._transition(KernelState.OBSERVER)

    def _error(self):
        main.restore_screen()
        self.current_intent = None
        self.current_plan = None
        self._transition(KernelState.OBSERVER)

    # -----------------------
    # Transition Guard
    # -----------------------

    def _transition(self, new_state: KernelState):
        print(f"[KERNEL] {self.state.name} -> {new_state.name}")
        self.state = new_state
