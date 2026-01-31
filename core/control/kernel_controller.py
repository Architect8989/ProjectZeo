# core/control/kernel_controller.py

from enum import Enum, auto
import time
from typing import Optional, Any

import operate.main as main  # runtime surface

from core.schemas.action_schema import validate_actions
from core.verification.screen_verifier import verify_execution
from core.memory.playbook_store import load_playbook, save_playbook


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

        # Safety budgets
        self.retry_count = 0
        self.max_retries = 5

        self.step_count = 0
        self.max_steps = 200

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

        # MEMORY LOOKUP
        cached = load_playbook(self.current_intent)
        if cached:
            print("[KERNEL] Loaded playbook from memory.")
            self.current_plan = cached
            self._transition(KernelState.EXECUTING)
            return

        self.current_plan = main.generate_plan(self.current_intent)

        if not validate_actions(self.current_plan):
            print("[KERNEL] Invalid plan schema. Replanning.")
            self._transition(KernelState.PLANNING)
            return

        self._transition(KernelState.EXECUTING)

    def _executing(self):
        # Step budget
        self.step_count += 1
        if self.step_count > self.max_steps:
            print("[KERNEL] Step limit exceeded.")
            self._transition(KernelState.ERROR)
            return

        if not validate_actions(self.current_plan):
            print("[KERNEL] Plan corrupted before execution. Replanning.")
            self._transition(KernelState.PLANNING)
            return

        main.run_soc(self.current_plan)
        self._transition(KernelState.VERIFYING)

    def _verifying(self):
        screenshot = main.get_latest_screenshot()

        success = verify_execution(
            self.current_plan,
            screenshot
        )

        if success:
            # SAVE MEMORY
            save_playbook(self.current_intent, self.current_plan)
            self._transition(KernelState.RESTORING)
        else:
            self.retry_count += 1

            if self.retry_count > self.max_retries:
                print("[KERNEL] Retry limit exceeded.")
                self._transition(KernelState.ERROR)
                return

            print("[KERNEL] Verification failed. Retrying execution.")
            self._transition(KernelState.EXECUTING)

    def _restoring(self):
        main.restore_screen()
        self.current_intent = None
        self.current_plan = None

        self.retry_count = 0
        self.step_count = 0

        self._transition(KernelState.OBSERVER)

    def _error(self):
        main.restore_screen()
        self.current_intent = None
        self.current_plan = None

        self.retry_count = 0
        self.step_count = 0

        self._transition(KernelState.OBSERVER)

    # -----------------------
    # Transition Guard
    # -----------------------

    def _transition(self, new_state: KernelState):
        print(f"[KERNEL] {self.state.name} -> {new_state.name}")
        self.state = new_state
