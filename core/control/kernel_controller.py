# core/control/kernel_controller.py

from enum import Enum, auto
import time
from typing import Optional, Any
import gc

import operate.main as main  # runtime surface

from core.schemas.action_schema import validate_actions
from core.verification.screen_verifier import verify_execution
from core.memory.playbook_store import load_playbook, save_playbook
from core.safety.runtime_watchdog import RuntimeWatchdog


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

    # Per-state hard timeouts (seconds)
    STATE_TIMEOUTS = {
        KernelState.PLANNING: 600,
        KernelState.EXECUTING: 1800,
        KernelState.VERIFYING: 300,
    }

    def __init__(self):
        self.state: KernelState = KernelState.OBSERVER
        self.current_intent: Optional[str] = None
        self.current_plan: Optional[Any] = None

        # Safety budgets
        self.retry_count = 0
        self.max_retries = 5

        self.step_count = 0
        self.max_steps = 200

        # Runtime watchdog
        self.watchdog = RuntimeWatchdog()

        # State timer
        self.state_enter_time = time.time()

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

        # Global resource guard
        self.watchdog.check()

        # State timeout guard
        timeout = self.STATE_TIMEOUTS.get(self.state)
        if timeout:
            if time.time() - self.state_enter_time > timeout:
                print("[KERNEL] State timeout:", self.state.name)
                self._transition(KernelState.ERROR)
                return

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

        cached = load_playbook(self.current_intent)
        if cached:
            print("[KERNEL] Loaded playbook from memory.")
            self.current_plan = cached
            self._transition(KernelState.EXECUTING)
            return

        self.current_plan = main.generate_plan(self.current_intent)

        if not validate_actions(self.current_plan):
            print("[KERNEL] Invalid plan schema. Replanning.")
            return

        self._transition(KernelState.EXECUTING)

    def _executing(self):

        self.step_count += 1
        if self.step_count > self.max_steps:
            print("[KERNEL] Step limit exceeded.")
            self._transition(KernelState.ERROR)
            return

        if not validate_actions(self.current_plan):
            print("[KERNEL] Plan corrupted before execution.")
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
        self._reset()
        self._transition(KernelState.OBSERVER)

    def _error(self):

        main.restore_screen()
        self._reset()
        self._transition(KernelState.OBSERVER)

    # -----------------------
    # Internal Helpers
    # -----------------------

    def _reset(self):
        self.current_intent = None
        self.current_plan = None
        self.retry_count = 0
        self.step_count = 0

        gc.collect()

    # -----------------------
    # Transition Guard
    # -----------------------

    def _transition(self, new_state: KernelState):
        print(f"[KERNEL] {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.state_enter_time = time.time()
