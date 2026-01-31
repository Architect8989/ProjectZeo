from enum import Enum, auto
import time
from typing import Optional, Any, List
import gc

import operate.main as main  # runtime surface

from core.schemas.action_schema import validate_actions
from core.verification.screen_verifier import verify_execution
from core.memory.playbook_store import load_playbook, save_playbook
from core.safety.runtime_watchdog import RuntimeWatchdog
from core.telemetry.logger import log_info, log_warn, log_error
from core.safety.checkpoint_store import (
    save_checkpoint,
    load_checkpoint,
    clear_checkpoint,
)

from core.planner.task_decomposer import TaskDecomposer, DecompositionError
from operate.models.apis_openrouter import get_next_action


# -----------------------------------------------------
# Kernel States
# -----------------------------------------------------

class KernelState(Enum):
    OBSERVER = auto()
    ARMED = auto()
    PLANNING = auto()
    EXECUTING = auto()
    VERIFYING = auto()
    RESTORING = auto()
    ERROR = auto()


# -----------------------------------------------------
# Kernel Controller
# -----------------------------------------------------

class KernelController:
    """
    Sovereign governing state machine.

    - Owns transitions
    - Owns persistence
    - Owns recovery
    - No business logic
    """

    TICK_INTERVAL = 0.05

    STATE_TIMEOUTS = {
        KernelState.PLANNING: 600,
        KernelState.EXECUTING: 1800,
        KernelState.VERIFYING: 300,
    }

    # -------------------------------------------------

    def __init__(self):
        self.state = KernelState.OBSERVER

        self.current_intent: Optional[str] = None
        self.steps: Optional[List[dict]] = None
        self.step_index: int = 0
        self.current_plan: Optional[Any] = None

        self.retry_count = 0
        self.max_retries = 5

        self.step_count = 0
        self.max_steps = 500

        self.watchdog = RuntimeWatchdog()
        self.state_enter_time = time.time()

        self.decomposer = TaskDecomposer(
            llm_call=lambda prompt: get_next_action(
                model="openai/gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                objective="planner",
            )[0]
        )

        # ----------------------------
        # Restore checkpoint
        # ----------------------------

        ckpt = load_checkpoint()
        if ckpt:
            log_warn("[KERNEL] Restoring from checkpoint")

            try:
                self.state = KernelState[ckpt["state"]]
                self.current_intent = ckpt["current_intent"]
                self.steps = ckpt["steps"]
                self.step_index = ckpt["step_index"]
                self.current_plan = ckpt["current_plan"]
                self.retry_count = ckpt["retry_count"]
                self.step_count = ckpt["step_count"]
            except Exception:
                log_error("[KERNEL] Corrupt checkpoint discarded")
                clear_checkpoint()

    # -------------------------------------------------
    # Public
    # -------------------------------------------------

    def start(self):
        while True:
            try:
                self._step()
            except SystemExit:
                self._transition(KernelState.ERROR)
            except Exception as e:
                log_error(f"[KERNEL] Unhandled exception: {e}")
                self._transition(KernelState.ERROR)

            time.sleep(self.TICK_INTERVAL)

    # -------------------------------------------------
    # Core Loop
    # -------------------------------------------------

    def _step(self):

        self.watchdog.check()

        save_checkpoint({
            "state": self.state.name,
            "current_intent": self.current_intent,
            "steps": self.steps,
            "step_index": self.step_index,
            "current_plan": self.current_plan,
            "retry_count": self.retry_count,
            "step_count": self.step_count,
        })

        timeout = self.STATE_TIMEOUTS.get(self.state)
        if timeout and (time.time() - self.state_enter_time > timeout):
            log_warn(f"[KERNEL] State timeout: {self.state.name}")
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

    # -------------------------------------------------
    # States
    # -------------------------------------------------

    def _observer(self):
        main.start_observer()
        intent = main.check_for_user_intent()

        if intent:
            self.current_intent = intent
            self._transition(KernelState.ARMED)

    def _armed(self):
        main.arm_system()
        self._transition(KernelState.PLANNING)

    # -------------------------------------------------

    def _planning(self):

        try:
            self.steps = self.decomposer.decompose(self.current_intent)
            self.step_index = 0
        except DecompositionError as e:
            log_warn(f"[KERNEL] Decomposition failed: {e}")
            return

        self._transition(KernelState.EXECUTING)

    # -------------------------------------------------

    def _executing(self):

        if self.step_index >= len(self.steps):
            self._transition(KernelState.RESTORING)
            return

        step_goal = self.steps[self.step_index]["goal"]

        cached = load_playbook(step_goal)
        if cached:
            self.current_plan = cached
        else:
            self.current_plan = main.generate_plan(step_goal)

        if not validate_actions(self.current_plan):
            log_warn("[KERNEL] Invalid plan schema")
            self._transition(KernelState.ERROR)
            return

        main.run_soc(self.current_plan)
        self._transition(KernelState.VERIFYING)

    # -------------------------------------------------

    def _verifying(self):

        screenshot = main.get_latest_screenshot()

        success = verify_execution(
            self.current_plan,
            screenshot
        )

        if success:
            save_playbook(
                self.steps[self.step_index]["goal"],
                self.current_plan,
            )
            self.step_index += 1
            self.retry_count = 0
            self._transition(KernelState.EXECUTING)
        else:
            self.retry_count += 1

            if self.retry_count > self.max_retries:
                log_warn("[KERNEL] Retry limit exceeded")
                self._transition(KernelState.ERROR)
                return

            self._transition(KernelState.EXECUTING)

    # -------------------------------------------------

    def _restoring(self):
        main.restore_screen()
        clear_checkpoint()
        self._reset()
        self._transition(KernelState.OBSERVER)

    def _error(self):
        main.restore_screen()
        clear_checkpoint()
        self._reset()
        self._transition(KernelState.OBSERVER)

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _reset(self):
        self.current_intent = None
        self.steps = None
        self.step_index = 0
        self.current_plan = None
        self.retry_count = 0
        self.step_count = 0
        gc.collect()

    def _transition(self, new_state: KernelState):
        log_info(f"[KERNEL] {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.state_enter_time = time.time()
