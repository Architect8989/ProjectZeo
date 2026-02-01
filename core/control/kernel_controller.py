from enum import Enum, auto
import time
from typing import Optional, Any, List
import gc
import json
import asyncio

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

from core.safety.restart_guard import (
    record_restart,
    restart_allowed,
)

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
# Planner Adapter
# -----------------------------------------------------

def planner_llm_call(prompt: str) -> str:
    """
    Synchronous adapter.
    Returns raw JSON string.
    """

    async def _call():
        ops, _ = await get_next_action(
            model="openai/gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            objective="planner",
        )
        return json.dumps({"steps": ops})

    return asyncio.run(_call())


# -----------------------------------------------------
# Kernel Controller
# -----------------------------------------------------

class KernelController:
    """
    Sovereign governing state machine.

    HARD GUARANTEES:
    - Single execution surface
    - No phantom wiring
    - Honest timeout semantics
    """

    TICK_INTERVAL = 0.05

    STATE_TIMEOUTS = {
        KernelState.PLANNING: 600,
        KernelState.EXECUTING: 1800,
        KernelState.VERIFYING: 300,
    }

    # -------------------------------------------------

    def __init__(self, *, config: Optional[dict] = None, operate_entry=None):
        # ---- Restart Guard ----
        if not restart_allowed():
            raise SystemExit("RESTART_GUARD_LOCKED")
        record_restart()

        if operate_entry is None or not callable(operate_entry):
            raise RuntimeError(
                "KernelController requires a callable operate_entry"
            )

        self.config = config or {}
        self.operate_entry = operate_entry

        self.state = KernelState.OBSERVER
        self.state_enter_time = time.time()

        self.current_intent: Optional[str] = None
        self.steps: Optional[List[dict]] = None
        self.step_index: int = 0
        self.current_plan: Optional[Any] = None

        self.retry_count = 0
        self.max_retries = 5

        self.step_count = 0
        self.max_steps = 500

        self.watchdog = RuntimeWatchdog()

        self.decomposer = TaskDecomposer(
            llm_call=planner_llm_call
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
                self.state_enter_time = time.time()
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

        # Best-effort watchdog (cannot interrupt blocking ops)
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
        # Intent must be injected externally
        if self.current_intent:
            self._transition(KernelState.ARMED)

    def _armed(self):
        self._transition(KernelState.PLANNING)

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

        if self.step_count >= self.max_steps:
            log_warn("[KERNEL] Step budget exceeded")
            self._transition(KernelState.ERROR)
            return

        step_goal = self.steps[self.step_index]["goal"]

        cached = load_playbook(step_goal)
        if cached:
            self.current_plan = cached
        else:
            self.current_plan = self.operate_entry(
                intent=self.current_intent,
                goal=step_goal,
            )

        if not validate_actions(self.current_plan):
            log_warn("[KERNEL] Invalid plan schema")
            self._transition(KernelState.ERROR)
            return

        self.step_count += 1
        self._transition(KernelState.VERIFYING)

    # -------------------------------------------------

    def _verifying(self):
        # Kernel does NOT own vision.
        # Verification is best-effort without screenshot.
        success = verify_execution(
            actions=self.current_plan,
            screenshot=None,
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
        clear_checkpoint()
        self._reset()
        self._transition(KernelState.OBSERVER)

    def _error(self):
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
