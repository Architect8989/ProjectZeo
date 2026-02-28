# ProjectZeo Architecture

## 1. System Overview

ProjectZeo is a deterministic, fail-closed system for executing computer interactions via an external LLM. It operates a real OS through mouse and keyboard control, grounded in screen observation.

**Core Principle:** Intelligence lives ONLY in the LLM during planning. All execution, verification, restoration, observation, and progress tracking are deterministic and LLM-free.

**Key Properties:**
- Not autonomous: requires explicit human intent per execution
- Not continuous: single-shot execution with mandatory restoration
- Not a background agent: foreground, observable, time-bounded
- Fail-closed: restoration always attempted, failures explicit
- Screen-grounded: all verification tied to observable evidence

---

## 2. Body vs Brain Separation

### 2.1 Brain (External LLM)

**Location:** Any LLM But VISION BASED
**Usage:** ONLY during planning phase  
**Function:** Convert human intent into structured ExecutionPlan  
**Constraints:**
- Time-bounded: 30s hard timeout per LLM call
- Output must be valid JSON matching ExecutionStep schema
- No direct OS access
- No execution authority
- No verification authority
- No retry logic

**Interface:**
```python
def llm_call(prompt: str) -> str:
    # Returns structured JSON only
    # Prompt contains: environment fingerprint, screen context, task
    # Output: List[ExecutionStep] in JSON format
```

### 2.2 Body (Deterministic System)

**Components:** Observer, Executor, Verifier, Restorer, Authority, Progress Tracker  
**Function:** Execute, verify, and restore without intelligence  
**Constraints:**
- No LLM calls during execution
- All verification evidence-based (filesystem, process, screen hash)
- Restoration always attempted
- Authority enforced deterministically

---

## 3. System Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                      OBSERVER MODE                          │
│  - Screen reading continuous (visionllm)                   │
│  - Vision health monitoring                                 │
│  - No automated input                                       │
│  - Listening for human intent                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Human provides intent
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      ARMED MODE                             │
│  - Intent captured and frozen                               │
│  - Pre-execution snapshot taken                             │
│  - Automation marked inactive                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Snapshot complete
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   PLANNING MODE (LLM ZONE)                  │
│  - LLM receives: intent + environment + screen context      │
│  - LLM produces: ExecutionPlan (structured steps)           │
│  - Hard timeout: 30 seconds                                 │
│  - Validation: schema conformance, dependency graph         │
│  - NO execution, NO verification, NO restoration            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Plan validated and attached
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   EXECUTING MODE (NO LLM)                   │
│  - Deterministic step execution                             │
│  - Authority arbitration per step                           │
│  - Evidence-based verification per step                     │
│  - Progress tracking                                        │
│  - Failure recovery (retry or abort)                        │
│  - Wall-clock timeout: 90 minutes hard cap                  │
│  - Human input → immediate yield                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Complete/Fail/Abort/Timeout
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   RESTORING (NO LLM)                        │
│  - Cease all automated input immediately                    │
│  - Release all OS resources                                 │
│  - Restore cursor position                                  │
│  - Restore window focus                                     │
│  - Restore active application                               │
│  - Verify restoration                                       │
│  - Transition to OBSERVER mode                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Restoration verified
                       ▼
                   OBSERVER MODE
                   (cycle complete)
```

**Transition Rules:**
- Linear: OBSERVER → ARMED → PLANNING → EXECUTING → OBSERVER
- No backward transitions except abort
- No skipped phases
- Restoration always occurs before returning to OBSERVER

---

## 4. LLM Zone Boundaries

### 4.1 ALLOWED (Planning Phase Only)

**When:** System is in PLANNING mode  
**What LLM sees:**
- Human intent (frozen, immutable)
- Environment fingerprint (OS, available tools)
- Screen context (read-only, up to 500 chars)

**What LLM produces:**
- List of ExecutionStep objects
- Each step has: type, description, action, verification criteria
- Dependencies between steps
- Estimated duration per step

**Validation:**
- JSON schema conformance
- Step types must be in StepType enum
- Dependencies form valid DAG (no cycles)
- Exactly one DONE step at end
- All UI actions conform to action schema

**Timeout:** 30 seconds hard limit via ThreadPoolExecutor

### 4.2 FORBIDDEN (Execution, Verification, Restoration, Observation)

**During EXECUTING mode:**
- No LLM calls for action decisions
- No LLM calls for verification
- No LLM calls for failure recovery
- No LLM calls for coordination

**During verification:**
- Evidence-based only: filesystem, process table, screen hash
- No LLM interpretation of screenshots
- No LLM assessment of success/failure

**During restoration:**
- Deterministic OS calls only
- No LLM decisions about what to restore
- No LLM involvement in verification

**During observation:**
- Screen reading via screenpipe (local vision model)
- Perception engine processes UI elements
- Observer core tracks health
- No LLM calls for understanding

---

## 5. Component Architecture

### 5.1 Observer Layer (NO LLM)

**ObserverCore:**
- Passive witness to screen state
- Tracks vision health
- Detects blindness (no frames, consecutive misses)
- Provides UI element query interface
- Deep-copies all state (snapshot isolation)
- Raises ObserverBlindnessError on persistent failure

**PerceptionEngine:**
- Processes raw screen data into structured UI snapshots
- Local vision model (no LLM)
- Extracts clickable elements with coordinates

**ScreenpipeAdapter:**
- Interfaces with screenpipe daemon
- Provides screen text, hash, timestamp
- Self-test capability
- Blindness detection

### 5.2 Authority Layer (NO LLM)

**ModeController:**
- Single source of truth for system state
- Enforces linear lifecycle
- Guards planning contract (plan must be attached)
- Freezes intent during planning
- Provides LLM callable ONLY during planning
- No intelligence, pure enforcement

**InputArbitrator:**
- Evaluates authority per step
- Detects human input
- Makes yield/proceed decisions
- No LLM involvement

**AuthorityPolicy:**
- Rule-based decision making
- High-risk steps flagged
- Human supremacy clause

### 5.3 Execution Layer (NO LLM)

**ExecutionPlanner (LLM BOUNDARY):**
- ONLY component that calls LLM
- ONLY during PLANNING mode
- Produces validated ExecutionPlan
- Hard timeout enforcement
- No execution, no verification

**ProgressTracker:**
- Deterministic step completion tracking
- Dependency satisfaction checking
- No intelligence

**FailureRecoveryManager:**
- Deterministic retry logic
- Step retryability from plan
- No LLM-based recovery

**StepVerifier:**
- Evidence-based verification ONLY
- Command: checks return code, output contains
- File: checks existence, content contains
- Tool: checks which, version
- UI: checks screen hash change OR explicit text match
- No LLM interpretation

### 5.4 Restoration Layer (NO LLM)

**SnapshotProvider:**
- Captures pre-execution state
- Cursor position, focused window, active app
- Mode state, screen state
- Extended state (best-effort)

**RestoreProvider:**
- Deterministic restoration
- Idempotent per snapshot
- Fail-closed: never lies about success
- Phase 0: Force release all automation
- Phase 1: Extended state (best-effort)
- Phase 2: Core state (fail-closed)
- Phase 3: Authority reset to OBSERVER
- Phase 4: Verification (cursor, focus, mode)

**RestoreVerifier:**
- Reads cursor position
- Reads focused window
- Checks mode == OBSERVER
- Raises RestorationError on mismatch

### 5.5 Safety Layer (NO LLM)

**RuntimeWatchdog:**
- Wall-clock timeout: 90 minutes
- Per-action timeout
- No grace period
- Immediate abort

**CheckpointStore:**
- Persists execution state
- Crash recovery
- Dirty flag tracking

**RestartGuard:**
- Crash detection
- Forces restoration on restart
- Forces OBSERVER mode

---

## 6. Safety Guarantees

### 6.1 Execution Boundaries

1. **Time-bounded:** 90 minute hard cap, enforced by wall-clock
2. **Single-shot:** One execution per intent, no continuation
3. **Foreground:** No background processes
4. **Observable:** All actions tied to screen state
5. **Human-yielding:** Human input always wins

### 6.2 Restoration Contract

**Always restored:**
- Cursor position (± 5px tolerance)
- Window focus (best-effort fallback to app activation)
- System mode = OBSERVER
- No automated input after restoration

**Never restored:**
- Clipboard contents
- Scroll position
- Application internal state
- Undo/redo history
- Network state

**Termination modes:** All trigger restoration
- Normal completion
- Execution error
- Vision failure
- Authority yield
- Human abort
- Process crash
- Forced termination

### 6.3 Fail-Closed Properties

1. **Restoration always attempted:** Even on process crash (via atexit, signal handlers)
2. **Verification mandatory:** Restoration success must be proven
3. **Failure explicit:** Silent failure prohibited
4. **No retry without intent:** Failure ends execution permanently
5. **Safe shutdown:** Force release all OS resources

### 6.4 Authority Hierarchy (Immutable)

1. Human physical input (immediate yield)
2. Human explicit intent (required to arm)
3. Authority arbitration (yield/proceed per step)
4. Execution Authority System (enforces lifecycle)
5. LLM outputs (planning only, validated)

---

## 7. Data Flow

### 7.1 Planning Phase (LLM ALLOWED)

```
Human Intent
    ↓
ModeController.arm(intent)
    ↓
SnapshotProvider.take_snapshot()
    ↓
ModeController.begin_planning()
    ↓
ExecutionPlanner.create_plan(
    llm_call=mode.get_llm_callable(),  ← LLM CALL HERE
    objective=intent,
    environment=fingerprint,
    screen_context=screenpipe.read()   ← Read-only
)
    ↓
[LLM receives prompt with environment + screen]
    ↓
[LLM returns JSON: List[ExecutionStep]]
    ↓
ExecutionPlan validation (schema + dependencies)
    ↓
ModeController.attach_execution_plan(plan_id)
    ↓
ModeController.mark_planning_complete()
    ↓
Ready for EXECUTING
```

### 7.2 Execution Phase (NO LLM)

```
For each ExecutionStep:
    ↓
  Check dependencies (ProgressTracker)
    ↓
  Authority gate (InputArbitrator) ← Deterministic
    ↓
  Capture before-screen (screenpipe)
    ↓
  Execute step (OperatingSystem backend)
    ↓
  Capture after-screen (screenpipe)
    ↓
  Verify step (StepVerifier) ← Evidence-based, NO LLM
    ↓
  [Success] → ProgressTracker.complete_step()
  [Failure] → FailureRecoveryManager ← Deterministic retry
    ↓
Next step or terminate
```

### 7.3 Restoration Phase (NO LLM)

```
Execution ends (any reason)
    ↓
RestoreProvider.restore_snapshot(snapshot_id)
    ↓
Phase 0: Force release all automation ← Fail-closed
Phase 1: Extended state restore ← Best-effort
Phase 2: Core state restore ← Fail-closed
Phase 3: Force OBSERVER mode ← Mandatory
Phase 4: Verify restoration ← Evidence-based
    ↓
[Success] → Return to OBSERVER
[Failure] → Raise RestorationError, halt system
```

---

## 8. Explicit Non-Goals

### 8.1 Not Autonomous

- Does NOT run continuously in background
- Does NOT auto-trigger on events
- Does NOT invent goals
- Does NOT continue after human intervention
- Requires explicit intent per execution

### 8.2 Not Adaptive

- Does NOT learn from failures
- Does NOT modify plans during execution
- Does NOT adjust strategies
- Plans are frozen after validation

### 8.3 Not Self-Healing

- Does NOT silently retry after fatal errors
- Does NOT hide failures
- Does NOT guess at restoration
- Failures are loud and terminal

### 8.4 Not Intelligent During Execution

- No LLM calls during execution
- No LLM-based verification
- No LLM-based recovery
- No interpretation of evidence

### 8.5 Not Complete Restoration

- Does NOT restore application state
- Does NOT restore clipboard
- Does NOT restore scroll position
- Does NOT rewind user work
- Restores ONLY: cursor, focus, mode, input release

---

## 9. Key Invariants

1. **Brain-Body Separation:** LLM ONLY in planning, NEVER in execution/verification/restoration
2. **Linear Lifecycle:** OBSERVER → ARMED → PLANNING → EXECUTING → OBSERVER (no shortcuts)
3. **Human Supremacy:** Human input always wins, immediate yield
4. **Evidence-Based Verification:** No LLM interpretation, filesystem/process/screen only
5. **Fail-Closed Restoration:** Always attempted, never silent failure
6. **Single-Shot Execution:** One intent = one execution = one restoration
7. **Time-Bounded:** 90 minute hard cap, no grace period
8. **Screen-Grounded:** All actions observable, no hidden processes
9. **Snapshot Requirement:** Pre-execution snapshot mandatory, validated
10. **Authority Enforcement:** No execution without completed plan, no plan without intent

---

## 10. File Organization

```
ProjectZeo-main/
├── main.py                    # Root entry point, lifecycle orchestration
├── core/
│   ├── mode_controller.py     # Authority + lifecycle enforcement
│   ├── planner/
│   │   └── execution_planner.py  # LLM BOUNDARY (planning only)
│   ├── execution/
│   │   ├── progress_tracker.py   # Deterministic tracking
│   │   └── failure_recovery.py   # Deterministic retry
│   ├── verification/
│   │   └── step_verifier.py      # Evidence-based, NO LLM
│   ├── safety/
│   │   ├── runtime_watchdog.py   # Wall-clock enforcement
│   │   └── checkpoint_store.py   # Crash recovery
│   └── schemas/
│       └── execution_plan.py     # Data contracts
├── observer/
│   ├── observer_core.py       # Passive witness, NO LLM
│   ├── perception_engine.py   # Local vision, NO LLM
│   └── screenpipe_adapter.py  # Screen reading
├── authority/
│   ├── authority_policy.py    # Rule-based decisions
│   └── input_arbitrator.py    # Yield logic
├── restoration/
│   ├── snapshot_provider.py   # Pre-execution capture
│   ├── restore_provider.py    # Deterministic restore
│   └── restore_verifier.py    # Evidence-based verification
├── operate/
│   └── operate.py             # Execution loop (NO LLM)
└── docs/
    ├── authority_constitution.md  # Authority laws
    └── restoration_contract.md    # Restoration guarantees
```

---

## 11. System Constraints

### 11.1 Hard Constraints (Non-Negotiable)

- LLM calls FORBIDDEN outside planning phase
- Restoration MANDATORY after every execution
- Verification MANDATORY before step completion
- Human input ALWAYS triggers yield
- Wall-clock timeout ALWAYS enforced
- Snapshot REQUIRED before planning

### 11.2 Implementation Requirements

- Single global lock for restoration (concurrency-safe)
- Deep copy for all observer state (snapshot isolation)
- Idempotent restoration (safe to re-run)
- Explicit error propagation (no silent failures)
- Signal handlers for fail-closed shutdown

### 11.3 Performance Bounds

- LLM call timeout: 30 seconds
- Execution timeout: 90 minutes
- Screen context: 500 characters max
- Cursor tolerance: 5 pixels
- Observer startup grace: 15 seconds or 30 ticks

---

## 12. Comparison to Autonomous Agents

| Aspect | ProjectZeo | Autonomous Agent |
|--------|-----------|------------------|
| Triggering | Explicit human intent | Auto-triggered on events |
| Execution | Single-shot, bounded | Continuous, unbounded |
| Planning | LLM during planning only | LLM during execution |
| Adaptation | Fixed plan | Dynamic replanning |
| Restoration | Always, mandatory | Often none |
| Failure | Explicit, terminal | Silent retry |
| Authority | Human > System | System autonomous |
| Observation | Passive witness | Active learning |
| Verification | Evidence-based | LLM-based |

---

## 13. Risk Mitigations

### 13.1 Runaway Execution

**Risk:** System executes beyond intended scope  
**Mitigation:** 90 minute hard timeout, human input yield, single-shot execution

### 13.2 Unverified Actions

**Risk:** Actions succeed but not verified  
**Mitigation:** Mandatory verification per step, evidence-based (no trust)

### 13.3 Failed Restoration

**Risk:** System leaves workspace in unknown state  
**Mitigation:** Fail-closed restoration, mandatory verification, explicit failure

### 13.4 Vision Loss

**Risk:** Execution without perception  
**Mitigation:** Observer health monitoring, blindness errors, execution abort

### 13.5 Authority Confusion

**Risk:** Ambiguous control between human and system  
**Mitigation:** Linear lifecycle, authority hierarchy, human supremacy clause

### 13.6 Silent Failure

**Risk:** Errors hidden or ignored  
**Mitigation:** Explicit error propagation, audit journal, no silent retry

---

## 14. Evolution Path

### 14.1 Allowed Changes

- Add new StepType enums (if verifiable)
- Extend environment fingerprint
- Add optional extended restoration state
- Improve evidence collection
- Add deterministic recovery strategies

### 14.2 Forbidden Changes

- LLM calls during execution
- Remove mandatory restoration
- Remove verification requirements
- Weaken authority hierarchy
- Add autonomous continuation
- Silent failure modes

### 14.3 Amendment Process

Any change to core guarantees requires:
1. Version bump
2. Backward compatibility review
3. Re-verification of restoration contract
4. Re-verification of authority constitution
5. Explicit documentation of safety impact

---

## 15. Summary

ProjectZeo is a **deterministic execution body** controlled by an **external LLM brain** during planning only. It operates a real OS through observable, time-bounded, single-shot executions that always restore to a known state. Intelligence exists ONLY in the planning phase; all execution, verification, and restoration are deterministic and evidence-based.

**Core Identity:**
- Body: Deterministic OS controller with fail-closed restoration
- Brain: External LLM used ONLY for planning
- Authority: Human > System, always
- Execution: Single-shot, time-bounded, observable
- Verification: Evidence-based, no interpretation
- Restoration: Always attempted, always verified

**Not:**
- Autonomous agent
- Continuous background process
- Self-learning system
- Adaptive executor
- Complete state restorer

---

## 16. Audit Findings — Known Architectural Gaps

This section documents confirmed deviations between stated architecture and runtime
behaviour. All items below have been reported via the 2026-02-28 adversarial audit.
Fix status is noted; gaps pending future work are marked **OPEN**.

### 16.1 ReasoningEngine — Unreachable in Normal Execution (OPEN)

**Component:** `core/cognition/reasoning_engine.py` / `operate/operate.py`

**Stated Behaviour:**  
Section 5.3 (Execution Layer) implies that `ReasoningEngine.propose_actions()` provides
dynamic candidate injection to recover stagnant tasks without triggering a full REPLAN.

**Actual Behaviour:**  
`propose_actions()` is only called in `_execute_autonomous_loop()` when `candidate_actions`
is empty.  `candidate_actions` is populated from `current_step.action`, which is always
non-empty (ExecutionPlanner guarantees a non-null action dict per step, and schema
validation rejects empty actions).  Therefore `ReasoningEngine` is never reached during
normal plan execution.

`ReasoningEngine` is only reachable if a step's action dict is null or empty — a condition
that plan schema validation prevents at planning time.

**Impact:**  
Stagnation can only be escaped via REPLAN (up to MAX_REPLANS=3), then TASK_FAILED.
Dynamic candidate injection — a stated stagnation-recovery feature — is inoperative.

**Condition for reachability:**  
`ReasoningEngine` would be reached if `ExecutionPlanner` produced a step with
`action = {}` or `action = None`.  This currently cannot happen due to schema validation.
If future plan sources (e.g., user-injected steps) bypass schema validation, the
ReasoningEngine path would become live.

**Fix status:** OPEN — architecture documentation updated (this section).  A future fix
would either (a) remove `ReasoningEngine` as dead code and document stagnation as REPLAN-
only, or (b) change the trigger condition so `ReasoningEngine` fires after N consecutive
stagnant iterations regardless of `candidate_actions` emptiness.

---

### 16.2 Restoration Scope — Cosmetic Only

**Stated behaviour:**  
Sections 6.2 and 6.3 describe restoration as fail-closed and mandatory.

**Actual behaviour:**  
Restoration is cosmetic: it restores **cursor position and window focus only**.  No OS
state mutated during task execution (spawned processes, open files, clipboard, network
connections) is restored.  This matches `restoration_contract.md` Section 5 (explicit
non-guarantees) but may conflict with implicit operator expectations.

**Fix status:** RESOLVED by documentation.  `RestorationSnapshot.to_dict()` now includes
an explicit `"restoration_scope": "cursor_and_focus_only"` field and a
`"restoration_not_restored"` list in the serialised snapshot for audit consumers.

---

### 16.3 Human Confirmation Timeout — Configurable (RESOLVED)

**Original defect:**  
`MAX_WAIT_RETRIES=10 × WAIT_RETRY_SECONDS=0.5` = 5 seconds.  Human confirmation via
`/tmp` signal file within 5 seconds was operationally unreachable.

**Fix:**  
Timeout is now configurable via `PROJECTZEO_CONFIRM_TIMEOUT_SECONDS` env var (default 60s).
See `operate/operate.py` `_resolve_confirm_timeout()`.

---

### 16.4 Title Distance Asymmetry — Resolved

**Original defect:**  
`RestoreProvider.MAX_TITLE_DISTANCE=5` vs `RestoreVerifier.MAX_TITLE_DISTANCE=2` caused
false-positive safe-shutdown on any window title drift of 3–5 characters.

**Fix:**  
Both now use `MAX_TITLE_DISTANCE=5` (aligned).  Both delegate to the shared
`levenshtein_distance()` and `title_match()` utility functions in `restoration/snapshot_types.py`
to prevent future divergence (H7 fix).

