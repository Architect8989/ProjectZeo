# 🛡️ Restoration Contract — Execution Authority System

> **Status:** FROZEN  
> **Audience:** Systems engineers, platform architects, auditors  
> **Purpose:** Define *exactly* what is restored, when, how, and how success is verified  
> **Non-Goal:** No intelligence, UX, or autonomy claims

---

## 🔭 1. Scope

This document defines the **binding restoration guarantees** of the Execution Authority System after **any execution attempt**, including abnormal or hostile termination.

This contract applies to:
- All execution paths
- All termination modes
- All future implementations

If an implementation cannot satisfy this contract, the implementation is invalid.

---

## 📘 2. Definitions

| Term | Definition |
|---|---|
| **Execution** | Period during which the system emits OS input |
| **Pre-Hijack State** | Workspace state captured immediately before execution |
| **Restoration** | Process of returning workspace to an acceptable state |
| **Human Intervention** | Any human-initiated input during execution |
| **Termination Mode** | Reason execution stopped |

---

## 🧠 3. System Architecture (High Level)

┌───────────────────────┐ │   👁 Observer Layer    │ │  (Vision + Witness)   │ └──────────┬────────────┘ │ ▼ ┌───────────────────────┐ │ 🧭 Authority Layer     │ │  - Arbitration         │ │  - Policy              │ │  - Yield / Abort       │ └──────────┬────────────┘ │ ▼ ┌───────────────────────┐ │ 🤖 SOC (Sealed Engine) │ │  - See → Decide → Act  │ │  - Screen as API       │ └──────────┬────────────┘ │ ▼ ┌───────────────────────┐ │ ♻️ Restoration Engine  │ │  (THIS CONTRACT)       │ └───────────────────────┘

**Key invariant:**  
> Restoration sits **outside** SOC and **after** authority resolution.

---

## ✅ 4. Guaranteed State (MUST RESTORE)

The system guarantees restoration of the following **minimum viable workspace state**:

### 🎯 4.1 Input & Focus
- 🖱 Cursor position (screen coordinates)
- 🪟 Foreground window focus
- 🧩 Active application process (best identifiable match)
- ⌨️ Keyboard modality enabled (no stuck modifiers)

### 🔐 4.2 System Control
- Execution mode reverted to `OBSERVER`
- No automated input after restoration completes

These guarantees apply **regardless of termination mode**, unless physically impossible (e.g., power loss).

---

## ❌ 5. Explicit Non-Guarantees (NOT RESTORED)

The system **does not** guarantee restoration of:

- 📋 Clipboard contents
- 📜 Scroll position
- 🎞️ UI animations / transitions
- 🌐 Network state
- 🧠 Application internal state
- ↩️ Undo / redo history

Anything not listed in Section 4 is **explicitly out of scope**.

---

## 🧨 6. Termination Modes

The system recognizes the following termination modes:

NORMAL_COMPLETION EXECUTION_ERROR VISION_FAILURE AUTHORITY_YIELD HUMAN_ABORT PROCESS_CRASH FORCED_TERMINATION (SIGKILL / power loss)

All termination modes MUST attempt restoration except where process death makes it impossible.

---

## 🔁 7. Restoration Order (MANDATORY)

Restoration MUST occur in the following order:

1️⃣ Cease all automated input immediately 2️⃣ Reassert keyboard/mouse availability 3️⃣ Restore cursor position 4️⃣ Restore window focus 5️⃣ Restore active application 6️⃣ Transition to OBSERVER mode

Deviation is **not permitted**.

---

## 🧪 8. Verification Criteria

Restoration is **successful** if and only if:

- 🖱 Cursor position matches pre-hijack position (± tolerance)
- 🪟 A valid window has focus
- 👁 System mode == `OBSERVER`
- 🚫 No further automated input occurs

Verification is **mandatory**.

---

## 🚨 9. Failure Semantics

If restoration cannot be verified:

- ⛔ Execution is permanently halted
- 🧾 Failure artifact is emitted
- 👁 System remains in `OBSERVER`
- 🔁 No automatic retry allowed

**Silent failure is prohibited.**

---

## 🧍 10. Human Intervention Semantics

If human input occurs:

- ♻️ Restoration MUST still be attempted
- ✋ Human input MUST NOT be overridden
- 🧠 Restoration adapts to current visible state

> The system never fights the human.

---

## ♻️ 11. Idempotency Requirements

Restoration logic MUST be:

- 🔁 Safe to re-run
- 🧩 Safe if partially applied
- ⚠️ Safe if interrupted

Repeated attempts MUST NOT degrade workspace state.

---

## 🗂️ 12. Data Schemas

### 📦 12.1 Pre-Hijack Snapshot Schema

```json
{
  "snapshot_id": "uuid",
  "timestamp": "epoch_ms",
  "cursor": { "x": 0, "y": 0 },
  "focused_window": "window_id",
  "active_app": "process_name",
  "execution_mode": "OBSERVER"
}

🧾 12.2 Restoration Result Schema

{
  "snapshot_id": "uuid",
  "restoration_attempted": true,
  "verified": true,
  "failure_reason": null,
  "timestamp": "epoch_ms"
}

🧩 13. State Transition Diagram

[ OBSERVER ]
     |
     | intent
     v
[ EXECUTING ]
     |
     | success / failure / yield / crash
     v
[ RESTORING ]
     |
     | verified
     v
[ OBSERVER ]

No other transitions are allowed.

🚫 14. Non-Goals

This system does NOT aim to:

Rewind application data

Recover unsaved user work

Enforce pixel-perfect layouts

Bypass OS security boundaries

🔒 15. Contract Status

This document is frozen.

Changes require:

Version bump

Backward compatibility review

Re-verification of all implementations

No code may violate this contract.


---

## 16. Audit Findings — Restoration Reality (2026-02-28)

This section documents confirmed gaps between the restoration contract and runtime
behaviour as of the 2026-02-28 adversarial audit.

### 16.1 Scope Reaffirmation

The contract in Section 4 (Guaranteed State) and Section 5 (Explicit Non-Guarantees)
is accurate.  **Restoration is cursor-position and window-focus only.**  No spawned
processes, file changes, clipboard, or network state is restored.  This is not a defect;
it is the stated and intentional scope.

Operators who require full-state rollback should run ProjectZeo inside a VM or container
with snapshotting (e.g. QEMU/KVM with `virsh snapshot-create-as`, or Docker with
`docker commit`).

### 16.2 Process Census Diff (IH-6)

**Status:** FIXED (2026-02-28, SI-A fix).

The process census diff (reporting processes spawned during task execution that were not
terminated by restoration) is now operational.  The original implementation read from
`snapshot.metadata["process_census_pids"]` — a key that was never written anywhere in the
codebase.  The fix corrects the key path to `metadata["extended"]["processes"]` (process
names) and `metadata["process_census_pids"]` (PIDs captured in `snapshot_types.create()`).

Post-restoration, `RestoreProvider._report_unrestored_processes()` now emits a WARNING
listing any process names present after restoration that were not present at snapshot time.

### 16.3 Secondary Verification Skip on Snapshot TTL Expiry (RT-D)

**Status:** FIXED (2026-02-28, RT-D fix).

When `snapshot_provider.get_snapshot()` returns `None` due to TTL expiry (3-hour limit),
`restore_verifier.verify()` was previously skipped silently and `restore_required=False`
was written to `auth_state` — signalling a clean exit without having verified restoration.

The fix adds an explicit WARNING log and sets `auth_state.verification_warning = True`
when the skip occurs, so the unverified restoration is visible in audit records.

### 16.4 ReasoningEngine and Stagnation Recovery

**Note:** `ReasoningEngine.propose_actions()` is not called during normal plan execution.
Stagnation recovery occurs exclusively via REPLAN (up to `MAX_REPLANS=3` attempts),
then `TASK_FAILED`.  See `ARCHITECTURE.md` Section 16.1 for the full analysis.

This does not affect the restoration contract directly, but operators should be aware
that stagnant tasks terminate via REPLAN → TASK_FAILED, not via dynamic recovery.
Restoration is still attempted for all termination modes including TASK_FAILED.
