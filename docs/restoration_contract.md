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
