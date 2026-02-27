# 🛡️ Restoration Scope Declaration — Execution Authority System

> **Status:** ACTIVE — v3.0  
> **Audience:** Systems engineers, platform architects, auditors  
> **Purpose:** Define *exactly* what is and is NOT restored, when, how, and how success is verified  
> **Non-Goal:** No intelligence, UX, or autonomy claims  
>
> **H-3 NOTE:** This document was previously titled "Restoration Contract".
> The term "contract" implied enforceable guarantees beyond what is actually delivered.
> The actual restoration guarantee is **cursor position + window focus + automation release only**.
> All other state (file system, browser, clipboard, processes) persists from task execution.

---

## ⚠️ RESTORATION IS SHALLOW — READ THIS FIRST

**CRITICAL: Restoration does NOT roll back OS state.**

When the system reports "restoration successful" it means:
- The cursor was moved back to its pre-task position (±5px)
- Window focus was returned to the pre-task window
- Automated input was released

**Everything else — files written, processes spawned, clipboard changed, browsers navigated, packages installed — persists.**

For full OS state rollback, run ProjectZeo inside a container, VM with snapshots, or a filesystem with copy-on-write support (btrfs/ZFS).

---

## ✅ 4. Guaranteed State (WILL BE RESTORED)

- 🖱 Cursor position (±5px tolerance)
- 🪟 Foreground window focus (Levenshtein ≤2 on title)
- 🧩 Active application (best match by title)
- ⌨️ Keyboard modifier keys released (Ctrl, Shift, Alt, Win/Cmd)
- 🔐 Execution mode reverted to `OBSERVER`
- 🚫 No automated input after restoration completes

---

## ❌ 5. Explicit Non-Guarantees (NOT RESTORED)

**H-1 FIX:** This section is authoritative and exhaustive. Any state not listed in Section 4 is NOT restored.

### 5.1 File System
- Files created, modified, or deleted during execution
- Package installations (apt, pip, npm, etc.)
- Configuration file modifications

### 5.2 Application State
- Browser URL / tabs / form contents / scroll position
- Clipboard contents
- Undo / redo history
- Application internal state (unsaved work)

### 5.3 System State
- Network connections
- Running processes (spawned child processes remain running)
- Window geometry (soft-verified only, not restored)
- Window Z-order (stub verification, not implemented)
- System registry / OS configuration changes

### 5.4 Partial Restoration (Best-Effort)
- `keyboard_modifiers_partially` — modifier keys (Ctrl, Shift, Alt, Win) are released via `force_release_all()`. Non-modifier key state is NOT restored.

---

## 🔁 7. Restoration Order (MANDATORY)

1. Cease all automated input immediately
2. Reassert keyboard/mouse availability (release modifier keys)
3. Restore cursor position
4. Restore window focus
5. Restore active application
6. Transition to OBSERVER mode

---

## 🧪 8. Verification Criteria

| Check | Tolerance |
|---|---|
| Cursor position | ±5px |
| Window focus title | Levenshtein ≤2 edits |
| System mode | must be OBSERVER |
| Automation released | must be False |

Extended checks (geometry, z-order, browser, media) are **best-effort soft checks** only.

---

## 🚨 9. Failure Semantics

If restoration cannot be verified: execution halted, failure recorded, system stays in OBSERVER, no automatic retry. Silent failure is prohibited.

---

## 🔒 12. Commitment Chain Trust Boundary

**H-7 FIX:** The `commitment_chain_hash` in `BeliefState` is a SHA-256 Merkle audit trail for intra-process session integrity. It is NOT tamper-proof against a compromised process — the hash lives in process memory with no external root of trust. Use it for forensic audit of uncompromised sessions only.

---

## 📜 Version History

| Version | Change |
|---|---|
| v1.0 | Initial |
| v2.0 | Extended verification stubs |
| v3.0 | **H-1/H-3:** Explicit non-restoration list. Title changed from "Restoration Contract" to "Restoration Scope Declaration". Keyboard modifiers partial-restoration documented. Trust boundary section added. |

---

## 📋 IH-6 ADDENDUM — Explicit Scope Declaration

**IH-6 FIX:** This section addresses the gap where the detailed scope was declared only in
`snapshot_types.py:to_dict()` (code comments) but not in this contract document.

### What "Restoration Successful" Means — Precise Definition

Restoration is **workspace aesthetics only**. The system:

1. **Moves the cursor** back to its pre-task XY position (within ±5px).
2. **Re-focuses the window** that was active before the task (fuzzy title match, Levenshtein ≤2).
3. **Re-activates the application** that was active before the task (best title match).
4. **Releases all automated input** (keyboard modifiers, mouse buttons).
5. **Resets execution mode** to OBSERVER.

### What "Restoration Successful" Does NOT Mean

- ❌ Files or directories created, modified, or deleted are **not reversed**.
- ❌ Processes spawned during the task are **not terminated**.
- ❌ Browser history, URLs, tabs, or form state are **not restored**.
- ❌ Clipboard contents are **not restored**.
- ❌ Network connections or downloads are **not reversed**.
- ❌ Package installations (apt, pip, npm, brew) are **not uninstalled**.
- ❌ Window geometry (position and size) is captured and soft-verified but **not hard-required**.
- ❌ Window Z-order (stacking) is captured (as of v3.1) and verified only if `xdotool` is available.

### Recommendation

For tasks requiring full OS state rollback, run ProjectZeo inside:
- A container with ephemeral storage
- A VM with snapshot-before/rollback-after
- A filesystem with copy-on-write support (btrfs, ZFS)

