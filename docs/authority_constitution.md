# Authority Constitution — Execution Authority System

## 1. Purpose

This constitution defines the **non-negotiable authority, control, and failure laws**
governing the Execution Authority System.

Its purpose is to ensure that:
- Human sovereignty is absolute
- Machine execution is bounded
- Failure is safe
- Responsibility is explicit
- No future feature can weaken these guarantees

This document supersedes all informal design intent.

---

## 2. Authority Hierarchy

Authority within the system SHALL always resolve in the following order:

1. Human physical input
2. Human explicit intent
3. Authority arbitration layer
4. Execution Authority System
5. Reasoning or model outputs

No lower authority may override a higher authority under any circumstance.

---

## 3. Execution Preconditions

Execution MUST NOT begin unless ALL of the following are true:

- The system is in `OBSERVER` mode
- Live visual perception is available and fresh
- A valid pre-hijack snapshot has been captured
- Human authority has not been revoked or yielded
- No unresolved failure state exists

If any precondition is violated, execution SHALL NOT occur.

---

## 4. Human Supremacy Clause

Human input SHALL always supersede machine input.

If human input is detected during execution:
- Machine execution MUST yield immediately
- No attempt may be made to override or negate human action
- Execution authority SHALL be relinquished

The system MUST never fight human control.

---

## 5. Execution Boundaries

The system SHALL only execute:
- Explicitly declared actions
- Within the bounds of observed environment state
- Without hidden background processes
- Without implicit continuation

The system MUST NOT:
- Invent goals
- Continue execution without authority
- Operate without perception
- Execute actions it cannot observe

---

## 6. Yield Semantics

Yielding execution is a mandatory safety mechanism.

Execution MUST yield when:
- Human input is detected
- Authority arbitration requires yield
- Perception becomes unreliable
- Execution confidence collapses
- Any invariant is violated

Yield MUST be immediate and irreversible for the current execution cycle.

---

## 7. Failure Semantics

Failure SHALL be explicit, loud, and terminal for the current execution.

On failure:
- Execution MUST stop
- No retries may occur without new intent
- Failure MUST be recorded
- The system MUST transition toward restoration

Silent failure is prohibited.

---

## 8. Restoration Mandate

After any execution attempt, regardless of outcome:

- Automated input MUST cease
- Human input MUST be enabled
- The workspace MUST be restored to the pre-hijack state within contract bounds
- Restoration MUST be verified

If restoration cannot be verified:
- Execution SHALL remain halted
- The system SHALL remain in `OBSERVER` mode
- No further execution may occur without explicit human intent

---

## 9. Non-Delegable Invariants

The following invariants SHALL NOT be delegated, bypassed, or relaxed:

- Human authority supremacy
- Pre-hijack snapshot requirement
- Restoration requirement
- Verification requirement
- Explicit execution modes
- Observable execution only

Any implementation violating these invariants is invalid.

---

## 10. Explicit Prohibitions

The system MUST NEVER:

- Execute without perception
- Execute without a snapshot
- Hide execution state
- Conceal failures
- Continue after authority loss
- Restore by guessing
- Assume human intent
- Operate autonomously by default

These prohibitions are absolute.

---

## 11. Amendment Rules

This constitution may only be amended if ALL of the following are true:

- The amendment is explicit and written
- Backward compatibility is preserved
- Existing guarantees are not weakened
- Restoration and authority laws remain intact
- The amendment is versioned and reviewed

No amendment may retroactively legitimize prohibited behavior.

---

## 12. Constitution Status

This document is **frozen**.

All current and future implementations MUST conform to this constitution.

If code conflicts with this constitution, the code is wrong.
