<div align="center">

# ⚡ ProjectZeo

### A Deterministic LLM-Driven Autonomous Kernel for Live OS Execution

**Local-first · Vision-powered · Pure autonomy · No scripted workflows**

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20Local-green?style=flat-square)](https://ollama.ai)
[![Model](https://img.shields.io/badge/Vision-Qwen2.5--VL-orange?style=flat-square)](https://ollama.ai/library/qwen2.5-vl)
[![License](https://img.shields.io/badge/License-See%20LICENSE-lightgrey?style=flat-square)](./LICENSE)

</div>

---

## What Is ProjectZeo?

ProjectZeo is an autonomous kernel that assigns a local vision LLM as the "brain" of your computer. You give it a task — it figures out everything else. No pre-installed tools required. No scripted workflows. No sandboxes. It operates your **live OS** exactly the way a human would.

**The core promise:** Drop the system onto a raw OS with only a browser and terminal. Assign it a 20-tool hackathon project. It browses official sites, downloads installers, configures environments, writes code, runs servers — all driven by the LLM watching the screen, deciding the next action, and executing it.

---

## Table of Contents

- [Core Philosophy](#core-philosophy)
- [How It Actually Works](#how-it-actually-works)
- [Architecture Overview](#architecture-overview)
- [Mode Lifecycle](#mode-lifecycle-state-machine)
- [Component Map](#component-map)
- [LLM Integration](#llm-integration--ollama-wiring)
- [Observer System](#observer-system)
- [Autonomous Installer](#autonomous-installer)
- [Snapshot & Restoration](#snapshot--restoration-contract)
- [Authority & Safety](#authority--safety-layer)
- [Data Flow Diagrams](#data-flow-diagrams)
- [File Structure](#file-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Known Limitations](#known-limitations)

---

## Core Philosophy

> **"Intelligence lives in the LLM. Execution is deterministic."**

This is the single rule the entire system is built around.

| What | Who Does It |
|------|-------------|
| Understand the screen | Vision LLM (Qwen2.5-VL via Ollama) |
| Decide what to do next | Vision LLM |
| Plan the steps | Vision LLM |
| Execute mouse/keyboard actions | Deterministic OS backend |
| Verify completion | Deterministic evidence check |
| Install missing tools | Autonomous browser-based installer (LLM-guided) |
| Restore screen state after task | Deterministic restore engine |

The LLM is never called during execution, verification, or restoration. It is only called during the **planning phase** and to **guide the installer**. Everything else is pure code.

---

## How It Actually Works

```
You type a task (e.g., "Build a Node.js + React app with a PostgreSQL backend")
                             │
                             ▼
        System takes a snapshot of your current screen state
                             │
                             ▼
        LLM looks at screen + environment fingerprint
        Figures out: what OS, what tools exist, what's missing
        Plans: step-by-step execution path using real environment
                             │
                             ▼
        Execution begins on LIVE OS — no sandbox
        If tool X is missing → browser opens → navigates to official site
        → LLM watches screen → clicks download → installs → verifies
                             │
                             ▼
        Task completes (or fails explicitly)
                             │
                             ▼
        Screen restored to exact state before task started
```

**The hostile environment scenario** (what this is built for):

```
Raw OS: only browser + terminal installed
Task:   Build a hackathon project using Node.js, React, Express,
        PostgreSQL, Redis, Nginx, Docker, and 15 other tools

System response:
  1. Fingerprint environment → "node: not found, npm: not found, ..."
  2. Plan: install each tool from official source
  3. Open browser → navigate to nodejs.org → download installer
  4. Watch screen → click through install wizard
  5. Verify: `node --version` returns value
  6. Continue with next tool...
  7. Clone/create project, write code, configure, run
```

This is not simulated. It runs on your actual screen.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROJECTZEO KERNEL                                 │
│                                                                             │
│  ┌──────────────┐      ┌─────────────────────────────────────────────────┐ │
│  │   run.py     │      │              MAIN LOOP (main.py)                │ │
│  │  Entry point │─────▶│  Orchestrates lifecycle, signals, heartbeat     │ │
│  │              │      └───────────────────────┬─────────────────────────┘ │
│  └──────────────┘                              │                            │
│                                                │                            │
│  ┌─────────────────────────────────────────────▼─────────────────────────┐ │
│  │                        ADAPTER LAYER                                  │ │
│  │  ┌────────────────────────────────────────────────────────────────┐   │ │
│  │  │  adapters/factory.py  →  adapters/qwen_ollama_adapter.py       │   │ │
│  │  │  Resolves model name → Builds QwenOllamaAdapter                │   │ │
│  │  │  Wraps async get_next_action() into sync callable              │   │ │
│  │  │  Enforces: temperature=0, bounded timeout, no cloud            │   │ │
│  │  └────────────────────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌──────────────────────────┐   ┌──────────────────────────────────────┐  │
│  │    OBSERVER SYSTEM       │   │         MODE CONTROLLER              │  │
│  │                          │   │                                      │  │
│  │  VisionRuntime           │   │  OBSERVER → ARMED → PLANNING         │  │
│  │  (Ollama Qwen2.5-VL)    │   │  → EXECUTING → RESTORING → OBSERVER  │  │
│  │  Captures screen         │   │                                      │  │
│  │  Returns structured UI   │   │  Enforces transition rules           │  │
│  │                          │   │  Guards snapshot contract            │  │
│  │  ObserverLoop            │   │  Controls LLM access                 │  │
│  │  5 Hz continuous watch   │   │  Logs every transition               │  │
│  │                          │   │                                      │  │
│  │  ObserverCore            │   │  Single source of truth              │  │
│  │  Passive witness         │   │  Thread-safe (RLock)                 │  │
│  │  Builds world graph      │   └──────────────────────────────────────┘  │
│  │  Detects blindness       │                                              │
│  └──────────────────────────┘                                              │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    EXECUTION PIPELINE                                │  │
│  │                                                                      │  │
│  │  ┌─────────────────┐   ┌────────────────┐   ┌───────────────────┐   │  │
│  │  │ ExecutionPlanner│   │  operate.py    │   │ AutonomousInstall │   │  │
│  │  │                 │   │                │   │                   │   │  │
│  │  │ LLM call here   │   │ Autonomous     │   │ Browser-based     │   │  │
│  │  │ Produces plan   │   │ execution loop │   │ tool installation │   │  │
│  │  │ Validates steps │   │ BeliefState    │   │ LLM-guided UI     │   │  │
│  │  │ NO execution    │   │ ActionRanker   │   │ Official sources  │   │  │
│  │  └─────────────────┘   └────────────────┘   └───────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────┐   ┌────────────────────────────────────────┐  │
│  │  RESTORATION ENGINE     │   │         OS BACKEND                     │  │
│  │                         │   │                                        │  │
│  │  SnapshotProvider       │   │  OperatingSystem (pyautogui)           │  │
│  │  (before-task capture)  │   │  click, type, press, exec, write       │  │
│  │                         │   │  get_cursor, get_window, activate_app  │  │
│  │  RestoreProvider        │   │  force_release_all (safety)            │  │
│  │  (after-task restore)   │   │  Heartbeat watchdog                    │  │
│  │  Verifies success       │   └────────────────────────────────────────┘  │
│  └─────────────────────────┘                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────┐
                    │    Ollama (Local)            │
                    │    Model: qwen2.5-vl:7b      │
                    │    Vision + Language         │
                    │    temperature=0             │
                    │    No internet required      │
                    └─────────────────────────────┘
```

---

## Mode Lifecycle (State Machine)

The system has exactly **5 modes**. Transitions are strictly enforced — no shortcuts, no skips.

```
                    ┌──────────────────────────────────────┐
                    │                                      │
                    ▼                                      │
          ┌─────────────────┐                             │
          │                 │                             │
          │    OBSERVER     │ ◀── System starts here      │
          │                 │                             │
          │ • LLM watches   │                             │
          │   screen 5 Hz   │                             │
          │ • Builds world  │                             │
          │   graph         │                             │
          │ • Waits for     │                             │
          │   user task     │                             │
          │ • No actions    │                             │
          │                 │                             │
          └────────┬────────┘                             │
                   │                                      │
           User submits task                              │
           (stdin or file)                                │
                   │                                      │
                   ▼                                      │
          ┌─────────────────┐                             │
          │                 │                             │
          │     ARMED       │                             │
          │                 │                             │
          │ • Intent frozen │                             │
          │ • Screen state  │                             │
          │   snapshotted   │                             │
          │ • Snapshot ID   │                             │
          │   generated     │                             │
          │                 │                             │
          └────────┬────────┘                             │
                   │                                      │
           Vision health confirmed                        │
           Observer healthy                               │
                   │                                      │
                   ▼                                      │
          ┌─────────────────┐                             │
          │                 │                             │
          │   PLANNING 🧠   │ ◀── ONLY LLM ZONE          │
          │                 │                             │
          │ • LLM receives: │                             │
          │   - Task intent │                             │
          │   - Env profile │                             │
          │   - Screen state│                             │
          │ • LLM produces: │                             │
          │   - Typed steps │                             │
          │   - Actions     │                             │
          │   - Verifications│                            │
          │ • 60s hard limit│                             │
          │                 │                             │
          └────────┬────────┘                             │
                   │                                      │
           Plan validated &                               │
           attached                                       │
                   │                                      │
                   ▼                                      │
          ┌─────────────────┐                             │
          │                 │                             │
          │   EXECUTING ⚙️  │                             │
          │                 │                             │
          │ • Step-by-step  │                             │
          │   execution     │                             │
          │ • Mouse/keyboard│                             │
          │   actions       │                             │
          │ • Install tools │                             │
          │   via browser   │                             │
          │ • Verify each   │                             │
          │   step          │                             │
          │ • 90min wall    │                             │
          │   clock limit   │                             │
          │                 │                             │
          └────────┬────────┘                             │
                   │                                      │
         Task done / failed /                             │
         timed out / user aborted                         │
                   │                                      │
                   ▼                                      │
          ┌─────────────────┐                             │
          │                 │                             │
          │   RESTORING ♻️  │                             │
          │                 │                             │
          │ • Stop all input│                             │
          │ • Restore cursor│                             │
          │ • Restore window│                             │
          │ • Restore app   │                             │
          │ • Verify success│                             │
          │                 │                             │
          └────────┬────────┘                             │
                   │                                      │
           Verified ──────────────────────────────────────┘
```

### Transition Rules (Hard Enforced by ModeController)

| From | To | Required Conditions |
|------|----|---------------------|
| OBSERVER | ARMED | Snapshot taken + Intent non-empty |
| ARMED | PLANNING | Observer healthy + Vision available |
| PLANNING | EXECUTING | Plan attached + Planning marked complete + Vision OK |
| EXECUTING | RESTORING | Always (any execution end) |
| RESTORING | OBSERVER | Restoration verified |

Any violation throws `ModeTransitionError` and halts execution.

---

## Component Map

```
ProjectZeo-main/
│
├── run.py                          ← Entry point. Resolves model, builds adapter,
│                                     wraps LLM, calls main()
│
├── main.py                         ← Kernel orchestrator. Lifecycle, signal handlers,
│                                     warmup, main loop, replan logic
│
├── adapters/
│   ├── factory.py                  ← Model registry + dynamic import
│   ├── qwen_ollama_adapter.py      ← LOCAL LLM. Ollama client, vision capture,
│   │                                  OCR coord resolution, JSON parsing
│   ├── apis_safety_layer.py        ← Patches legacy APIs: blocks cloud fallbacks,
│   │                                  enforces temperature=0, blocks screenshot writes
│   └── pure_llm_wrapper.py         ← Cloud wrapper (unused in Ollama path)
│
├── core/
│   ├── mode_controller.py          ← State machine. THE authority. Enforces all
│   │                                  mode transitions. Thread-safe. Logs to JSONL.
│   ├── intent_listener.py          ← Polls stdin + /tmp/projectzeo.intent for tasks
│   ├── environment_fingerprint.py  ← Read-only OS scan. Never crashes, never executes.
│   │
│   ├── planner/
│   │   ├── execution_planner.py    ← LLM BOUNDARY. Only component that calls LLM.
│   │   │                              Produces ExecutionPlan. Hard timeout enforced.
│   │   ├── task_planner.py         ← High-level task decomposition
│   │   ├── task_decomposer.py      ← Breaks complex tasks into sub-goals
│   │   └── __init__.py
│   │
│   ├── cognition/
│   │   ├── belief_state.py         ← Bayesian belief tracker for execution confidence
│   │   ├── action_ranker.py        ← Ranks candidate actions by belief state
│   │   └── reasoning_engine.py     ← Sanitizes/normalizes LLM outputs
│   │
│   ├── execution/
│   │   ├── progress_tracker.py     ← Tracks step completion deterministically
│   │   └── failure_recovery.py     ← Retry logic, stagnation detection
│   │
│   ├── verification/
│   │   ├── step_verifier.py        ← Evidence-based step verification (no LLM)
│   │   ├── plan_verifier.py        ← Validates ExecutionPlan structure
│   │   ├── screen_verifier.py      ← Screen hash comparison
│   │   └── task_validator.py       ← Task-level completion check
│   │
│   ├── vision/
│   │   ├── vision_runtime.py       ← Ollama screen capture loop. PIL ImageGrab.
│   │   ├── world_graph.py          ← Structured representation of screen entities
│   │   └── semantic_resolver.py    ← Maps LLM descriptions to screen coordinates
│   │
│   ├── safety/
│   │   ├── action_timeout.py       ← Per-action timeout context manager
│   │   ├── checkpoint_store.py     ← Crash-safe execution state persistence
│   │   ├── restart_guard.py        ← Detects crash + forces restoration on restart
│   │   └── runtime_watchdog.py     ← Wall-clock enforcement thread
│   │
│   ├── memory/
│   │   └── playbook_store.py       ← Stores successful task patterns
│   │
│   ├── schemas/
│   │   └── execution_plan.py       ← ExecutionPlan + ExecutionStep dataclasses
│   │
│   └── tools/
│       ├── autonomous_installer.py ← Browser-based tool installer. LLM watches
│       │                              screen, clicks through installer UI.
│       └── tool_manager.py         ← Tool availability tracking
│
├── observer/
│   ├── observer_core.py            ← Passive witness. Deep-copy snapshots.
│   │                                  Blindness detection. ZERO execution authority.
│   ├── observer_loop.py            ← 5 Hz daemon. Feeds ObserverCore + WorldGraph.
│   ├── perception_engine.py        ← Processes raw screen data into UI elements
│   ├── self_healing.py             ← Observer health recovery
│   └── ui_schema.py                ← UI element type definitions
│
├── operate/
│   ├── operate.py                  ← Autonomous execution loop. BeliefState.
│   │                                  ActionRanker. Per-step execution + verify.
│   ├── config.py                   ← Runtime configuration
│   ├── exceptions.py               ← Custom exceptions
│   │
│   ├── models/
│   │   ├── apis_openrouter.py      ← Cloud path (OpenRouter). Not used for local.
│   │   ├── prompts.py              ← System/user prompt templates
│   │   └── weights/best.pt         ← YOLO weights for UI element detection
│   │
│   ├── legacy/
│   │   └── apis.py                 ← Original multi-provider API handlers
│   │
│   └── utils/
│       ├── operating_system.py     ← OS boundary. pyautogui wrapper. Heartbeat
│       │                              watchdog. cursor, window, app management.
│       ├── screenshot.py           ← Screen capture utilities
│       ├── ocr.py                  ← EasyOCR text coordinate resolution
│       ├── label.py                ← UI label handling
│       └── misc.py                 ← Coordinate conversion utilities
│
├── restoration/
│   ├── snapshot_provider.py        ← Pre-task state capture (cursor, window, app)
│   ├── restore_provider.py         ← Post-task restoration. Verify after each step.
│   ├── restore_verifier.py         ← Evidence-based restoration check
│   └── snapshot_types.py           ← Snapshot dataclasses
│
├── authority/
│   ├── authority_policy.py         ← Rule-based authority decisions
│   ├── input_arbitrator.py         ← Human input detection + yield logic
│   └── input_tracker.py            ← Tracks input events
│
├── state/
│   └── serializer.py               ← Auth state persistence (dirty flag, crash detect)
│
├── audit/
│   └── journal.py                  ← Action audit log. Records every execution event.
│
├── policy/
│   └── engine.py                   ← Policy evaluation engine
│
├── utils/
│   └── accessibility.py            ← Accessibility backend wiring
│
├── config/
│   └── timeouts.py                 ← Centralized timeout config
│                                      LLM_CALL: 30s, THREAD: 40s
│
├── docs/
│   ├── authority_constitution.md   ← Immutable authority laws
│   └── restoration_contract.md     ← Binding restoration guarantees
│
└── temp/
    └── arm_system.intent           ← Drop a task here to trigger execution
```

---

## LLM Integration — Ollama Wiring

### How The LLM Is Bootstrapped

```
run.py
  │
  ├─ resolve_model_name()
  │    Reads from: sys.argv[1]  OR  $LLM_MODEL env var
  │    Example: "qwen2.5-vl:7b-instruct"
  │
  ├─ build_llm(model_name)          [adapters/factory.py]
  │    Validates model name format
  │    Applies safety patches (temperature enforcement, cloud disable)
  │    Registry lookup: "qwen2.5-vl" → QwenOllamaAdapter
  │    Returns adapter instance
  │
  └─ _make_llm_callable(adapter)
       Checks adapter has get_next_action()
       Wraps async coroutine into sync callable
       Enforces LLM_THREAD_TIMEOUT_SECONDS (40s hard limit)
       Returns: def llm_callable(messages, objective, session_id) → List[dict]
```

### What The LLM Sees (Planning Prompt Structure)

```
System:   "You are a deterministic planner."

User:     Environment:
          {
            "os": "Linux",
            "architecture": "x86_64",
            "tools": {"node": false, "npm": false, "git": true, "docker": false},
            "display_available": true,
            "running_in_container": false
          }

          Screen:
          button: Download Node.js
          link: Documentation
          input: Search
          button: macOS Installer
          button: Linux Installer

          Goal:
          "Install Node.js from the official website"

          Return STRICT JSON list of steps.
```

### What The LLM Returns (Expected Format)

```json
[
  {
    "type": "ui_interaction",
    "description": "Click the Linux installer download button",
    "action": {
      "operation": "click",
      "text": "Linux Installer"
    },
    "verification": {
      "screen_changed": true
    },
    "estimated_duration": 2.0,
    "retryable": true
  },
  {
    "type": "command_execution",
    "description": "Make installer executable and run it",
    "action": {
      "operation": "command",
      "command": "chmod +x node-installer.sh && ./node-installer.sh"
    },
    "verification": {
      "command": "node --version",
      "output_contains": "v"
    },
    "estimated_duration": 30.0,
    "retryable": false
  }
]
```

### Allowed Step Types

| Type | Purpose |
|------|---------|
| `ui_interaction` | Mouse clicks, keyboard input, hotkeys |
| `command_execution` | Terminal commands |
| `file_creation` | Write files to disk |
| `verification` | Check a condition is true |
| `tool_installation` | Install a tool (triggers AutonomousInstaller) |
| `done` | Signal task completion (auto-appended by planner) |

### Timeout Hierarchy

```
LLM_CALL_TIMEOUT_SECONDS  = 30s   ← asyncio.wait_for inside ExecutionPlanner
LLM_THREAD_TIMEOUT_SECONDS = 40s   ← threading.Thread.join in run.py
                                      (fires AFTER planner timeout as backup)
```

---

## Observer System

The observer is a **pure watchdog**. It watches the screen continuously and builds a world model. It never plans, never acts, never changes mode.

```
                    Screen
                      │
                      ▼ (every 500ms)
            ┌──────────────────┐
            │  VisionRuntime   │
            │                  │
            │  PIL ImageGrab   │
            │  → base64 PNG    │
            │  → Ollama call   │
            │  → structured    │
            │    perception    │
            └────────┬─────────┘
                     │ {elements, text, focused_app, frame_ts}
                     ▼ (every 200ms)
            ┌──────────────────┐
            │  ObserverLoop    │
            │  (5 Hz daemon)   │
            │                  │
            │  Pulls latest    │
            │  perception      │
            └────────┬─────────┘
                     │
           ┌─────────┴──────────┐
           │                    │
           ▼                    ▼
  ┌─────────────────┐  ┌────────────────┐
  │  ObserverCore   │  │   WorldGraph   │
  │                 │  │                │
  │  Passive witness│  │  Structured    │
  │  Health tracking│  │  entity map    │
  │  Blindness det. │  │  Spatial index │
  │  Deep-copy snap │  │  Delta compute │
  │  ZERO authority │  │  History track │
  └─────────────────┘  └────────────────┘
```

### Blindness Detection

The observer tracks consecutive perception misses. If the screen goes dark or the vision model stops responding:

```
First miss    → increment consecutive_misses counter
Miss 15 times → mark_blind(reason)
                → ObserverBlindnessError raised
                → Execution loop stops
                → Restoration triggered
```

Recovery: If perception resumes within 5 seconds, observer heals automatically.

---

## Autonomous Installer

When the LLM's plan includes a `tool_installation` step, the **AutonomousInstaller** takes over. It uses the same LLM to navigate a real browser and install the tool from its official website.

```
LLM Plan includes:
{
  "type": "tool_installation",
  "action": {
    "operation": "install",
    "tool": {
      "name": "Node.js",
      "official_url": "https://nodejs.org/en/download",
      "version_command": "node --version",
      "min_version": "18.0.0"
    }
  }
}

                    │
                    ▼
         AutonomousInstaller.install_tool(tool)

         Check: is it already installed?
           ─── node --version → success? → skip
           ─── not found? → proceed

                    │
                    ▼
         Open browser (os.open_browser())
         Navigate to https://nodejs.org/en/download

                    │
                    ▼
         ┌──────────────────────────────────┐
         │   INSTALL LOOP (max 120 iter)    │
         │                                  │
         │  1. Capture current screen       │
         │  2. Ask LLM: "What to do next    │
         │     to install Node.js? Here's   │
         │     the screen perception."      │
         │  3. LLM returns: click/type/wait │
         │  4. Execute the action           │
         │  5. Wait 1s for UI to settle     │
         │  6. Check if installed yet       │
         │  7. Repeat                       │
         └──────────────────────────────────┘
                    │
                    ▼
         Verify: node --version ≥ 18.0.0
         ✅ Installed → continue plan
         ❌ Timeout  → InstallationError
```

**Key constraint:** `official_url` must use `https://`. No arbitrary URLs. No package managers scripted in advance — the LLM figures out the UI.

---

## Snapshot & Restoration Contract

Before any task executes, the system captures a **pre-task snapshot**. After the task (success or failure), it restores to that snapshot.

### What Gets Captured

```
SnapshotProvider.take_snapshot()
  │
  ├─ Cursor position      → {x: 847, y: 532}
  ├─ Focused window title → "Terminal — bash"
  ├─ Active application   → "Terminal"
  ├─ Execution mode       → "OBSERVER"
  ├─ Vision frame ts      → 1738234567.234
  └─ Capture duration     → 12.4ms (must be < 250ms)

All stored in LRU registry (128 entries, 1 hour TTL)
Identified by UUID snapshot_id
```

### Snapshot Contract Rules

```
attach_snapshot() → only allowed in OBSERVER mode
consume_snapshot() → only allowed in ARMED mode, one-time use
                     (prevents duplicate execution against same snapshot)
```

### What Gets Restored

```
RestoreProvider.restore(snapshot)
  │
  Phase 1: Stop all automated input
           Force release all modifier keys (shift, ctrl, alt, cmd)
           Release mouse buttons
  │
  Phase 2: Restore application
           activate_application({title: "Terminal"})
           wait 80ms
  │
  Phase 3: Restore window focus
           focus_window({title: "Terminal — bash"})
           wait 80ms
  │
  Phase 4: Restore cursor position
           set_cursor_position({x: 847, y: 532})
           wait 80ms
  │
  Phase 5: Verify (up to 5 attempts)
           ├─ cursor within ±5px?  ✅/❌
           ├─ window title match?  ✅/❌ (Levenshtein ≤ 2)
           └─ app title match?     ✅/❌
  │
  Phase 6: Mark snapshot_id as completed in ledger
           (idempotent — safe to re-run)
```

### What Is NOT Restored

- Clipboard contents
- Scroll position
- Application internal state (tabs, unsaved work)
- Network connections
- Undo/redo history
- Running processes started during task

---

## Authority & Safety Layer

### Authority Hierarchy (Immutable)

```
1. Human physical input        ← HIGHEST AUTHORITY
   (keyboard/mouse during task)
   → IMMEDIATE yield, execution stops

2. Human explicit intent
   (the task you submitted)
   → Required to arm the system

3. InputArbitrator
   → Evaluates each action: CONTINUE / YIELD / ABORT

4. ModeController
   → Enforces lifecycle, gates LLM access

5. LLM outputs                 ← LOWEST AUTHORITY
   → Planning only, fully validated before use
```

### Per-Action Authority Check

Before every action in the execution loop:

```python
authority = input_arbitrator.evaluate(
    input_event_ts=time.monotonic(),
    high_risk=action.operation in {"command", "install"},
    soc_confident=belief.environment_stability > 0.7,
)

if authority == AuthorityDecision.ABORT:
    raise AuthorityAbortError()   # Human said stop

if authority != AuthorityDecision.CONTINUE:
    raise RuntimeError("REPLAN_REQUIRED")  # Something changed
```

### Stagnation Detection

The execution loop tracks how many consecutive steps failed verification:

```
MAX_STAGNANT_ITERS = 12

Each failed step  → stagnant_iterations += 1
12 consecutive failures → raise RuntimeError("REPLAN_REQUIRED")
                          → back to planning with fresh world snapshot
MAX_REPLANS = 3  → after 3 replans, task fails permanently
```

### Watchdog Heartbeat

The OS backend requires a heartbeat from the execution loop every 2 seconds:

```
os_backend.heartbeat()  ← Called before each action

If no heartbeat for 2s AND automation_active:
  → Watchdog thread triggers force_release_all()
  → All modifier keys released
  → Mouse buttons released
  → automation_active = False
```

---

## Data Flow Diagrams

### Full System Startup

```
python run.py qwen2.5-vl:7b-instruct
│
├── resolve_model_name("qwen2.5-vl:7b-instruct")
├── AdapterFactory.build_llm("qwen2.5-vl:7b-instruct")
│     └── apply_patches()  [safety hardening]
│     └── QwenOllamaAdapter(model_name="qwen2.5-vl:7b-instruct")
├── _make_llm_callable(adapter)
└── main(llm_callable, "qwen2.5-vl:7b-instruct")
    │
    ├── OperatingSystem()
    ├── AuthorityStateSerializer(".authority_state.json")
    ├── ObserverCore()
    ├── VisionRuntime("qwen2.5-vl:7b-instruct")
    │     └── validate_display_environment()  [checks $DISPLAY]
    ├── WorldGraph()
    ├── ObserverLoop(observer, vision, world_graph).start()
    │     └── Daemon thread: 5 Hz perception loop begins
    │
    ├── collect_environment_fingerprint()
    │     └── Reads: OS, arch, tools (shutil.which), display, container
    │
    ├── auth_state.load()
    │     └── dirty=True? → crash recovery → force_observer()
    │
    ├── vision_runtime.start()
    │
    ├── WARMUP: wait up to 8s for 3 stable perception frames
    │
    ├── SnapshotProvider(observer, os_backend, mode_controller)
    ├── RestoreProvider(os_backend, mode_controller, snapshot_provider)
    ├── IntentListener(mode, snapshot_provider).start()
    │     └── Polls stdin / /tmp/projectzeo.intent every 100ms
    │
    └── MAIN LOOP begins
```

### Task Execution Flow

```
User types: "Set up a React app with TypeScript"
                │
                ▼ (IntentListener picks it up)
  snapshot_id = snapshot_provider.take_snapshot()
  mode.attach_snapshot(snapshot_id)
  mode.arm("Set up a React app with TypeScript")
         ── mode = ARMED ──

                │
                ▼ (main loop detects ARMED)
  snapshot_id = mode.consume_snapshot()
  intent = mode.get_intent()
  planner = ExecutionPlanner(llm_callable, env_fingerprint, world_graph)

  mode.begin_planning()
         ── mode = PLANNING ──

                │
                ▼
  execution_plan = planner.create_plan(
      objective="Set up a React app with TypeScript",
      requirements={"environment": env_fingerprint},
      high_level_steps=[{"goal": "Set up a React app with TypeScript"}]
  )
  ┌─ INSIDE create_plan: ──────────────────────────────────────────────┐
  │  planner._expand_goal(goal)                                        │
  │    → _call_llm_sync(prompt)                                        │
  │    → asyncio.run(_call_llm_async(prompt))                          │
  │    → llm_callable(messages, objective="planning", session="plann..")│
  │    → QwenOllamaAdapter.get_next_action()                           │
  │    → ollama.Client.chat(model="qwen2.5-vl:7b-instruct", ...)       │
  │    → LLM returns JSON list of steps                                │
  │    → Validate each step (type, action, duration, command safety)   │
  │    → Return List[ExecutionStep]                                    │
  └────────────────────────────────────────────────────────────────────┘

  mode.attach_execution_plan("plan_1738234600")
  mode.mark_planning_complete()

                │
                ▼
  auth_state.persist(dirty=True, restore_required=True, ...)
  mode.execute()
         ── mode = EXECUTING ──

                │
                ▼
  operate_main(intent, execution_plan, planner, observer, world_graph, os_backend)
  ┌─ INSIDE operate_main: ─────────────────────────────────────────────┐
  │  For each step in execution_plan.steps:                            │
  │    1. observer.snapshot() → get current screen                     │
  │    2. world_graph.update(perception)                               │
  │    3. belief.bayesian_update(world_snapshot)                       │
  │    4. selected_action = action_ranker.select(candidates, belief)   │
  │    5. input_arbitrator.evaluate() → CONTINUE/YIELD/ABORT           │
  │    6. os_backend.heartbeat()                                       │
  │    7. with action_timeout(30):                                     │
  │         _execute_decision(action, os_backend, installer)           │
  │    8. verifier.verify_step(step, result, screen)                   │
  │    9. belief.record_action(key, reward)                            │
  │   10. advance to next step                                         │
  └────────────────────────────────────────────────────────────────────┘

                │
                ▼ (task done or failed)
  mode.begin_restoration()
         ── mode = RESTORING ──

  restore_provider.restore_snapshot(snapshot_id)
  auth_state.persist(dirty=False, restore_required=False, ...)

  mode.complete_execution()
         ── mode = OBSERVER ──
```

---

## File Structure

```
ProjectZeo-main/
├── run.py                    ← START HERE
├── main.py                   ← Kernel main loop
├── evaluate.py               ← Evaluation harness
├── setup.py                  ← Package setup
├── requirements.txt          ← Python dependencies
├── requirements-audio.txt    ← Optional audio deps
│
├── adapters/                 ← LLM provider layer
├── core/                     ← Kernel subsystems
├── observer/                 ← Screen perception
├── operate/                  ← Execution engine
├── restoration/              ← Snapshot + restore
├── authority/                ← Input arbitration
├── state/                    ← Persistence
├── audit/                    ← Action logging
├── policy/                   ← Rule engine
├── utils/                    ← Shared utilities
├── config/                   ← Timeouts + settings
│
├── temp/
│   └── arm_system.intent     ← Drop a task here
│
├── docs/
│   ├── authority_constitution.md
│   └── restoration_contract.md
│
└── logs/
    └── mode_transitions.jsonl  ← Auto-created at runtime
```

---

## Quick Start

### Prerequisites

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull the vision model
ollama pull qwen2.5-vl:7b-instruct

# 3. Verify Ollama is running
ollama list
# Should show: qwen2.5-vl:7b-instruct
```

### Installation

```bash
# Clone the repo
git clone https://github.com/yourname/ProjectZeo.git
cd ProjectZeo-main

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Method 1: CLI argument
python run.py qwen2.5-vl:7b-instruct

# Method 2: Environment variable
export LLM_MODEL="qwen2.5-vl:7b-instruct"
python run.py
```

### Submitting a Task

Once the system is running and you see `[OBSERVER] Initialized`, you have two ways to give it a task:

**Option A — Standard input (if running in a terminal):**
```
Type your task and press Enter:
> Set up a Python Flask API with SQLite database
```

**Option B — Intent file:**
```bash
# Write your task to the intent file
echo "Install Node.js and create a React app" > /tmp/projectzeo.intent

# The system picks it up within 100ms automatically
```

### Stopping

```bash
Ctrl+C
# System sends SIGINT → triggers safe shutdown → restores screen state
```

---

## Configuration

### Timeout Configuration (`config/timeouts.py`)

```python
LLM_CALL_TIMEOUT_SECONDS  = 30.0   # How long the planner waits for LLM
LLM_THREAD_TIMEOUT_SECONDS = 40.0  # Thread-level hard cap (must be > LLM_CALL)
```

### Main Loop Constants (`main.py`)

```python
HEARTBEAT_INTERVAL    = 2.0    # Main loop sleep between heartbeats (seconds)
MAX_TASK_SECONDS      = 5400   # 90 minutes max per task
MAX_REPLANS           = 3      # Max replan attempts before task fails
WARMUP_STABLE_FRAMES  = 3      # Frames required before accepting tasks
```

### Observer Constants (`observer/observer_core.py`)

```python
STARTUP_GRACE_TICKS       = 30    # Ticks before blindness enforced at startup
STARTUP_GRACE_SECONDS     = 15.0  # Seconds before blindness enforced at startup
MAX_CONSECUTIVE_MISSES    = 15    # Misses before going blind
BLIND_RECOVERY_SECONDS    = 5.0   # Auto-recovery window
```

### Execution Constants (`operate/operate.py`)

```python
MAX_PERCEPTION_ENTITIES   = 20    # Max UI elements fed to belief state per tick
MAX_STAGNANT_ITERS        = 12    # Failed steps before REPLAN_REQUIRED
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_MODEL` | If no CLI arg | — | Model name (e.g. `qwen2.5-vl:7b-instruct`) |
| `DISPLAY` | Linux only | — | X display (e.g. `:0`) |
| `WAYLAND_DISPLAY` | Linux/Wayland | — | Wayland display |

---

## How Intent Is Delivered

```
                ┌─────────────────────────────────────┐
                │         IntentListener               │
                │         (100ms poll loop)            │
                └──────────────┬──────────────────────┘
                               │
                ┌──────────────┴──────────────────┐
                │                                 │
        stdin (if tty)                    /tmp/projectzeo.intent
                │                                 │
        readline()                        Security checks:
        strip whitespace                  - Must be regular file
        return if non-empty               - Must be owned by current user
                                          - Must have 0o600 permissions
                                          - Max 4096 bytes
                                          - File deleted after reading
```

The intent file approach is useful when running in environments where stdin is not a tty (e.g., systemd service, tmux background session).

---

## Logs & Audit

### Mode Transition Log

Every mode change is written to `logs/mode_transitions.jsonl`:

```json
{"ts": 1738234567.12, "from": "OBSERVER", "to": "ARMED", "reason": "intent armed", "forced": false, "vision_ok": true, "observer_healthy": true, "plan_attached": false, "plan_id": null}
{"ts": 1738234567.89, "from": "ARMED", "to": "PLANNING", "reason": "planning started", "forced": false, "vision_ok": true, "observer_healthy": true, "plan_attached": false, "plan_id": null}
{"ts": 1738234572.34, "from": "PLANNING", "to": "EXECUTING", "reason": "execution started (plan=plan_1738234572)", "forced": false, "vision_ok": true, "observer_healthy": true, "plan_attached": true, "plan_id": "plan_1738234572"}
```

### Authority State

`.authority_state.json` — persists across crashes:

```json
{
  "execution_mode": "OBSERVER",
  "automation_active": false,
  "restore_required": false,
  "last_snapshot_id": null,
  "dirty": false
}
```

If `dirty=true` when the system starts, it knows it crashed during a task and forces restoration before accepting new tasks.

### Restore Ledger

`memory/restore_ledger.json` — tracks which snapshots have been successfully restored (prevents double-restoration):

```json
["snap_abc123", "snap_def456", "snap_ghi789"]
```

---

## Known Limitations

### Current Adapter Registry

Only `qwen2.5-vl` model family is registered. Other Ollama models require adding entries to `adapters/factory.py`:

```python
_ADAPTER_REGISTRY = {
    "qwen2.5-vl": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
    # Add more here:
    # "llava": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
}
```

### What Is Not Restored

The restoration contract restores cursor, window focus, and active application. It does **not** restore application internal state (open tabs, unsaved text, scroll position). If a task opens new browser tabs, they remain open after restoration.

### Screen Must Be Accessible

The vision system requires a real display (not headless). On Linux, `$DISPLAY` or `$WAYLAND_DISPLAY` must be set. Running over SSH requires X11 forwarding or a virtual display (e.g., Xvfb).

### Snapshot Requires Observer Health

Tasks cannot start if:
- Observer has gone blind (no perception for 15 consecutive ticks)
- Vision runtime is unhealthy (Ollama not responding)
- System is not in OBSERVER mode

### Single Task at a Time

The system processes one task at a time. A new task cannot start until the current one completes and restoration finishes.

---

## Core Guarantees

| Guarantee | Mechanism |
|-----------|-----------|
| Screen state restored after every task | Snapshot + RestoreProvider (always runs) |
| LLM never called during execution | ModeController gates LLM to PLANNING only |
| No runaway execution | 90-minute wall-clock timeout |
| Human always wins | InputArbitrator yields on any human input |
| No silent failures | All errors raised explicitly, logged to journal |
| Crash recovery | dirty flag in `.authority_state.json` |
| Deterministic planning | temperature=0 enforced by safety layer |
| No cloud API calls in local mode | Safety layer disables all cloud fallbacks |

---

## License

See [LICENSE](./LICENSE) for terms.

---

<div align="center">

**ProjectZeo — Your OS is the sandbox. The LLM is the brain.**

</div>
