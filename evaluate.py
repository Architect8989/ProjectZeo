from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

def _supports_ansi() -> bool:
    plat = platform.system()
    return (plat != "Windows" or "ANSICON" in os.environ) and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

if _supports_ansi():
    GREEN = "\033[32m"; BRIGHT_GREEN = "\033[92m"; RESET = "\033[0m"
    BLUE = "\033[94m"; YELLOW = "\033[33m"; RED = "\033[31m"; MAGENTA = "\033[95m"
else:
    GREEN = BRIGHT_GREEN = RESET = BLUE = YELLOW = RED = MAGENTA = ""


# ---------------------------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
_DEFAULT_JOURNAL = str(_PROJECT_ROOT / "logs" / "action_audit.jsonl")
_DEFAULT_LIKELIHOODS = str(_PROJECT_ROOT / "likelihoods.json")

_LIKELIHOOD_KEYS = [
    "app_match_with_delta",
    "app_match_no_delta",
    "ui_rich",
    "ui_sparse",
    "ui_empty",
    "neutral_with_delta",
    "neutral_no_delta",
    "ENTITY_RICH_THRESHOLD",
]


def _load_journal_entries(journal_path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    try:
        with open(journal_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"{YELLOW}[CALIBRATE] Skip malformed line {lineno}: {e}{RESET}", file=sys.stderr)
    except FileNotFoundError:
        print(f"{RED}[CALIBRATE] Journal not found: {journal_path}{RESET}", file=sys.stderr)
    except OSError as e:
        print(f"{RED}[CALIBRATE] Cannot read journal: {e}{RESET}", file=sys.stderr)
    return entries


def _extract_observation_pairs(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract (observation, success) pairs from journal entries."""
    pairs: List[Dict[str, Any]] = []
    for entry in entries:
        perception = entry.get("perception")
        if not isinstance(perception, dict):
            continue

        focused_app = perception.get("focused_app") or ""
        elements = perception.get("elements", [])
        entity_count = len(elements) if isinstance(elements, list) else 0

        had_delta = bool(entry.get("had_delta") or entry.get("world_changed"))

        objective = str(entry.get("objective") or entry.get("intent") or "").lower()
        app_lower = focused_app.lower()
        app_matched = bool(app_lower and objective and app_lower in objective)

        success = bool(entry.get("success", entry.get("step_success", True)))

        pairs.append({
            "focused_app": focused_app,
            "entity_count": entity_count,
            "had_delta": had_delta,
            "app_matched": app_matched,
            "success": success,
        })
    return pairs


def _compute_entity_rich_threshold(pairs: List[Dict[str, Any]]) -> int:
    counts = sorted(p["entity_count"] for p in pairs if p["success"])
    if not counts:
        return 10
    n = len(counts)
    median = counts[n // 2] if n % 2 == 1 else (counts[n // 2 - 1] + counts[n // 2]) // 2
    return max(5, min(50, median))


def _compute_ratios(pairs: List[Dict[str, Any]], threshold: int) -> Dict[str, Optional[float]]:
    buckets: Dict[str, List[int]] = {
        k: [0, 0] for k in _LIKELIHOOD_KEYS if k != "ENTITY_RICH_THRESHOLD"
    }
    total = len(pairs)
    total_success = sum(1 for p in pairs if p["success"])

    if total == 0:
        return {}

    for p in pairs:
        matched = p["app_matched"]
        delta = p["had_delta"]
        count = p["entity_count"]
        suc = int(p["success"])

        if matched and delta:
            buckets["app_match_with_delta"][1] += 1
            buckets["app_match_with_delta"][0] += suc
        elif matched and not delta:
            buckets["app_match_no_delta"][1] += 1
            buckets["app_match_no_delta"][0] += suc

        if count >= threshold:
            buckets["ui_rich"][1] += 1
            buckets["ui_rich"][0] += suc
        elif count > 0:
            buckets["ui_sparse"][1] += 1
            buckets["ui_sparse"][0] += suc
        else:
            buckets["ui_empty"][1] += 1
            buckets["ui_empty"][0] += suc

        if not matched and delta:
            buckets["neutral_with_delta"][1] += 1
            buckets["neutral_with_delta"][0] += suc
        elif not matched and not delta:
            buckets["neutral_no_delta"][1] += 1
            buckets["neutral_no_delta"][0] += suc

    prior = total_success / total if total > 0 else 0.5
    ratios: Dict[str, Optional[float]] = {}
    for key, (hits, n) in buckets.items():
        if n < 10:
            ratios[key] = None  # insufficient data
            continue
        p_obs = hits / n
        ratio = (p_obs / prior) if prior > 0 else 1.0
        ratios[key] = round(max(0.1, min(10.0, ratio)), 4)
    return ratios


def calibrate(
    journal_path: str = _DEFAULT_JOURNAL,
    likelihoods_out: str = _DEFAULT_LIKELIHOODS,
    min_samples: int = 20,
    dry_run: bool = False,
) -> int:
    print(f"{BLUE}[CALIBRATE] Reading journal: {journal_path}{RESET}")
    entries = _load_journal_entries(journal_path)
    if not entries:
        print(f"{RED}[CALIBRATE] No journal entries found — aborting.{RESET}", file=sys.stderr)
        return 1

    print(f"{BLUE}[CALIBRATE] Loaded {len(entries)} journal entries.{RESET}")
    pairs = _extract_observation_pairs(entries)

    if len(pairs) < min_samples:
        print(
            f"{YELLOW}[CALIBRATE] Only {len(pairs)} observation pairs extracted "
            f"(minimum required: {min_samples}).  Likelihoods NOT updated.\n"
            f"  Run more tasks to build a calibration dataset.{RESET}",
            file=sys.stderr,
        )
        return 2

    print(f"{BLUE}[CALIBRATE] {len(pairs)} observation pairs extracted.{RESET}")
    threshold = _compute_entity_rich_threshold(pairs)
    ratios = _compute_ratios(pairs, threshold)

    # Load existing likelihoods as fallback
    existing: Dict[str, Any] = {}
    try:
        with open(likelihoods_out, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except Exception:
        pass

    updated: Dict[str, Any] = dict(existing)
    updated["ENTITY_RICH_THRESHOLD"] = threshold

    insufficient_keys = []
    for key, val in ratios.items():
        if val is None:
            insufficient_keys.append(key)
        else:
            updated[key] = val

    if insufficient_keys:
        print(f"{YELLOW}[CALIBRATE] Insufficient data for: {insufficient_keys}. Existing values retained.{RESET}")

    print(f"\n{GREEN}[CALIBRATE] Calibrated likelihood ratios:{RESET}")
    for key in _LIKELIHOOD_KEYS:
        val = updated.get(key, "N/A")
        marker = "*" if key in ratios and ratios.get(key) is not None else " "
        print(f"  {marker} {key:30s} = {val}")
    print(f"\n  (* = empirically calibrated from {len(pairs)} samples)")

    if dry_run:
        print(f"\n{YELLOW}[CALIBRATE] Dry run — likelihoods.json NOT written.{RESET}")
        return 0

    tmp_path = likelihoods_out + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, likelihoods_out)
        print(f"\n{GREEN}[CALIBRATE] Written: {likelihoods_out}{RESET}")
        return 0
    except OSError as e:
        print(f"{RED}[CALIBRATE] Failed to write {likelihoods_out}: {e}{RESET}", file=sys.stderr)
        return 1
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LEGACY TASK EVAL
# ---------------------------------------------------------------------------

TEST_CASES = {
    "Go to Github.com": "A Github page is visible.",
    "Go to Youtube.com and play a video": "The YouTube video player is visible.",
}

EVALUATION_PROMPT = """\
Your job is to look at the given screenshot and determine if the following guideline is met.
Respond ONLY in this format:
{{ "guideline_met": (true|false), "reason": "..." }}

Guideline: {guideline}
"""

SCREENSHOT_PATH = os.path.join("screenshots", "screenshot.png")


def format_evaluation_prompt(guideline: str) -> str:
    return EVALUATION_PROMPT.format(guideline=guideline)


def parse_eval_content(content: str) -> bool:
    content = re.sub(r"```json|```", "", content).strip()
    try:
        res = json.loads(content)
        print(res["reason"])
        return bool(res["guideline_met"])
    except (json.JSONDecodeError, KeyError, TypeError):
        print("The model gave a bad evaluation response and it couldn't be parsed. Exiting...")
        sys.exit(1)


def evaluate_final_screenshot_ollama(guideline: str, model: str = "qwen2.5-vl:7b-instruct") -> bool:
    """
    Evaluate using local Ollama vision model — compatible with OLLAMA_ONLY architecture.
    """
    import base64
    try:
        import ollama
    except ImportError:
        print(f"{RED}ollama package not installed. pip install ollama{RESET}", file=sys.stderr)
        sys.exit(1)

    with open(SCREENSHOT_PATH, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": format_evaluation_prompt(guideline), "images": [img_b64]}],
    )
    content = response["message"]["content"] if isinstance(response, dict) else response.message.content
    return parse_eval_content(content)


def evaluate_final_screenshot_openai(guideline: str) -> bool:
    import base64
    try:
        import openai
    except ImportError:
        print(f"{RED}openai package not installed. pip install openai{RESET}", file=sys.stderr)
        sys.exit(1)

    with open(SCREENSHOT_PATH, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": format_evaluation_prompt(guideline)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        temperature=0.0,
    )
    return parse_eval_content(response.choices[0].message.content)


def legacy_eval(model: str, use_ollama: bool = True) -> int:
    if not use_ollama:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
        except ImportError:
            pass

    print(f"{BLUE}[EVALUATING MODEL `{model}`]{RESET}")
    passed = failed = 0
    for objective, guideline in TEST_CASES.items():
        print(f"{BLUE}[EVALUATING]{RESET} '{objective}'")
        subprocess.run(["operate", "-m", model, "--prompt", f'"{objective}"'], stdout=subprocess.DEVNULL)
        try:
            result = (evaluate_final_screenshot_ollama if use_ollama else evaluate_final_screenshot_openai)(
                guideline, **({} if not use_ollama else {"model": model})
            )
        except OSError:
            print("[Error] Couldn't open the screenshot for evaluation")
            result = False
        if result:
            print(f"{GREEN}[PASSED]{RESET} '{objective}'"); passed += 1
        else:
            print(f"{RED}[FAILED]{RESET} '{objective}'"); failed += 1

    print(
        f"{MAGENTA}[EVALUATION COMPLETE]{RESET} "
        f"{passed} test{'' if passed == 1 else 's'} passed, "
        f"{failed} test{'' if failed == 1 else 's'} failed"
    )
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="ProjectZeo Evaluation & Calibration Runner")
    parser.add_argument("--calibrate", action="store_true", default=True, help="Run Bayesian calibration (default)")
    parser.add_argument("--journal-path", default=_DEFAULT_JOURNAL, help="Path to action_audit.jsonl")
    parser.add_argument("--likelihoods-out", default=_DEFAULT_LIKELIHOODS, help="Output likelihoods.json path")
    parser.add_argument("--min-samples", type=int, default=20, help="Min observation pairs required")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing")
    parser.add_argument("--legacy-eval", action="store_true", help="Run legacy screenshot-based task evaluation")
    parser.add_argument("-m", "--model", default="qwen2.5-vl:7b-instruct", help="Model for eval")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI GPT-4o for evaluation")
    args = parser.parse_args()

    if args.legacy_eval:
        return legacy_eval(model=args.model, use_ollama=not args.use_openai)

    return calibrate(
        journal_path=args.journal_path,
        likelihoods_out=args.likelihoods_out,
        min_samples=args.min_samples,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
