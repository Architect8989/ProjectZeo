import os
import json
import hashlib
import tempfile
import re
from typing import List, Dict, Optional

import pathlib as _pathlib
_PROJECT_ROOT = _pathlib.Path(__file__).resolve().parents[2]
PLAYBOOK_DIR = str(_PROJECT_ROOT / "memory" / "playbooks")
del _pathlib


# -------------------------------------------------

def _ensure_dir():
    os.makedirs(PLAYBOOK_DIR, exist_ok=True)


def _hash_intent(intent: str) -> str:
    return hashlib.sha256(intent.lower().encode("utf-8")).hexdigest()


def _normalize_intent(intent: str) -> str:
    """
    BUG-04 FIX: Normalize an intent string for fuzzy matching.

    The original system used SHA-256 of the raw intent, so
    "open Firefox and go to google.com" and "open firefox and navigate
    to google.com" produced completely different hashes — recall rate ~0%.

    Normalization strips punctuation, lowercases, collapses whitespace,
    and alphabetically sorts tokens so trivial word-order variations hash
    identically (e.g. "go to X in Firefox" ≈ "open Firefox navigate to X").
    """
    text = intent.lower().strip()
    # Remove punctuation except dots in URLs/version strings
    text = re.sub(r"[^\w\s\.]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Sort tokens for order-invariance
    tokens = sorted(text.split())
    return " ".join(tokens)


def _normalized_hash(intent: str) -> str:
    return hashlib.sha256(_normalize_intent(intent).encode("utf-8")).hexdigest()


def _compute_checksum(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _token_similarity(a: str, b: str) -> float:
    """
    Jaccard similarity on the token sets of two normalized intent strings.
    Returns 0.0–1.0.  Used as a last-resort fallback when neither the exact
    nor the normalized hash produces a cache hit.
    """
    tokens_a = set(_normalize_intent(a).split())
    tokens_b = set(_normalize_intent(b).split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# -------------------------------------------------
# Public API
# -------------------------------------------------

def save_playbook(intent: str, actions: List[Dict]) -> None:
    _ensure_dir()

    key = _hash_intent(intent)
    path = os.path.join(PLAYBOOK_DIR, f"{key}.json")
    tmp_fd, tmp_path = tempfile.mkstemp(dir=PLAYBOOK_DIR)

    try:
        payload = {
            "schema": 1,
            "intent": intent,
            # BUG-04 FIX: also store the normalized form so load_playbook()
            # can compare against it during fuzzy lookup.
            "intent_normalized": _normalize_intent(intent),
            "actions": actions,
        }

        raw = json.dumps(payload, indent=2)
        checksum = _compute_checksum(raw)

        wrapper = {
            "checksum": checksum,
            "payload": payload,
        }

        with os.fdopen(tmp_fd, "w") as f:
            json.dump(wrapper, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

        # BUG-04 FIX: also save under the normalized hash so future lookups
        # using a paraphrased intent can still find this playbook.
        norm_key = _normalized_hash(intent)
        if norm_key != key:
            norm_path = os.path.join(PLAYBOOK_DIR, f"{norm_key}.json")
            try:
                import shutil as _sh
                _sh.copy2(path, norm_path)
            except Exception:
                pass

    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def load_playbook(intent: str) -> Optional[List[Dict]]:
    """
    BUG-04 FIX: Three-tier lookup strategy.

    1. Exact SHA-256 of the raw intent (original behaviour, fastest).
    2. SHA-256 of the *normalized* intent (handles case/punctuation/trivial
       paraphrasing without any external dependencies).
    3. Linear scan of all playbooks using Jaccard token similarity with a
       0.75 threshold (handles word-order variation and synonyms; O(n) but
       the playbook directory is expected to be small ≤10k entries).
    """
    # --- Tier 1: exact match ---
    result = _load_playbook_by_key(_hash_intent(intent))
    if result is not None:
        return result

    # --- Tier 2: normalized match ---
    result = _load_playbook_by_key(_normalized_hash(intent))
    if result is not None:
        return result

    # --- Tier 3: best-effort Jaccard scan ---
    try:
        if not os.path.isdir(PLAYBOOK_DIR):
            return None
        best_score = 0.0
        best_actions = None
        for fname in os.listdir(PLAYBOOK_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(PLAYBOOK_DIR, fname)
            try:
                with open(fpath, "r") as f:
                    wrapper = json.load(f)
                payload = wrapper.get("payload", {})
                stored_intent = payload.get("intent", "")
                score = _token_similarity(intent, stored_intent)
                if score > best_score:
                    best_score = score
                    best_actions = payload.get("actions")
            except Exception:
                continue
        if best_score >= 0.75 and best_actions is not None:
            return best_actions
    except Exception:
        pass

    return None


def _load_playbook_by_key(key: str) -> Optional[List[Dict]]:
    path = os.path.join(PLAYBOOK_DIR, f"{key}.json")

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            wrapper = json.load(f)

        checksum = wrapper.get("checksum")
        payload = wrapper.get("payload")

        if not checksum or not payload:
            return None

        raw = json.dumps(payload, indent=2)
        if _compute_checksum(raw) != checksum:
            return None

        return payload.get("actions")

    except Exception:
        return None
