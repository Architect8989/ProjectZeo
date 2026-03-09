"""
restoration/browser_snapshot_provider.py

Differential browser state capture and restore via Playwright CDP.

Design contract:
  - If a browser was ALREADY OPEN before the task → capture its state and
    restore it afterward.
  - If the agent OPENED the browser → just close what it opened.  Nothing
    to restore because the user had no browser session.
  - If no browser was involved at all → this provider is a no-op.

Browser sessions are identified by a running process check before the task
starts. That pre-task flag drives every subsequent decision.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

_CDP_URL = os.environ.get("PROJECTZEO_CDP_URL", "http://localhost:9222").strip()
_CAPTURE_TIMEOUT_MS = 4000
_RESTORE_TIMEOUT_MS = 6000

_BROWSER_PROCESS_NAMES = frozenset({
    "chrome", "chromium", "chromium-browser", "google-chrome",
    "firefox", "firefox-esr", "brave", "brave-browser",
    "msedge", "microsoft-edge",
})

try:
    from playwright.sync_api import sync_playwright as _sync_playwright
    _PW_AVAILABLE = True
except ImportError:
    _sync_playwright = None  # type: ignore
    _PW_AVAILABLE    = False


@dataclass
class TabRecord:
    url:    str
    title:  str
    active: bool = False


@dataclass
class BrowserSnapshot:
    captured_at:         float
    browser_was_open:    bool
    tabs:                List[TabRecord] = field(default_factory=list)
    active_tab_index:    int             = 0
    cdp_url:             str             = _CDP_URL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "captured_at":      self.captured_at,
            "browser_was_open": self.browser_was_open,
            "active_tab_index": self.active_tab_index,
            "cdp_url":          self.cdp_url,
            "tabs":             [asdict(t) for t in self.tabs],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BrowserSnapshot":
        tabs = [TabRecord(**t) for t in d.get("tabs", [])]
        return cls(
            captured_at=float(d.get("captured_at", 0.0)),
            browser_was_open=bool(d.get("browser_was_open", False)),
            tabs=tabs,
            active_tab_index=int(d.get("active_tab_index", 0)),
            cdp_url=str(d.get("cdp_url", _CDP_URL)),
        )


def _browser_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-x", "-f", "|".join(_BROWSER_PROCESS_NAMES)],
            capture_output=True, text=True, timeout=3.0,
        ).stdout.strip()
        if out:
            return True
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True, timeout=3.0,
        ).stdout
        return any(name in result.lower() for name in _BROWSER_PROCESS_NAMES)
    except Exception:
        return False


def _cdp_capture(cdp_url: str) -> List[TabRecord]:
    if not _PW_AVAILABLE:
        return []
    tabs: List[TabRecord] = []
    try:
        with _sync_playwright() as pw:
            browser = pw.chromium.connect_over_cdp(
                cdp_url, timeout=_CAPTURE_TIMEOUT_MS
            )
            ctx = browser.contexts[0] if browser.contexts else None
            if ctx is None:
                return []
            pages = ctx.pages
            for i, page in enumerate(pages):
                try:
                    tabs.append(TabRecord(
                        url=page.url,
                        title=page.title(),
                        active=(i == 0),
                    ))
                except Exception:
                    pass
    except Exception as exc:
        _logger.debug("[BrowserSnap] CDP capture failed: %s", exc)
    return tabs


def capture() -> BrowserSnapshot:
    browser_open = _browser_running()
    snap = BrowserSnapshot(
        captured_at=time.time(),
        browser_was_open=browser_open,
    )
    if not browser_open:
        _logger.debug("[BrowserSnap] No browser running pre-task — nothing to capture.")
        return snap

    tabs = _cdp_capture(_CDP_URL)
    snap.tabs = tabs
    _logger.debug(
        "[BrowserSnap] Captured %d tabs from pre-existing browser session.", len(tabs)
    )
    return snap


def restore(
    pre_snap: BrowserSnapshot,
    agent_opened_browser: bool,
) -> bool:
    if not pre_snap.browser_was_open:
        if agent_opened_browser:
            _close_agent_browser()
        return True

    if not _PW_AVAILABLE:
        _logger.warning("[BrowserSnap] playwright not available — cannot restore browser.")
        return False

    if not pre_snap.tabs:
        _logger.debug("[BrowserSnap] No tab data in snapshot — skipping restore.")
        return True

    try:
        _restore_tabs(pre_snap)
        return True
    except Exception as exc:
        _logger.warning("[BrowserSnap] Tab restore failed: %s", exc)
        return False


def _restore_tabs(snap: BrowserSnapshot) -> None:
    with _sync_playwright() as pw:
        try:
            browser = pw.chromium.connect_over_cdp(
                snap.cdp_url, timeout=_RESTORE_TIMEOUT_MS
            )
        except Exception:
            _logger.debug("[BrowserSnap] Cannot connect to CDP for restore.")
            return

        ctx = browser.contexts[0] if browser.contexts else None
        if ctx is None:
            return

        existing_pages = ctx.pages

        target_urls = [t.url for t in snap.tabs if t.url and t.url not in ("about:blank", "chrome://newtab/")]

        existing_urls = set()
        for page in existing_pages:
            try:
                existing_urls.add(page.url)
            except Exception:
                pass

        for url in target_urls:
            if url not in existing_urls:
                try:
                    page = ctx.new_page()
                    page.goto(url, timeout=_RESTORE_TIMEOUT_MS)
                except Exception:
                    pass

        active_idx = min(snap.active_tab_index, len(ctx.pages) - 1)
        if ctx.pages and active_idx >= 0:
            try:
                ctx.pages[active_idx].bring_to_front()
            except Exception:
                pass

    _logger.info("[BrowserSnap] Browser tabs restored to pre-task state.")


def _close_agent_browser() -> None:
    try:
        for name in _BROWSER_PROCESS_NAMES:
            subprocess.run(["pkill", "-f", name], capture_output=True, timeout=3.0)
        _logger.info("[BrowserSnap] Closed agent-opened browser.")
    except Exception as exc:
        _logger.debug("[BrowserSnap] Close agent browser error: %s", exc)
