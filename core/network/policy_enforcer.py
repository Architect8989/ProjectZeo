from __future__ import annotations

import ipaddress
import logging
import re
import socket
import time
import threading
from dataclasses import dataclass, field

# ExfiltrationGuard integration — audit fix for curl --data / wget --post-data
try:
    from core.network.exfiltration_guard import check_command as _exfil_check
    _EXFIL_GUARD_AVAILABLE = True
except ImportError:
    _EXFIL_GUARD_AVAILABLE = False
from typing import FrozenSet, List, Optional, Set, Tuple

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision constants
# ---------------------------------------------------------------------------
ALLOW = "ALLOW"
DENY = "DENY"

# ---------------------------------------------------------------------------
# Compile-time pattern table for command-level network detection
# ---------------------------------------------------------------------------

# Pattern: (compiled_regex, description_for_log, requires_domain_check)
_COMMAND_NETWORK_PATTERNS: List[Tuple[re.Pattern, str, bool]] = [
    # curl / wget outbound
    (re.compile(r"\bcurl\b", re.IGNORECASE), "curl", True),
    (re.compile(r"\bwget\b", re.IGNORECASE), "wget", True),
    # SSH outbound
    (re.compile(r"\bssh\b", re.IGNORECASE), "ssh", True),
    (re.compile(r"\bscp\b", re.IGNORECASE), "scp", True),
    (re.compile(r"\brsync\b.*ssh", re.IGNORECASE), "rsync+ssh", True),
    # Netcat / socat (reverse-shell risk)
    (re.compile(r"\bnc\b|\bnetcat\b|\bncat\b", re.IGNORECASE), "netcat", True),
    (re.compile(r"\bsocat\b", re.IGNORECASE), "socat", True),
    # DNS lookup tools
    (re.compile(r"\bnslookup\b|\bdig\b|\bhost\b", re.IGNORECASE), "dns-lookup", True),
    # git over SSH or HTTPS
    (re.compile(r"\bgit\s+clone\b.*://", re.IGNORECASE), "git-clone", True),
    (re.compile(r"\bgit\s+push\b|\bgit\s+fetch\b|\bgit\s+pull\b", re.IGNORECASE), "git-remote", False),
    # Python / Node HTTP requests
    (re.compile(r"\bpython[23]?\b.*\brequests\b", re.IGNORECASE), "python-requests", True),
    (re.compile(r"\bnode\b.*\bhttps?:\b", re.IGNORECASE), "node-http", True),
    # FTP
    (re.compile(r"\bftp\b|\bsftp\b", re.IGNORECASE), "ftp", True),
    # Generic http:// or https:// URL presence
    (re.compile(r"https?://([^\s'\"/]+)", re.IGNORECASE), "http-url", True),
]

# Regex that extracts the hostname from common URL patterns in commands
_URL_HOST_RE = re.compile(
    r"(?:https?://|ssh://|git://|ftp://|sftp://|@)([a-zA-Z0-9\-._]+)",
    re.IGNORECASE,
)

# SSH host argument patterns
_SSH_HOST_RE = re.compile(
    r"\bssh\b(?:\s+-\w+\s*\S+)*\s+(?:[^@\s]+@)?([a-zA-Z0-9\-._]+)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NetworkDecision:
    """Result of a network policy check."""
    verdict: str         # ALLOW or DENY
    reason: str          # Human-readable policy reason
    host: str = ""       # Hostname/IP that was checked
    port: Optional[int] = None
    matched_rule: str = ""


@dataclass
class NetworkAuditEntry:
    """Single audit log entry for a network policy decision."""
    timestamp: float
    verdict: str
    host: str
    port: Optional[int]
    matched_rule: str
    command_snippet: str = ""


# ---------------------------------------------------------------------------
# NetworkPolicyEnforcer
# ---------------------------------------------------------------------------

class NetworkPolicyEnforcer:
    

    # Hard-coded always-denied domains regardless of policy.yaml
    _HARDCODED_DENIED_DOMAINS: FrozenSet[str] = frozenset({
        "ngrok.io", "ngrok.com",
        "pastebin.com", "hastebin.com",
        "webhook.site", "requestbin.com", "requestbin.net",
        "hookbin.com", "beeceptor.com",
        "pipedream.net",
        "serveo.net",            # SSH tunneling service
        "localtunnel.me",
        "127.0.0.1.xip.io",
    })

    # Hard-coded always-denied ports regardless of policy.yaml
    _HARDCODED_DENIED_PORTS: FrozenSet[int] = frozenset({
        31337,  # Back Orifice
        4444,   # Common Metasploit default
        1337,   # Common RAT port
        9999,   # Common reverse shell
    })

    _MAX_AUDIT_ENTRIES = 2000

    def __init__(
        self,
        *,
        denied_domains: Optional[List[str]] = None,
        allow_outbound_ssh: bool = False,
        allow_outbound_http: bool = True,
        allowed_ports: Optional[List[int]] = None,
        denied_ports: Optional[List[int]] = None,
        enable_dns_resolution: bool = False,
    ) -> None:
        self._lock = threading.RLock()

        # Build denied domain set (hardcoded + policy.yaml)
        policy_denied = frozenset(
            d.lower().strip().lstrip("*.") for d in (denied_domains or []) if d
        )
        self._denied_domains: FrozenSet[str] = self._HARDCODED_DENIED_DOMAINS | policy_denied

        self._allow_outbound_ssh: bool = allow_outbound_ssh
        self._allow_outbound_http: bool = allow_outbound_http

        # Port policy
        self._allowed_ports: Optional[FrozenSet[int]] = (
            frozenset(int(p) for p in allowed_ports if p) if allowed_ports else None
        )
        policy_denied_ports = frozenset(
            int(p) for p in (denied_ports or []) if p
        )
        self._denied_ports: FrozenSet[int] = self._HARDCODED_DENIED_PORTS | policy_denied_ports

        # Optional DNS resolution for IP-based bypass detection
        self._enable_dns_resolution: bool = enable_dns_resolution

        # Audit log (ring buffer)
        self._audit_log: List[NetworkAuditEntry] = []

        _logger.info(
            "[NetworkPolicyEnforcer] Initialised. "
            "denied_domains=%d allow_ssh=%s allow_http=%s allowed_ports=%s",
            len(self._denied_domains),
            allow_outbound_ssh,
            allow_outbound_http,
            allowed_ports,
        )

    # =========================================================================
    # Factory
    # =========================================================================

    @classmethod
    def from_network_cfg(cls, network_cfg: dict) -> "NetworkPolicyEnforcer":
        
        if not isinstance(network_cfg, dict):
            _logger.warning(
                "[NetworkPolicyEnforcer] network_cfg is not a dict — using safe defaults."
            )
            return cls()

        denied_domains_raw = network_cfg.get("denied_domains", [])
        denied_domains = (
            [str(d) for d in denied_domains_raw if d]
            if isinstance(denied_domains_raw, list) else []
        )

        allow_ssh = bool(network_cfg.get("allow_outbound_ssh", False))
        allow_http = bool(network_cfg.get("allow_outbound_http", True))
        enable_dns = bool(network_cfg.get("enable_dns_resolution", False))

        ap_raw = network_cfg.get("allowed_ports")
        allowed_ports = (
            [int(p) for p in ap_raw if p] if isinstance(ap_raw, list) else None
        )

        dp_raw = network_cfg.get("denied_ports", [])
        denied_ports = (
            [int(p) for p in dp_raw if p] if isinstance(dp_raw, list) else None
        )

        return cls(
            denied_domains=denied_domains,
            allow_outbound_ssh=allow_ssh,
            allow_outbound_http=allow_http,
            allowed_ports=allowed_ports,
            denied_ports=denied_ports,
            enable_dns_resolution=enable_dns,
        )

    # =========================================================================
    # Primary validation entry points
    # =========================================================================

    def validate_host_port(
        self,
        host: str,
        port: Optional[int] = None,
        *,
        protocol: str = "tcp",
    ) -> NetworkDecision:
        
        host = (host or "").strip().lower()
        if not host:
            return self._decide(DENY, host, port, "empty-host", "Empty host rejected")

        # 1. Port-level checks
        if port is not None:
            port_decision = self._check_port(port, protocol)
            if port_decision is not None:
                self._audit(port_decision)
                return port_decision

        # 2. Domain deny list
        domain_decision = self._check_domain(host)
        if domain_decision is not None:
            self._audit(domain_decision)
            return domain_decision

        # 3. Optional DNS resolution — check resolved IPs against private ranges
        if self._enable_dns_resolution and not self._is_ip(host):
            ip_decision = self._check_dns_resolved_ips(host, port)
            if ip_decision is not None:
                self._audit(ip_decision)
                return ip_decision

        decision = self._decide(ALLOW, host, port, "policy-pass", "Connection permitted")
        self._audit(decision)
        return decision

    def validate_command(
        self,
        command: str,
        *,
        command_snippet_for_log: str = "",
    ) -> NetworkDecision:
        
        if not isinstance(command, str) or not command.strip():
            return NetworkDecision(verdict=ALLOW, reason="Empty command", host="", port=None)

        snippet = command_snippet_for_log or command[:80]

        # ── Exfiltration guard (audit fix: curl --data, wget --post-data, etc.) ──
        if _EXFIL_GUARD_AVAILABLE:
            try:
                exfil_result = _exfil_check(command)
                if exfil_result.decision == "DENY":
                    deny_dec = NetworkDecision(
                        verdict=DENY,
                        reason=f"ExfiltrationGuard DENY: {exfil_result.reason}",
                        host="",
                        port=None,
                        matched_rule="exfiltration-guard",
                    )
                    self._audit(deny_dec, command_snippet=snippet)
                    _logger.warning(
                        "[NetworkPolicyEnforcer] ExfilGuard DENY: pattern=%r | cmd=%r",
                        exfil_result.pattern, snippet,
                    )
                    return deny_dec
                elif exfil_result.decision == "REQUIRE_HUMAN_CONFIRMATION":
                    confirm_dec = NetworkDecision(
                        verdict="REQUIRE_HUMAN_CONFIRMATION",
                        reason=f"ExfiltrationGuard: {exfil_result.reason}",
                        host="",
                        port=None,
                        matched_rule="exfiltration-guard-confirm",
                    )
                    _logger.info(
                        "[NetworkPolicyEnforcer] ExfilGuard REQUIRE_HUMAN: pattern=%r | cmd=%r",
                        exfil_result.pattern, snippet,
                    )
                    return confirm_dec
            except Exception as _eg_exc:
                _logger.debug("[NetworkPolicyEnforcer] ExfilGuard error (fail-open): %s", _eg_exc)
        # ── end exfiltration guard ──

        # Extract all hostnames/IPs from command
        hosts_found: List[Tuple[str, Optional[int]]] = []

        # URL pattern extraction
        for m in _URL_HOST_RE.finditer(command):
            hostname = m.group(1).lower().strip()
            port = self._infer_port_from_url(command, m.start())
            if hostname:
                hosts_found.append((hostname, port))

        # SSH-specific host extraction
        for m in _SSH_HOST_RE.finditer(command):
            hostname = m.group(1).lower().strip()
            if hostname:
                hosts_found.append((hostname, 22))

        # Check each found host
        for host, port in hosts_found:
            decision = self.validate_host_port(host, port)
            if decision.verdict == DENY:
                denied_decision = NetworkDecision(
                    verdict=DENY,
                    reason=f"Command blocked: {decision.reason} (host={host!r})",
                    host=host,
                    port=port,
                    matched_rule=decision.matched_rule,
                )
                self._audit(denied_decision, command_snippet=snippet)
                _logger.warning(
                    "[NetworkPolicyEnforcer] DENY command: host=%r port=%r "
                    "rule=%r snippet=%r",
                    host, port, decision.matched_rule, snippet,
                )
                return denied_decision

        # Check for SSH if no URL found but ssh command detected
        if not hosts_found:
            for pat, desc, needs_domain in _COMMAND_NETWORK_PATTERNS:
                if needs_domain and pat.search(command):
                    if desc == "ssh" and not self._allow_outbound_ssh:
                        decision = self._decide(
                            DENY, "", 22, "ssh-denied",
                            "allow_outbound_ssh=false and ssh detected in command"
                        )
                        self._audit(decision, command_snippet=snippet)
                        return decision

        allow_decision = NetworkDecision(
            verdict=ALLOW,
            reason="No denied hosts or protocols detected",
            host="",
            port=None,
            matched_rule="policy-pass",
        )
        return allow_decision

    # =========================================================================
    # Internal checks
    # =========================================================================

    def _check_port(
        self, port: int, protocol: str
    ) -> Optional[NetworkDecision]:
        """Return a DENY decision if port is forbidden, else None."""
        # Hard-coded denied ports
        if port in self._denied_ports:
            return self._decide(
                DENY, "", port, f"hardcoded-denied-port-{port}",
                f"Port {port} is unconditionally denied (known malicious port)"
            )

        # SSH enforcement
        if port == 22 and not self._allow_outbound_ssh:
            return self._decide(
                DENY, "", port, "ssh-denied",
                "allow_outbound_ssh=false: outbound SSH connections are forbidden"
            )

        # HTTP enforcement
        if port == 80 and not self._allow_outbound_http:
            return self._decide(
                DENY, "", port, "http-denied",
                "allow_outbound_http=false: plaintext HTTP (port 80) is forbidden"
            )

        # Allowed-port allowlist
        if self._allowed_ports is not None and port not in self._allowed_ports:
            return self._decide(
                DENY, "", port, "port-not-allowlisted",
                f"Port {port} not in allowed_ports={sorted(self._allowed_ports)}"
            )

        return None  # port is permitted

    def _check_domain(self, host: str) -> Optional[NetworkDecision]:
        """Return DENY if host or any parent domain is in the deny list, else None."""
        host = host.strip().lower()
        if not host:
            return None

        # Exact match
        if host in self._denied_domains:
            return self._decide(
                DENY, host, None, f"denied-domain-exact:{host}",
                f"Host {host!r} is in the denied_domains list"
            )

        # Subdomain match: check each suffix of the host
        parts = host.split(".")
        for i in range(len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in self._denied_domains:
                return self._decide(
                    DENY, host, None, f"denied-domain-subdomain:{parent}",
                    f"Host {host!r} is a subdomain of denied domain {parent!r}"
                )

        return None

    def _check_dns_resolved_ips(
        self, host: str, port: Optional[int]
    ) -> Optional[NetworkDecision]:
        
        try:
            addrs = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
            for addr in addrs:
                ip_str = addr[4][0]
                try:
                    ip = ipaddress.ip_address(ip_str)
                    # Block AWS/GCP/Azure metadata endpoints
                    if ip in ipaddress.ip_network("169.254.0.0/16"):
                        return self._decide(
                            DENY, host, port, "metadata-endpoint",
                            f"Host {host!r} resolves to cloud metadata endpoint {ip_str}"
                        )
                    # Block loopback (127.x) connections to external-looking hostnames
                    # Only flag if the hostname looks external (contains a dot)
                    if ip.is_loopback and "." in host and not host.endswith(".local"):
                        _logger.warning(
                            "[NetworkPolicyEnforcer] Host %r resolves to loopback %s — "
                            "possible DNS rebinding. Allowing but flagging.",
                            host, ip_str,
                        )
                except ValueError:
                    pass
        except Exception as exc:
            # DNS lookup failure: fail-open (don't block on DNS outage)
            _logger.debug(
                "[NetworkPolicyEnforcer] DNS lookup failed for %r: %s — allowing.",
                host, exc,
            )
        return None

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _decide(
        verdict: str,
        host: str,
        port: Optional[int],
        matched_rule: str,
        reason: str,
    ) -> NetworkDecision:
        return NetworkDecision(
            verdict=verdict,
            reason=reason,
            host=host,
            port=port,
            matched_rule=matched_rule,
        )

    def _audit(
        self,
        decision: NetworkDecision,
        *,
        command_snippet: str = "",
    ) -> None:
        entry = NetworkAuditEntry(
            timestamp=time.time(),
            verdict=decision.verdict,
            host=decision.host,
            port=decision.port,
            matched_rule=decision.matched_rule,
            command_snippet=command_snippet,
        )
        with self._lock:
            self._audit_log.append(entry)
            if len(self._audit_log) > self._MAX_AUDIT_ENTRIES:
                self._audit_log = self._audit_log[-self._MAX_AUDIT_ENTRIES :]

    @staticmethod
    def _is_ip(host: str) -> bool:
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False

    @staticmethod
    def _infer_port_from_url(command: str, url_start: int) -> Optional[int]:
        """Extract port number from a URL like https://host:8080/path."""
        port_re = re.compile(r":(\d{1,5})/")
        # Search in the substring starting at url_start
        m = port_re.search(command, url_start)
        if m:
            try:
                p = int(m.group(1))
                if 1 <= p <= 65535:
                    return p
            except ValueError:
                pass
        # Infer from scheme
        if "ssh://" in command[max(0, url_start - 6):url_start + 6]:
            return 22
        if "ftp://" in command[max(0, url_start - 5):url_start + 5]:
            return 21
        if "http://" in command[max(0, url_start - 6):url_start + 6]:
            return 80
        if "https://" in command[max(0, url_start - 7):url_start + 7]:
            return 443
        return None

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_stats(self) -> dict:
        """Return summary statistics for monitoring/diagnostics."""
        with self._lock:
            total = len(self._audit_log)
            denied = sum(1 for e in self._audit_log if e.verdict == DENY)
        return {
            "total_checks": total,
            "denied_count": denied,
            "allow_count": total - denied,
            "denied_domains_count": len(self._denied_domains),
            "allow_outbound_ssh": self._allow_outbound_ssh,
            "allow_outbound_http": self._allow_outbound_http,
        }

    def get_recent_denials(self, n: int = 10) -> List[dict]:
        """Return the n most recent DENY decisions for debugging."""
        with self._lock:
            denials = [e for e in self._audit_log if e.verdict == DENY]
            recent = denials[-n:]
        return [
            {
                "timestamp": e.timestamp,
                "host": e.host,
                "port": e.port,
                "matched_rule": e.matched_rule,
                "command_snippet": e.command_snippet,
            }
            for e in recent
        ]
