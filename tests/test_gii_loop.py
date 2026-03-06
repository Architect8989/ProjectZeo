"""
tests/test_gii_loop.py

UNIT TESTS: GIIGoalDirectedLoop — stagnation, denial propagation, goal completion
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import time
from unittest.mock import MagicMock, call

from core.gii.gii_loop import GIIGoalDirectedLoop


def make_gii_controller(actions):
    """Create a mock GIIController that returns actions in sequence."""
    call_count = [0]
    gii = MagicMock()

    def decide_next_action(world_state, **kwargs):
        idx = min(call_count[0], len(actions) - 1)
        call_count[0] += 1
        action = actions[idx]
        if action is None:
            return None, "No action"
        return action, "test decision"

    gii.decide_next_action.side_effect = decide_next_action
    gii.record_denial = MagicMock()
    gii.record_outcome = MagicMock()
    return gii


def make_policy_engine(decision="ALLOW"):
    pe = MagicMock()
    pe.validate_action_dict.return_value = (decision, "test reason")
    return pe


def make_loop(gii, policy=None, **kwargs):
    if policy is None:
        policy = make_policy_engine()
    journal = MagicMock()
    journal.record = MagicMock()
    return GIIGoalDirectedLoop(
        gii_controller=gii,
        os_backend=None,
        world_graph=None,
        policy_engine=policy,
        journal=journal,
        objective="test objective",
        execute_decision_fn=lambda action, ws: {"success": True, "output": "ok"},
        **kwargs,
    )


class TestGIILoopGoalCompletion:

    def test_done_action_completes_loop(self):
        actions = [
            {"operation": "click", "x": 0.5, "y": 0.5},
            {"operation": "done", "summary": "Task complete"},
        ]
        gii = make_gii_controller(actions)
        loop = make_loop(gii)
        result = loop.run()
        assert result["success"] is True
        assert "Task complete" in result["reason"]

    def test_no_action_increments_stagnation(self):
        actions = [None, None, None]
        gii = make_gii_controller(actions)
        loop = make_loop(gii, max_stagnant=2)
        result = loop.run()
        assert result["success"] is False
        assert "Stagnation" in result["reason"]

    def test_timeout_terminates_loop(self):
        def slow_decide(world_state, **kwargs):
            time.sleep(0.1)
            return {"operation": "click", "x": 0.5}, "ok"

        gii = MagicMock()
        gii.decide_next_action.side_effect = slow_decide
        gii.record_denial = MagicMock()
        gii.record_outcome = MagicMock()

        loop = make_loop(gii, max_wallclock_seconds=0)  # Instant timeout
        result = loop.run(start_ts=time.time() - 10)  # Already past timeout
        assert result["success"] is False
        assert "timeout" in result["reason"].lower()

    def test_policy_deny_blocks_action(self):
        actions = [
            {"operation": "command", "command": "rm -rf /"},
            {"operation": "done", "summary": "done"},
        ]
        gii = make_gii_controller(actions)
        policy = make_policy_engine("DENY")
        loop = make_loop(gii, policy=policy, max_stagnant=3)
        result = loop.run()
        # After 3 denials → stagnation
        assert result["success"] is False
        gii.record_denial.assert_called()

    def test_repeated_action_detected_as_stagnation(self):
        """Same action key repeated 3+ times must trigger stagnation."""
        repeated_action = {"operation": "click", "x": 0.5, "y": 0.5}
        actions = [repeated_action] * 10
        gii = make_gii_controller(actions)
        loop = make_loop(gii, max_stagnant=5)
        result = loop.run()
        assert result["success"] is False
        assert result["stagnant_count"] > 0

    def test_max_iterations_terminates(self):
        actions = [{"operation": "click", "x": 0.5, "y": 0.5}]
        gii = make_gii_controller(actions)
        loop = make_loop(gii, max_iterations=3, max_stagnant=1000)
        result = loop.run()
        assert result["success"] is False
        assert "iterations" in result["reason"].lower()
        assert result["iterations"] == 3


class TestApprovalSemantics:

    def test_approval_signal_path_not_deleted(self):
        """AUDIT-CRITICAL-4: Approval must require CREATE, not DELETE."""
        import core.operate.operate as operate_mod  # noqa: import check
        # Verify the approval logic uses .APPROVE suffix (create-to-approve)
        with open(os.path.join(os.path.dirname(__file__), "../operate/operate.py")) as f:
            source = f.read()
        assert "_approve_path" in source, (
            "AUDIT FAILURE: Approval inversion not applied. "
            "Expected create-to-approve (_approve_path) mechanism."
        )
        assert "APPROVE" in source, "APPROVE file mechanism not found in operate.py"
        # The old "if not _file_present: _phc_approved = True" (delete-to-approve) should be gone
        assert "_approve_present" in source, (
            "AUDIT FAILURE: delete-to-approve logic still present. "
            "Must check for file PRESENCE (create-to-approve), not ABSENCE."
        )
