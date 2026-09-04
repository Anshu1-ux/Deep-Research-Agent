"""
Tests for should_continue_loop -- this is the single most important function
in the project, since a bug here is what turns a bounded reflection loop
into an unbounded (expensive, possibly infinite) one.
"""

from src.nodes.critic import should_continue_loop
from src import config


def make_state(iteration: int, score_history: list[float]) -> dict:
    return {"iteration": iteration, "score_history": score_history, "stop_reason": ""}


def test_loops_when_score_below_threshold_and_iterations_remain():
    state = make_state(iteration=1, score_history=[0.4])
    assert should_continue_loop(state) == "researcher"


def test_stops_when_threshold_met():
    state = make_state(iteration=1, score_history=[0.85])
    assert should_continue_loop(state) == "finalizer"
    assert state["stop_reason"] == "threshold_met"


def test_stops_at_max_iterations_regardless_of_score():
    state = make_state(iteration=config.MAX_ITERATIONS, score_history=[0.1, 0.2, 0.3])
    assert should_continue_loop(state) == "finalizer"
    assert state["stop_reason"] == "max_iterations"


def test_stops_on_diminishing_returns():
    # improvement of 0.01 is below MIN_SCORE_IMPROVEMENT (0.05)
    state = make_state(iteration=2, score_history=[0.60, 0.61])
    assert should_continue_loop(state) == "finalizer"
    assert state["stop_reason"] == "diminishing_returns"


def test_continues_when_improvement_is_meaningful():
    state = make_state(iteration=1, score_history=[0.40, 0.60])
    assert should_continue_loop(state) == "researcher"
