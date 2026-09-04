"""
Shared state schema for the research graph.

LangGraph nodes read from and write to this single typed dict. Keeping it
here (rather than scattered dataclasses per node) makes the state transitions
easy to reason about and easy to explain in an interview: every node's
job is "read some of these fields, write some of these fields."
"""

from __future__ import annotations

from typing import TypedDict, Annotated
import operator


class SubQuestion(TypedDict):
    question: str
    findings: str          # accumulated research notes for this sub-question
    sources: list[str]      # URLs / doc paths cited


class CriticFeedback(TypedDict):
    score: float             # 0.0-1.0, completeness + groundedness combined
    gaps: list[str]          # specific missing pieces, used to re-research
    reasoning: str


class ResearchState(TypedDict):
    # input
    query: str

    # planner output
    sub_questions: list[SubQuestion]

    # loop bookkeeping -- this is what makes the critic loop auditable.
    # Every iteration's score is appended, never overwritten, so the final
    # report (and any debugging session) can show the full trajectory:
    # e.g. [0.52, 0.68, 0.71] proves the loop was making real progress,
    # not just spinning.
    iteration: int
    score_history: Annotated[list[float], operator.add]
    critic_feedback: CriticFeedback | None

    # output
    final_report: str
    stop_reason: str   # "threshold_met" | "max_iterations" | "diminishing_returns"
