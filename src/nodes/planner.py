"""
Planner node. Runs once, at the start of the graph. Breaks the user's query
into a bounded set of sub-questions -- bounded because an uncapped planner
is just as capable of blowing the budget as an uncapped critic loop; it's
the same failure mode at a different stage.
"""

from __future__ import annotations

from src import config
from src.llm import call_llm, parse_json_response
from src.state import ResearchState

PLANNER_PROMPT = """You are a research planner. Break the following query into \
{max_subquestions} or fewer focused sub-questions that, together, would let \
someone write a thorough, well-grounded answer.

Query: {query}

Respond with ONLY a JSON object of this exact shape, no other text:
{{"sub_questions": ["...", "..."]}}
"""


def planner_node(state: ResearchState) -> dict:
    prompt = PLANNER_PROMPT.format(
        max_subquestions=config.MAX_SUBQUESTIONS, query=state["query"]
    )
    raw = call_llm(prompt, temperature=config.LLM_TEMPERATURE_PLANNER, json_mode=True)
    parsed = parse_json_response(raw)

    questions = parsed["sub_questions"][: config.MAX_SUBQUESTIONS]
    sub_questions = [
        {"question": q, "findings": "", "sources": []} for q in questions
    ]

    return {
        "sub_questions": sub_questions,
        "iteration": 0,
        "score_history": [],
    }
