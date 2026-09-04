"""
Critic node. Scores the current findings on completeness and groundedness,
and lists specific gaps. This score is what the graph's conditional edge
uses to decide whether to loop back to the researcher or move on to the
finalizer -- see src/graph.py for the actual stop conditions.
"""

from __future__ import annotations

from src import config
from src.llm import call_llm, parse_json_response
from src.state import ResearchState

CRITIC_PROMPT = """You are reviewing research findings for completeness and \
groundedness before they're turned into a final report.

Original query: {query}

Findings so far:
{findings}

Score from 0.0 to 1.0 how complete and well-grounded these findings are \
(1.0 = fully answers the query with solid evidence, no gaps).
List any specific sub-questions whose findings are still weak, missing \
evidence, or too shallow -- copy the exact sub-question text for each gap.

Respond with ONLY a JSON object of this exact shape, no other text:
{{"score": 0.0, "gaps": ["exact sub-question text", "..."], "reasoning": "..."}}
"""


def critic_node(state: ResearchState) -> dict:
    findings_block = "\n\n".join(
        f"Q: {sq['question']}\nFindings: {sq['findings']}"
        for sq in state["sub_questions"]
    )
    prompt = CRITIC_PROMPT.format(query=state["query"], findings=findings_block)
    raw = call_llm(prompt, temperature=config.LLM_TEMPERATURE_CRITIC, json_mode=True)
    parsed = parse_json_response(raw)

    feedback = {
        "score": float(parsed["score"]),
        "gaps": parsed.get("gaps", []),
        "reasoning": parsed.get("reasoning", ""),
    }

    return {
        "critic_feedback": feedback,
        "score_history": [feedback["score"]],
        "iteration": state["iteration"] + 1,
    }


def should_continue_loop(state: ResearchState) -> str:
    # conditional edge fn for the graph. 3 independent stop conditions,
    # any one of them ends the loop - see README for why it's 3 and not 1
    scores = state["score_history"]
    latest_score = scores[-1]

    if state["iteration"] >= config.MAX_ITERATIONS:
        state["stop_reason"] = "max_iterations"
        return "finalizer"

    if latest_score >= config.SCORE_THRESHOLD:
        state["stop_reason"] = "threshold_met"
        return "finalizer"

    if len(scores) >= 2:
        improvement = latest_score - scores[-2]
        if improvement < config.MIN_SCORE_IMPROVEMENT:
            state["stop_reason"] = "diminishing_returns"
            return "finalizer"

    return "researcher"
