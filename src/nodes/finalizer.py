# runs once at the end, synthesizes everything into one report.
# if we stopped early (max iters / diminishing returns, not because the
# critic was actually satisfied) we say so in the output instead of just
# pretending the answer is complete

from __future__ import annotations

from src.llm import call_llm
from src.state import ResearchState

FINALIZER_PROMPT = """Synthesize the following research findings into a single \
coherent, well-organized report answering the original query. Preserve \
citation markers from the findings. Use clear section headers per theme, \
not necessarily one section per sub-question.

Original query: {query}

Findings:
{findings}
{gap_disclosure}

Write the final report now.
"""


def finalizer_node(state: ResearchState) -> dict:
    findings_block = "\n\n".join(
        f"Q: {sq['question']}\nFindings: {sq['findings']}\nSources: {', '.join(sq['sources'])}"
        for sq in state["sub_questions"]
    )

    gap_disclosure = ""
    stop_reason = state.get("stop_reason", "threshold_met")
    feedback = state.get("critic_feedback")
    if stop_reason != "threshold_met" and feedback and feedback["gaps"]:
        gap_disclosure = (
            "\n\nNote: research stopped due to "
            f"'{stop_reason}' before every gap was resolved. Add a short "
            "closing section titled 'Known Limitations' listing these "
            f"unresolved gaps: {', '.join(feedback['gaps'])}"
        )

    prompt = FINALIZER_PROMPT.format(
        query=state["query"], findings=findings_block, gap_disclosure=gap_disclosure
    )
    report = call_llm(prompt, temperature=0.2)

    return {"final_report": report}
