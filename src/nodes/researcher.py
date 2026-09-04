"""
Researcher node. For each sub-question, pulls evidence from both the web
and the local document corpus, then asks the LLM to synthesize findings
grounded in that evidence. Re-entered on every loop iteration -- on repeat
visits, it also incorporates the critic's gap list from the prior round so
each pass targets what's actually missing, rather than re-researching
everything from scratch.
"""

from __future__ import annotations

from src.llm import call_llm
from src.state import ResearchState
from src.tools.local_docs import search_local_docs
from src.tools.web_search import web_search

RESEARCH_PROMPT = """Answer this sub-question using ONLY the evidence below. \
Cite sources inline like [1], [2] matching the numbered list. If the evidence \
doesn't fully answer it, say so explicitly rather than guessing.

Sub-question: {question}
{gap_context}

Evidence:
{evidence}

Write a concise, well-grounded set of findings (not a full report, just the \
notes for this sub-question).
"""


def researcher_node(state: ResearchState) -> dict:
    feedback = state.get("critic_feedback")
    gap_lookup = set(feedback["gaps"]) if feedback else set()

    updated_sub_questions = []
    for sq in state["sub_questions"]:
        # On re-visits, skip sub-questions the critic didn't flag as gaps --
        # this is what keeps later iterations cheap instead of re-running
        # every sub-question every round.
        if state["iteration"] > 0 and sq["question"] not in gap_lookup:
            updated_sub_questions.append(sq)
            continue

        web_results = web_search(sq["question"])
        local_results = search_local_docs(sq["question"])

        evidence_lines = []
        sources = []
        for i, r in enumerate(web_results, start=1):
            evidence_lines.append(f"[{i}] (web: {r['url']}) {r['content']}")
            sources.append(r["url"])
        offset = len(web_results)
        for i, r in enumerate(local_results, start=1):
            evidence_lines.append(f"[{offset + i}] (local: {r['source']}) {r['content']}")
            sources.append(r["source"])

        gap_context = ""
        if feedback and sq["question"] in gap_lookup:
            gap_context = f"\nNote: a previous review found this gap to address: {feedback['reasoning']}"

        prompt = RESEARCH_PROMPT.format(
            question=sq["question"],
            gap_context=gap_context,
            evidence="\n".join(evidence_lines) or "(no evidence found)",
        )
        findings = call_llm(prompt, temperature=0.1)

        updated_sub_questions.append(
            {"question": sq["question"], "findings": findings, "sources": sources}
        )

    return {"sub_questions": updated_sub_questions}
