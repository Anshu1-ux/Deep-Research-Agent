"""
Command-line entry point.

    python -m src.cli "What are the tradeoffs of RRF vs weighted score fusion?"

Each run gets a unique thread_id, which is what Postgres checkpointing keys
on -- re-running with the SAME thread_id (via --thread-id) resumes/replays
that run from its last checkpoint instead of starting over.
"""

from __future__ import annotations

import argparse
import uuid

from src.graph import get_checkpointed_graph


def run(query: str, thread_id: str | None = None) -> None:
    thread_id = thread_id or str(uuid.uuid4())
    print(f"[thread_id: {thread_id}]\n")

    with get_checkpointed_graph() as app:
        result = app.invoke(
            {"query": query},
            config={"configurable": {"thread_id": thread_id}},
        )

    print(f"Stopped after {result['iteration']} iteration(s) "
          f"({result.get('stop_reason', 'unknown')}). "
          f"Score history: {result['score_history']}\n")
    print("=" * 80)
    print(result["final_report"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Research question to investigate")
    parser.add_argument("--thread-id", default=None, help="Resume a specific run")
    args = parser.parse_args()
    run(args.query, args.thread_id)
