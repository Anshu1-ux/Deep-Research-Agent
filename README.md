# Deep Research Agent

A LangGraph agent that plans a query into sub-questions, researches each one
against live web search **and** a local document corpus, critiques its own
findings, and loops back to fill gaps -- with three independent, hard-coded
stop conditions so the reflection loop can never run away. Checkpointed to
Postgres so any run can be paused, inspected, or resumed.

## Architecture

```
                    ┌─────────────┐
   query ─────────► │   Planner   │  (splits into ≤5 sub-questions)
                    └──────┬──────┘
                           │
                           ▼
              ┌───►┌─────────────┐
              │    │ Researcher  │  (web search + local doc retrieval)
              │    └──────┬──────┘
              │           ▼
              │    ┌─────────────┐
              │    │   Critic    │  (scores completeness/groundedness, lists gaps)
              │    └──────┬──────┘
              │           │
        loop if score < 0.7 AND iteration < 3 AND improving
              │           │
              └───────────┤
                           │ else
                           ▼
                    ┌─────────────┐
                    │  Finalizer  │  (synthesizes cited report, discloses gaps)
                    └─────────────┘
```

Every node reads/writes a single shared `ResearchState` (see `src/state.py`)
rather than passing ad hoc arguments -- this is what makes Postgres
checkpointing work: LangGraph serializes that state after every node.

## Project layout

```
deep-research-agent/
├── src/
│   ├── config.py        # all tunables, including the loop-bound constants
│   ├── state.py           # shared typed state schema
│   ├── llm.py               # provider-agnostic LLM call (ollama/openai)
│   ├── graph.py               # StateGraph wiring + Postgres checkpointer
│   ├── cli.py                   # run from the command line
│   ├── tools/
│   │   ├── web_search.py          # Tavily web search
│   │   └── local_docs.py            # Chroma-indexed local corpus retrieval
│   └── nodes/
│       ├── planner.py                 # query -> bounded sub-questions
│       ├── researcher.py                # evidence-gathering + drafting
│       ├── critic.py                      # scoring + the stop-condition logic
│       └── finalizer.py                     # synthesis + gap disclosure
├── tests/
│   └── test_critic_loop.py    # covers the stop-condition logic specifically
├── data/local_corpus/         # drop .txt/.md files here to index
├── docker-compose.yml           # local Postgres for checkpointing
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add TAVILY_API_KEY at minimum

docker compose up -d              # starts Postgres on localhost:5442
python -m src.tools.local_docs --index   # index data/local_corpus/
```

## Run it

```bash
python -m src.cli "What are the tradeoffs between RRF and weighted score fusion in hybrid retrieval?"
```

Output shows the thread ID, how many iterations ran, the full score
trajectory, and the final report. Re-run with `--thread-id <id>` to resume
that exact run from its last Postgres checkpoint instead of starting over.

## Design decisions (interview-ready reasoning)

| Decision | Why |
|---|---|
| Three independent loop-stop conditions (max iterations, score threshold, diminishing returns) | Any single condition alone is fragile -- a miscalibrated scorer defeats a threshold-only stop; a slow-but-real improvement defeats a returns-only stop. Requiring the loop to check all three is what makes it robust to any one of them misbehaving. |
| `MAX_ITERATIONS = 3` is a hard ceiling, not a suggestion | The critic's own score is the thing deciding whether to keep going -- if that score is ever wrong (bad prompt, bad LLM day), the loop needs a backstop that doesn't depend on the same fallible signal. |
| Researcher only re-researches sub-questions the critic flagged as gaps | Keeps iteration 2 and 3 cheap. Re-running all N sub-questions every round would make the loop's cost scale with iterations × sub-questions instead of iterations × gaps, which is the difference between a $0.30 run and a $2 run. |
| Postgres checkpointing, not in-memory | A multi-iteration research run can take minutes; if the process crashes or you want to inspect state mid-run, in-memory state is just gone. Checkpointing to Postgres makes runs resumable and inspectable -- a real production pattern for long-running agent jobs. |
| Finalizer discloses *why* it stopped when it wasn't `threshold_met` | An agent that silently presents a "max iterations hit" answer identically to a "fully satisfied" answer is misleading by omission. Making the stop reason visible in the output is a small design choice that says a lot about how the system was built. |
| `score_history` accumulates (never overwrites) | Lets you audit the whole trajectory (e.g. `[0.52, 0.68, 0.71]`) instead of just the final number -- proves the loop was making real progress, which matters both for debugging and for explaining the system's behavior. |

## Extending this project

- Add a second, deliberately "hard" query designed to hit `max_iterations`
  and one designed to hit `threshold_met` on iteration 1 -- capture both
  score trajectories as evidence the bounds actually work as intended.
- Add a `--dry-run-cost` flag that estimates LLM + search API cost per run
  before executing, given the sub-question count and iteration cap.
- Swap `call_llm()` in `src/llm.py` for IBM watsonx.ai to match the rest of
  your stack -- same one-function swap pattern as project 1.
