"""
Central configuration for the Deep Research Agent.

The critic-loop parameters below are the most important values in this file.
An unbounded reflection loop is the single biggest way this kind of agent
blows its time/cost budget -- these three settings exist specifically to
prevent that, and each is deliberately a hard ceiling, not a suggestion.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

LOCAL_CORPUS_DIR = ROOT_DIR / "data" / "local_corpus"

# ---------------------------------------------------------------------------
# Critic loop bounds (see README "Design decisions" for the reasoning)
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 3            # hard ceiling -- loop stops here no matter what
SCORE_THRESHOLD = 0.7         # loop continues only if score is below this
MIN_SCORE_IMPROVEMENT = 0.05  # stop early if the last round barely helped

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
WEB_SEARCH_RESULTS_PER_SUBQUESTION = 4
LOCAL_DOCS_TOP_K = 4
MAX_SUBQUESTIONS = 5           # planner is capped too -- bounds fan-out cost

# ---------------------------------------------------------------------------
# Embeddings / local corpus indexing
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = ROOT_DIR / "storage" / "chroma"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")   # "ollama" | "openai"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE_PLANNER = 0.2
LLM_TEMPERATURE_RESEARCHER = 0.1
LLM_TEMPERATURE_CRITIC = 0.0   # scoring should be as deterministic as possible

# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
# TAVILY_API_KEY read from environment by src/tools/web_search.py directly.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ---------------------------------------------------------------------------
# Postgres checkpointing
# ---------------------------------------------------------------------------
POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://research_agent:research_agent@localhost:5442/research_agent",
)
