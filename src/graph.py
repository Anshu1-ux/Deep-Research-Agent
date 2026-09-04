# planner -> researcher -> critic -> (loop back or move to finalizer)
# checkpointed to postgres so a run can be resumed if it crashes/gets killed
# mid-loop, don't want to lose 2 iterations of research to a laptop sleep

from __future__ import annotations

from contextlib import contextmanager

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from src import config
from src.state import ResearchState
from src.nodes.planner import planner_node
from src.nodes.researcher import researcher_node
from src.nodes.critic import critic_node, should_continue_loop
from src.nodes.finalizer import finalizer_node


def build_graph_definition() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalizer", finalizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        should_continue_loop,
        {"researcher": "researcher", "finalizer": "finalizer"},
    )
    graph.add_edge("finalizer", END)

    return graph


@contextmanager
def get_checkpointed_graph():
    """
    Context manager so callers get a compiled, Postgres-checkpointed graph
    with the connection cleaned up automatically:

        with get_checkpointed_graph() as app:
            result = app.invoke({...}, config={"configurable": {"thread_id": "..."}})
    """
    with PostgresSaver.from_conn_string(config.POSTGRES_URI) as checkpointer:
        checkpointer.setup()  # idempotent -- creates tables on first run
        graph = build_graph_definition()
        yield graph.compile(checkpointer=checkpointer)
