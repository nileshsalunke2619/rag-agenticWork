# =============================================================================
# app/graph/graph.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This file wires our nodes (from app/graph/nodes/rag_nodes.py) into an
# actual LangGraph "graph" - a flowchart describing the ORDER in which
# things happen, now with a conditional BRANCH:
#
#     START -> retrieve -> detect_conflict --(no conflict)--> generate_single -> END
#                                           \--(conflict)----> generate_conflict -> END
#
# app/api/routes/chat.py never needs to know about individual nodes or
# how they connect - it just imports the finished, compiled graph from
# this file and calls .invoke() on it. The final state still has
# `answer` and `answeroption` filled in either way, so chat.py and
# app/models/schemas.py did not need to change.
# =============================================================================

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.rag_nodes import (
    GraphState,
    detect_conflict_node,
    generate_conflict_node,
    generate_single_node,
    retrieve_node,
    route_after_conflict_detection,
)


def build_graph():
    """
    Builds and "compiles" our LangGraph graph, then returns it ready to
    use. retrieve and detect_conflict always run; detect_conflict then
    branches to EXACTLY ONE of generate_single / generate_conflict via a
    conditional edge - never both, never neither.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("detect_conflict", detect_conflict_node)
    workflow.add_node("generate_single", generate_single_node)
    workflow.add_node("generate_conflict", generate_conflict_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "detect_conflict")

    # add_conditional_edges: after "detect_conflict" runs, call
    # route_after_conflict_detection(state) - whatever string it returns
    # ("conflict" or "no_conflict") is looked up in this path_map to pick
    # the next node.
    workflow.add_conditional_edges(
        "detect_conflict",
        route_after_conflict_detection,
        {
            "conflict": "generate_conflict",
            "no_conflict": "generate_single",
        },
    )

    workflow.add_edge("generate_single", END)
    workflow.add_edge("generate_conflict", END)

    return workflow.compile()


# We build the graph once, when this module is first imported, and reuse
# the same compiled graph for every request.
rag_graph = build_graph()
