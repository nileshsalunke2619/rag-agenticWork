# =============================================================================
# app/graph/graph.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This file wires our two nodes (from app/graph/nodes/rag_nodes.py) into
# an actual LangGraph "graph" - a simple flowchart describing the ORDER
# in which things happen:
#
#     START -> retrieve -> generate -> END
#
# app/api/routes/chat.py never needs to know about individual nodes or
# how they connect - it just imports the finished, compiled graph from
# this file and calls .invoke() on it.
# =============================================================================

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.rag_nodes import GraphState, generate_node, retrieve_node


def build_graph():
    """
    Builds and "compiles" our LangGraph graph, then returns it ready to
    use. START -> retrieve -> generate -> END, with no branching and no
    loops - a deliberately simple, fixed pipeline.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# We build the graph once, when this module is first imported, and reuse
# the same compiled graph for every request.
rag_graph = build_graph()
