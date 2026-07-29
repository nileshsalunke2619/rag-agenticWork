# =============================================================================
# graph.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This file wires our two nodes (from nodes.py) together into an actual
# LangGraph "graph" - a simple flowchart describing the ORDER in which
# things happen:
#
#     START -> retrieve -> generate -> END
#
# app.py never needs to know about individual nodes or how they connect -
# it just imports the finished, compiled graph from this file and calls
# .invoke() on it.
# =============================================================================

from langgraph.graph import END, START, StateGraph

from nodes import GraphState, generate_node, retrieve_node


def build_graph():
    """
    WHAT THIS FUNCTION DOES:
    Builds and "compiles" our LangGraph graph, then returns it ready to
    use.

    STEP BY STEP:
    1. StateGraph(GraphState) creates a new, empty graph that will pass
       around a GraphState object (see nodes.py) between steps.
    2. add_node(name, function) registers each of our node functions
       under a name we can reference when drawing edges.
    3. add_edge(from, to) draws an arrow from one step to the next,
       describing the order of execution.
    4. compile() turns this description into an actual runnable graph.

    WHY START -> retrieve -> generate -> END (and nothing else):
    - START always goes to "retrieve" first, because we can't generate
      an answer before we've fetched the relevant chunks.
    - "retrieve" always goes to "generate" next - there's only one
      possible next step, so no decision-making (no "conditional edge")
      is needed.
    - "generate" always goes to END, because once Claude has produced
      an answer, there's nothing left to do.
    """
    workflow = StateGraph(GraphState)

    # Register our two node functions under simple string names.
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Describe the (fixed, linear) order they run in.
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # compile() validates the graph and returns an object with an
    # .invoke(state) method we can call to actually run it.
    return workflow.compile()


# We build the graph once, when this module is first imported, and reuse
# the same compiled graph for every request - there's no need to rebuild
# it every time someone calls the API.
rag_graph = build_graph()
