# langgraph_pipeline/build_graph.py
from langgraph.graph import StateGraph, START, END

# Use MemorySaver for session-based persistence, or replace with a persistent one like SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

# Import the state definition
from .graph_state import RivalryState

# Import the node functions
from .nodes import call_openai_node, format_gemini_prompt_node, call_gemini_node
import traceback  # For error logging


# --- Build the Actual LangGraph ---
def build_rivalry_graph():
    """
    Builds and compiles the LangGraph StateGraph for the LLM rivalry pipeline.

    Returns:
        CompiledStateGraph: The compiled LangGraph ready for execution.
    """
    print("Building LangGraph workflow...")
    workflow = StateGraph(RivalryState)

    # Add nodes to the graph
    print("Adding nodes: call_openai, format_gemini_prompt, call_gemini")
    workflow.add_node("call_openai", call_openai_node)
    workflow.add_node("format_gemini_prompt", format_gemini_prompt_node)
    workflow.add_node("call_gemini", call_gemini_node)

    # Define the execution flow using edges
    print(
        "Adding edges: START -> call_openai -> format_gemini_prompt -> call_gemini -> END"
    )
    workflow.add_edge(START, "call_openai")
    workflow.add_edge("call_openai", "format_gemini_prompt")
    workflow.add_edge("format_gemini_prompt", "call_gemini")

    # Define the end point after Gemini runs
    # In a real scenario, you might have a node for composite draft saving
    # or other final steps before ending.
    workflow.add_edge("call_gemini", END)

    # Set up memory saver for checkpointing (persists state within the session)
    # Replace MemorySaver with a persistent checkpointer (e.g., SqliteSaver, PostgresSaver)
    # if you need persistence across app restarts or users.
    # checkpointer = SqliteSaver.from_conn_string(":memory:") # Example using SQLite in-memory
    checkpointer = MemorySaver()
    print(f"Using checkpointer: {type(checkpointer).__name__}")

    # Compile the graph
    # The checkpointer enables LangGraph to save/load state between steps
    # and potentially resume interrupted runs.
    try:
        compiled_graph = workflow.compile(checkpointer=checkpointer)
        print("LangGraph rivalry graph compiled successfully.")
        return compiled_graph
    except Exception as e:
        print(f"Error compiling LangGraph: {e}\n{traceback.format_exc()}")
        raise  # Re-raise the error after printing


# --- Helper function to get the graph ---
# Removed the use_mock flag as we are directly building the real graph now.
def get_graph():
    """
    Builds and returns the compiled LangGraph.

    Returns:
        CompiledStateGraph: The graph object.
    """
    print("Attempting to build actual LangGraph...")
    # Ensure you have LangGraph installed and configured properly
    try:
        # This is where you'd import and call the actual build function
        return build_rivalry_graph()
    except ImportError as e:
        # This error should ideally be caught earlier during client init,
        # but good to have a fallback message.
        print(f"LangGraph components not found or import error during build: {e}.")
        raise  # Re-raise import errors related to LangGraph itself
    except Exception as e:
        print(f"Error building LangGraph: {e}\n{traceback.format_exc()}")
        raise  # Re-raise other compilation errors
