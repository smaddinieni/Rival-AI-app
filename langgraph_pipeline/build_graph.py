# langgraph_pipeline/build_graph.py
from langgraph.graph import StateGraph, START, END

# Use MemorySaver for session-based persistence, or replace with a persistent one like SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

# Import the state definition
from .graph_state import RivalryState

# Import the node functions
from .nodes import call_openai_node, format_gemini_prompt_node, call_gemini_node

# Import the mock classes if needed for testing without full LangGraph setup
from .llm_clients import MockLLM  # Assuming MockLLM is in llm_clients.py
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)  # Needed for mock graph state
import uuid
from datetime import datetime


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
        print(f"Error compiling LangGraph: {e}")
        raise  # Re-raise the error after printing


# --- Mock Compiled Graph (For UI testing without full LangGraph setup) ---
# This allows testing the Streamlit UI structure and flow even if
# LangGraph or its dependencies aren't fully set up or if API keys are missing.
class MockCompiledGraph:
    """A mock class simulating the compiled LangGraph interface."""

    def __init__(self):
        # Store nodes in order for simple sequential execution simulation
        self.nodes_in_order = [
            call_openai_node,
            format_gemini_prompt_node,
            call_gemini_node,
        ]
        # Include a mock checkpointer if needed for testing state logic
        self.checkpointer = (
            MemorySaver()
        )  # Use the real MemorySaver for state format compatibility
        print("Initialized MockCompiledGraph for UI testing.")

    def invoke(self, initial_state: dict, config: dict = None) -> dict:
        """Simulates invoking the graph sequentially."""
        current_state = initial_state.copy()
        thread_id = config.get("configurable", {}).get(
            "thread_id", "mock_invoke_thread"
        )
        print(f"\n--- Invoking Mock Graph (Thread: {thread_id}) ---")
        # Simulate loading state (not fully implemented here)
        # checkpoint = self.checkpointer.get(config)
        # if checkpoint: current_state = checkpoint.copy()

        print(f"Initial State: {self._format_state_for_log(current_state)}")

        for node_func in self.nodes_in_order:
            node_name = node_func.__name__
            if current_state.get("status") == "error":
                print(f"Skipping mock node {node_name} due to error state.")
                break
            print(f"Executing mock node: {node_name}")
            try:
                update = node_func(current_state)  # Call the actual node function
                current_state.update(update)  # Apply updates
                print(
                    f"State after {node_name}: {self._format_state_for_log(current_state)}"
                )
                # Simulate saving state (not fully implemented here)
                # self.checkpointer.put(config, current_state, {})
            except Exception as e:
                error_msg = f"Error in mock node {node_name}: {e}"
                print(error_msg)
                current_state["status"] = "error"
                current_state["error_message"] = error_msg
                # Optionally save error state
                # self.checkpointer.put(config, current_state, {})
                break  # Stop on error

        print(f"--- Mock Graph Invocation Complete (Thread: {thread_id}) ---")
        return current_state

    def stream(self, initial_state: dict, config: dict = None):
        """Simulates streaming updates from the graph."""
        current_state = initial_state.copy()
        thread_id = config.get("configurable", {}).get(
            "thread_id", "mock_stream_thread"
        )
        print(f"\n--- Streaming Mock Graph (Thread: {thread_id}) ---")
        print(f"Initial State: {self._format_state_for_log(current_state)}")
        yield {"initial_state_stream": current_state}  # Example initial yield

        for node_func in self.nodes_in_order:
            node_name = node_func.__name__
            if current_state.get("status") == "error":
                print(f"Streaming interrupted due to error state before {node_name}.")
                # Yield the error state update
                yield {
                    node_name: {
                        "status": "error",
                        "error_message": current_state.get("error_message"),
                    }
                }
                break  # Stop streaming on error

            print(f"Executing mock node for stream: {node_name}")
            try:
                update = node_func(current_state)  # Call the actual node function
                if update:  # Only update and yield if there's an update
                    current_state.update(update)
                    print(
                        f"Streaming update from {node_name}: {self._format_state_for_log(update)}"
                    )
                    yield {node_name: update}
                    # Simulate saving state during stream if necessary
                    # self.checkpointer.put(config, current_state, {})
                else:
                    print(f"No update from mock node: {node_name}")
            except Exception as e:
                error_msg = f"Error in mock node {node_name}: {e}"
                print(error_msg)
                current_state["status"] = "error"
                current_state["error_message"] = error_msg
                # Yield the error state update
                yield {node_name: {"status": "error", "error_message": error_msg}}
                # Optionally save error state
                # self.checkpointer.put(config, current_state, {})
                break  # Stop streaming on error

        print(f"--- Mock Graph Stream Complete (Thread: {thread_id}) ---")

    def _format_state_for_log(self, state_dict):
        """Helper to format state for cleaner logging, truncating long strings."""
        log_state = {}
        for k, v in state_dict.items():
            if isinstance(v, str) and len(v) > 100:
                log_state[k] = v[:100] + "..."
            elif isinstance(v, list) and k == "messages":
                log_state[k] = f"[{len(v)} messages]"  # Summarize message list
            elif isinstance(v, list) and k == "iteration_history":
                log_state[k] = f"[{len(v)} history entries]"  # Summarize history list
            else:
                log_state[k] = v
        return log_state


# --- Helper function to get either the real or mock graph ---
def get_graph(use_mock: bool = True):
    """
    Returns either the compiled LangGraph or a mock graph object.

    Args:
        use_mock: If True, returns the MockCompiledGraph. Otherwise,
                  attempts to build and return the actual LangGraph.

    Returns:
        Union[CompiledStateGraph, MockCompiledGraph]: The graph object.
    """
    if use_mock:
        print("Using Mock Graph for UI testing.")
        return MockCompiledGraph()
    else:
        print("Attempting to build actual LangGraph...")
        # Ensure you have LangGraph installed and configured properly
        try:
            # This is where you'd import and call the actual build function
            return build_rivalry_graph()
        except ImportError as e:
            print(
                f"LangGraph components not found or import error: {e}. Falling back to Mock Graph."
            )
            return MockCompiledGraph()
        except Exception as e:
            print(f"Error building LangGraph: {e}. Falling back to Mock Graph.")
            return MockCompiledGraph()
