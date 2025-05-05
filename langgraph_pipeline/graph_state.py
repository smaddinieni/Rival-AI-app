# langgraph_pipeline/graph_state.py
from typing import TypedDict, Annotated, List, Union, Literal
import operator
from langchain_core.messages import (
    BaseMessage,
)  # Use actual import if using LangChain messages


# Helper reducer to append messages correctly, handling potential non-list inputs
def add_messages(left, right):
    """Append new messages to the existing list, ensuring both are lists."""
    if not isinstance(left, list):
        left = []
    if not isinstance(right, list):
        right = []
    # Simple append, LangGraph's internal mechanism might handle complex merges/updates
    return left + right


class RivalryState(TypedDict):
    """
    Represents the state of a single content generation task as it flows through the graph.
    Each key corresponds to a piece of information managed by the LangGraph pipeline.
    """

    # --- Task Identification & Input ---
    task_id: str  # Unique identifier for the task instance
    name: str  # User-defined name for the task (e.g., "Landing Page Copy")
    original_prompt: str  # The initial prompt provided by the user for OpenAI

    # --- Pipeline Configuration ---
    # Template used to generate the prompt for the refinement LLM (Gemini)
    # Example: "OpenAI's draft scored {score}/10—can you elevate emotional impact? Here's the draft:\n\n{draft}"
    feedback_prompt_template: str

    # --- Intermediate Outputs ---
    openai_output: str  # The raw text output from the first LLM (OpenAI)
    # The fully constructed prompt fed into the second LLM (Gemini)
    gemini_input_prompt: str
    gemini_output: str  # The raw text output from the second LLM (Gemini)

    # --- Final Output & State Tracking ---
    # The final version of the content, potentially edited by the user
    composite_draft: str
    # Stores the history of messages (Human, AI, Tool) for the LangGraph execution flow.
    # The 'add_messages' reducer ensures new messages are appended correctly.
    messages: Annotated[List[BaseMessage], add_messages]
    # Records metadata about each significant step (LLM call) in the pipeline
    iteration_history: List[
        dict
    ]  # List of dicts: {"timestamp", "model", "prompt", "output"}
    # Tracks the current stage of the task lifecycle
    status: Literal[
        "pending",  # Task created, not started
        "openai_done",  # OpenAI call completed
        "gemini_prompt_formatted",  # Prompt for Gemini ready
        "gemini_done",  # Gemini call completed
        "composite_done",  # Final draft saved/edited
        "error",  # An error occurred
    ]
    # Stores any error message encountered during pipeline execution
    error_message: str

    # --- Task Metadata ---
    creator: str  # Identifier for the user who created the task
    created_at: str  # ISO format timestamp of task creation
