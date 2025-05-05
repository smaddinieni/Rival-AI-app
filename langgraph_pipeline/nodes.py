# langgraph_pipeline/nodes.py
from .graph_state import RivalryState
from .llm_clients import get_llm_client

# Ensure you have these imports if using actual LangChain messages
# If using dicts, you might not need them directly in this file
# but they are defined in the state.
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from datetime import datetime
import streamlit as st  # Use streamlit for error reporting in nodes
import traceback  # For more detailed error logging

# --- LangGraph Nodes ---


def call_openai_node(state: RivalryState) -> dict:
    """
    Calls the OpenAI LLM with the original prompt and updates the state.
    """
    # Avoid re-running if this step or a later step was already completed or errored
    if state.get("status") != "pending":
        print(f"Skipping OpenAI node: Status is '{state.get('status')}'.")
        return {}

    print(">>> Executing OpenAI Node...")
    openai_llm = get_llm_client("openai")
    if not openai_llm:
        error_msg = "OpenAI client failed to initialize."
        st.error(error_msg)
        # Update state to reflect the error
        return {
            "status": "error",
            "error_message": error_msg,
            "messages": state.get("messages", []),
        }

    prompt = state.get("original_prompt")
    if not prompt:
        error_msg = "Original prompt is missing in state for OpenAI node."
        st.error(error_msg)
        return {
            "status": "error",
            "error_message": error_msg,
            "messages": state.get("messages", []),
        }

    # Start message history with the user's original prompt
    # Ensure messages list exists and append the new HumanMessage
    current_messages = state.get("messages", [])
    messages_for_openai = current_messages + [HumanMessage(content=prompt)]

    try:
        # Invoke the LLM with the prepared messages list
        ai_response = openai_llm.invoke(messages_for_openai)
        timestamp = datetime.now().isoformat()
        # Record this iteration
        iteration = {
            "timestamp": timestamp,
            "model": "openai",
            "prompt": prompt,  # Log the specific prompt used
            "output": ai_response.content,  # Access content attribute
        }
        print(f"OpenAI Output received: {ai_response.content[:100]}...")

        # Prepare the state update
        # Append the AI response to the message history
        updated_messages = messages_for_openai + [ai_response]
        updated_history = state.get("iteration_history", []) + [iteration]

        return {
            "openai_output": ai_response.content,
            "messages": updated_messages,
            "iteration_history": updated_history,
            "status": "openai_done",  # Update status to indicate completion
        }
    except Exception as e:
        error_msg = f"Error calling OpenAI: {e}\n{traceback.format_exc()}"
        print(error_msg)
        st.error(error_msg)
        # Update state with error status and message
        return {
            "status": "error",
            "error_message": error_msg,
            "messages": messages_for_openai,
        }


def format_gemini_prompt_node(state: RivalryState) -> dict:
    """
    Formats the prompt for Gemini using the feedback template and OpenAI's output.
    Updates the state with the formatted prompt.
    """
    # Only run if OpenAI has finished and the prompt isn't already formatted
    if state.get("status") != "openai_done":
        print(f"Skipping Gemini prompt formatting: Status is '{state.get('status')}'.")
        return {}
    # Avoid re-running if already done
    if state.get("gemini_input_prompt"):
        print("Skipping Gemini prompt formatting: Already formatted.")
        return {"status": "gemini_prompt_formatted"}  # Ensure status is correct

    print(">>> Formatting Gemini Prompt Node...")
    openai_draft = state.get("openai_output")
    feedback_template = state.get("feedback_prompt_template")

    if not openai_draft:
        error_msg = "Cannot format Gemini prompt: OpenAI output is missing from state."
        print(error_msg)
        st.error(error_msg)
        return {"status": "error", "error_message": error_msg}
    if not feedback_template:
        error_msg = (
            "Cannot format Gemini prompt: Feedback template is missing from state."
        )
        print(error_msg)
        st.error(error_msg)
        return {"status": "error", "error_message": error_msg}

    # Basic feedback mechanism - replace with actual scoring if needed
    # For simplicity, let's assume a fixed score or pass it in state if dynamic
    score = 7  # Example score
    try:
        # Check if placeholders exist before formatting
        placeholders = ["{score}", "{draft}"]
        missing_placeholders = [p for p in placeholders if p not in feedback_template]
        if missing_placeholders:
            st.warning(
                f"Feedback template might be missing placeholders: {', '.join(missing_placeholders)}. Proceeding with formatting."
            )

        # Perform the formatting
        gemini_prompt = feedback_template.format(score=score, draft=openai_draft)
        print(f"Formatted Gemini Prompt: {gemini_prompt[:100]}...")
        # Return the update for the state
        return {
            "gemini_input_prompt": gemini_prompt,
            "status": "gemini_prompt_formatted",  # Update status
        }
    except KeyError as e:
        # More specific error if format keys are wrong
        error_msg = f"Error formatting Gemini prompt: Missing key {e} in template. Template was: '{feedback_template}'"
        print(error_msg)
        st.error(error_msg)
        return {"status": "error", "error_message": error_msg}
    except Exception as e:
        # Catch other potential formatting errors
        error_msg = (
            f"Unexpected error formatting Gemini prompt: {e}\n{traceback.format_exc()}"
        )
        print(error_msg)
        st.error(error_msg)
        return {"status": "error", "error_message": error_msg}


def call_gemini_node(state: RivalryState) -> dict:
    """
    Calls the Gemini LLM with the formatted prompt and updates the state.
    """
    # Only run if the prompt is ready and Gemini hasn't run or errored
    if state.get("status") != "gemini_prompt_formatted":
        print(f"Skipping Gemini node: Status is '{state.get('status')}'.")
        return {}

    print(">>> Executing Gemini Node...")
    gemini_llm = get_llm_client("gemini")
    if not gemini_llm:
        error_msg = "Gemini client failed to initialize."
        st.error(error_msg)
        return {"status": "error", "error_message": error_msg}

    gemini_prompt = state.get("gemini_input_prompt")
    if not gemini_prompt:
        error_msg = "Cannot call Gemini: Formatted prompt is missing from state."
        print(error_msg)
        st.error(error_msg)
        return {"status": "error", "error_message": error_msg}

    # Prepare messages for Gemini: include previous history and the new formatted prompt
    previous_messages = state.get("messages", [])
    # Ensure it's a list, defensively
    if not isinstance(previous_messages, list):
        print(
            f"Warning: 'messages' in state was not a list ({type(previous_messages)}). Resetting."
        )
        previous_messages = []

    # Add the formatted prompt as the latest HumanMessage
    messages_for_gemini = previous_messages + [HumanMessage(content=gemini_prompt)]

    try:
        # Invoke the Gemini LLM
        ai_response = gemini_llm.invoke(messages_for_gemini)
        timestamp = datetime.now().isoformat()
        # Record this iteration
        iteration = {
            "timestamp": timestamp,
            "model": "gemini",
            "prompt": gemini_prompt,  # Log the specific prompt used
            "output": ai_response.content,  # Access content attribute
        }
        print(f"Gemini Output received: {ai_response.content[:100]}...")

        # Append Gemini's response to the main message history
        updated_messages = messages_for_gemini + [ai_response]
        updated_history = state.get("iteration_history", []) + [iteration]

        # Return the state update
        return {
            "gemini_output": ai_response.content,
            "messages": updated_messages,  # Return the full updated list
            "iteration_history": updated_history,
            "status": "gemini_done",  # Update status
        }
    except Exception as e:
        error_msg = f"Error calling Gemini: {e}\n{traceback.format_exc()}"
        print(error_msg)
        st.error(error_msg)
        # Update state with error status and message
        return {
            "status": "error",
            "error_message": error_msg,
            "messages": messages_for_gemini,
        }
