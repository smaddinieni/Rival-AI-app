# components/output_display.py
import streamlit as st
from typing import Dict, Any  # Use Dict for type hinting the state


def display_outputs(task_state: Dict[str, Any]):
    """
    Renders the OpenAI and Gemini outputs side-by-side in Streamlit columns.

    Args:
        task_state: The dictionary representing the current state of the selected task.
                    Expected keys: 'openai_output', 'gemini_output'.
    """
    st.divider()
    st.subheader("↔️ Output Comparison")
    col1, col2 = st.columns(2)

    openai_val = task_state.get("openai_output", "Not generated yet.")
    gemini_val = task_state.get("gemini_output", "Not generated yet.")
    task_id = task_state.get("task_id", "no_id")  # Get task_id for unique keys

    with col1:
        st.markdown("**🤖 OpenAI Output**")
        st.text_area(
            "OpenAI Draft",
            value=openai_val,
            height=300,
            key=f"openai_out_display_{task_id}",  # Unique key using task_id
            disabled=True,
        )
        # --- Placeholder for Qualitative Analysis ---
        # In a real app, you might run NLP analysis here (e.g., using spaCy or NLTK)
        # or call another LLM to analyze tone, sentiment, etc.
        openai_len = len(openai_val.split())
        st.caption(
            f"Length: {openai_len} words | Tone: Neutral (Mock) | Sentiment: Positive (Mock)"
        )

    with col2:
        st.markdown("**♊ Gemini Output (Refined)**")
        st.text_area(
            "Gemini Draft",
            value=gemini_val,
            height=300,
            key=f"gemini_out_display_{task_id}",  # Unique key using task_id
            disabled=True,
        )
        # --- Placeholder for Qualitative Analysis ---
        gemini_len = len(gemini_val.split())
        st.caption(
            f"Length: {gemini_len} words | Tone: Empathetic (Mock) | Sentiment: Very Positive (Mock)"
        )


# Example of how to use this in app.py:
# from components.output_display import display_outputs
#
# # Inside the main display area in app.py, after getting current_task_state:
# if selected_task_id and selected_task_id in st.session_state.tasks:
#     current_task_state = st.session_state.tasks[selected_task_id]
#     # ... other UI elements ...
#     display_outputs(current_task_state) # Call the display function
#     # ... rest of the UI ...
