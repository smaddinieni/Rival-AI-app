# app.py
import streamlit as st
import uuid
from datetime import datetime
import time  # For simulating delays if needed

# Import necessary components from your pipeline structure
from langgraph_pipeline.graph_state import RivalryState  # Import the state definition

# Import the function to get the compiled graph (mock or real)
from langgraph_pipeline.build_graph import get_graph

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="LangGraph Rivalry App")
st.title("✨ LangGraph LLM Rivalry ✨")
st.caption(
    "Generate content with OpenAI, then refine it with Gemini using a feedback loop."
)

# --- Initialize Session State ---
if "tasks" not in st.session_state:
    st.session_state.tasks = (
        {}
    )  # Dictionary to store task states {task_id: RivalryState}
if "selected_task_id" not in st.session_state:
    st.session_state.selected_task_id = None
if "graph" not in st.session_state:
    # Build the graph once and store it in session state
    # Set use_mock=False to attempt using the actual LangGraph build
    st.session_state.graph = get_graph(use_mock=True)


# --- Task Management Sidebar ---
st.sidebar.header("🚀 Create New Task")
new_task_name = st.sidebar.text_input(
    "Task Name (e.g., Landing Page Copy)", key="new_task_name_input"
)
base_prompt = st.sidebar.text_area(
    "Base Prompt (Input for OpenAI)", height=150, key="base_prompt_input"
)
feedback_template_default = "OpenAI's draft scored {score}/10—can you elevate emotional impact? Here's the draft:\n\n{draft}"
feedback_template = st.sidebar.text_area(
    "Feedback Prompt Template for Gemini",
    value=feedback_template_default,
    height=100,
    key="feedback_template_input",
    help="Use {score} and {draft} placeholders.",
)

if st.sidebar.button("Start New Task", key="start_task_button"):
    if new_task_name and base_prompt and feedback_template:
        task_id = str(uuid.uuid4())
        # Define the initial state for the new task
        initial_task_state: RivalryState = {
            "task_id": task_id,
            "name": new_task_name,
            "original_prompt": base_prompt,
            "feedback_prompt_template": feedback_template,
            "openai_output": "",
            "gemini_input_prompt": "",
            "gemini_output": "",
            "composite_draft": "",
            "messages": [],  # Initialize empty message list
            "iteration_history": [],
            "status": "pending",
            "error_message": "",
            "creator": "User",  # Replace with actual user if auth is implemented
            "created_at": datetime.now().isoformat(),
        }
        st.session_state.tasks[task_id] = initial_task_state
        st.session_state.selected_task_id = task_id  # Select the new task
        st.sidebar.success(f"Task '{new_task_name}' created and selected!")

        # --- Execute LangGraph Pipeline ---
        graph_to_run = st.session_state.graph  # Get graph from session state
        config = {"configurable": {"thread_id": task_id}}  # Use task_id as thread_id
        progress_placeholder = (
            st.empty()
        )  # Placeholder for status updates in the main area

        try:
            print(f"\n--- Starting Graph Execution for Task {task_id} ---")
            # Use the graph's stream method for step-by-step updates
            for step_update in graph_to_run.stream(initial_task_state, config=config):
                # The key of the dict indicates the node that produced the update
                # The value contains the state update
                node_name = list(step_update.keys())[0]
                update_data = step_update[node_name]

                # Important: Update the task state in session_state IMMEDIATELY
                if task_id in st.session_state.tasks:
                    st.session_state.tasks[task_id].update(update_data)
                    status_message = f"Executing: {node_name}..."
                    if "status" in update_data:
                        status_message += f" Status -> {update_data['status']}"
                    progress_placeholder.info(status_message)
                    print(
                        f"Update from {node_name}: { {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in update_data.items()} }"
                    )  # Log update
                    # Optional small delay to make streaming visible in mock mode
                    # time.sleep(0.1)
                else:
                    print(
                        f"Warning: Task {task_id} not found in session state during stream update."
                    )
                    break  # Stop processing if task disappears

            # Check final status
            if task_id in st.session_state.tasks:
                final_status = st.session_state.tasks[task_id].get("status", "unknown")
                if final_status != "error":
                    progress_placeholder.success(
                        f"Pipeline finished! Final status: {final_status}"
                    )
                else:
                    progress_placeholder.error(
                        f"Pipeline finished with error: {st.session_state.tasks[task_id].get('error_message')}"
                    )
            print(f"--- Graph Execution Finished for Task {task_id} ---")

        except Exception as e:
            error_msg = f"An error occurred during pipeline execution: {e}"
            st.error(error_msg)
            print(error_msg)  # Log error
            if task_id in st.session_state.tasks:
                st.session_state.tasks[task_id]["status"] = "error"
                st.session_state.tasks[task_id]["error_message"] = str(e)
            progress_placeholder.error("Pipeline execution failed.")

        # Rerun Streamlit to reflect the updated state in the UI
        st.rerun()

    else:
        st.sidebar.warning(
            "Please provide a task name, base prompt, and feedback template."
        )

# --- Task Selection Dropdown ---
st.sidebar.divider()
st.sidebar.header("View Existing Task")
if st.session_state.tasks:
    task_options = {
        tid: st.session_state.tasks[tid].get("name", f"Task {i+1}")
        for i, tid in enumerate(st.session_state.tasks.keys())
    }
    # Use the task ID as the key for the selectbox
    selected_tid = st.sidebar.selectbox(
        "Select Task",
        options=list(task_options.keys()),
        format_func=lambda tid: task_options[tid],  # Show name, store ID
        key="task_selector_dropdown",
        index=(
            list(task_options.keys()).index(st.session_state.selected_task_id)
            if st.session_state.selected_task_id in task_options
            else 0
        ),
    )
    # Update the selected task ID in session state when the dropdown changes
    if selected_tid != st.session_state.selected_task_id:
        st.session_state.selected_task_id = selected_tid
        st.rerun()  # Rerun to update the main display
else:
    st.sidebar.info("No tasks available.")

# --- Main Display Area ---
st.header("📊 Task Dashboard")

selected_task_id = st.session_state.get("selected_task_id")

if not st.session_state.tasks:
    st.info("No tasks created yet. Use the sidebar to start a new task.")
elif not selected_task_id or selected_task_id not in st.session_state.tasks:
    st.warning("Please select a valid task from the sidebar.")
    # Optionally select the first task if none is selected
    if st.session_state.tasks:
        st.session_state.selected_task_id = list(st.session_state.tasks.keys())[0]
        st.rerun()
else:
    # Display the selected task
    current_task_state = st.session_state.tasks[selected_task_id]
    task_name = current_task_state.get("name", "N/A")

    st.subheader(f"Viewing Task: {task_name}")
    st.markdown(f"**Task ID:** `{selected_task_id}`")
    st.markdown(f"**Status:** `{current_task_state.get('status', 'N/A')}`")
    if current_task_state.get("status") == "error":
        st.error(f"Error: {current_task_state.get('error_message', 'Unknown error')}")

    # --- Output Comparison Dashboard ---
    st.divider()
    st.subheader("↔️ Output Comparison")
    col1, col2 = st.columns(2)
    openai_val = current_task_state.get("openai_output", "Not generated yet.")
    gemini_val = current_task_state.get("gemini_output", "Not generated yet.")

    with col1:
        st.markdown("**🤖 OpenAI Output**")
        st.text_area(
            "OpenAI Draft",
            value=openai_val,
            height=300,
            key=f"openai_out_{selected_task_id}",
            disabled=True,
        )
        # Simple qualitative analysis (replace with actual NLP if needed)
        openai_len = len(openai_val.split())
        st.caption(f"Length: {openai_len} words")

    with col2:
        st.markdown("**♊ Gemini Output (Refined)**")
        st.text_area(
            "Gemini Draft",
            value=gemini_val,
            height=300,
            key=f"gemini_out_{selected_task_id}",
            disabled=True,
        )
        gemini_len = len(gemini_val.split())
        st.caption(f"Length: {gemini_len} words")

    # --- Composite Draft Builder ---
    st.divider()
    st.subheader("🛠️ Composite Draft Builder")
    # Initialize composite draft if empty, prioritizing Gemini > OpenAI > Original
    initial_composite = current_task_state.get(
        "composite_draft",
        current_task_state.get(
            "gemini_output",
            current_task_state.get(
                "openai_output", current_task_state.get("original_prompt", "")
            ),
        ),
    )

    edited_composite = st.text_area(
        "Edit and Refine Final Draft (Markdown Supported)",
        value=initial_composite,
        height=400,
        key=f"composite_edit_{selected_task_id}",  # Unique key per task
    )
    if st.button("Save Composite Draft", key=f"save_comp_{selected_task_id}"):
        st.session_state.tasks[selected_task_id]["composite_draft"] = edited_composite
        st.session_state.tasks[selected_task_id][
            "status"
        ] = "composite_done"  # Update status
        st.success("Composite draft saved!")
        # Short delay before rerun allows message to be seen
        time.sleep(1)
        st.rerun()

    # --- Export Options ---
    final_draft = current_task_state.get("composite_draft")
    if current_task_state.get("status") == "composite_done" and final_draft:
        st.download_button(
            label="Export as Markdown (.md)",
            data=final_draft,
            file_name=f"{task_name.replace(' ', '_')}_final_draft.md",
            mime="text/markdown",
            key=f"export_md_{selected_task_id}",
        )
        # PDF export requires extra libraries (e.g., reportlab, fpdf)
        # st.caption("PDF export requires additional setup.")

    # --- Versioning & Audit Trail ---
    st.divider()
    with st.expander("📜 Pipeline Execution History (Audit Trail)"):
        history = current_task_state.get("iteration_history", [])
        if history:
            # Display history nicely - convert to DataFrame for better view
            import pandas as pd

            try:
                df = pd.DataFrame(history)
                # Optionally reorder columns or format timestamps
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                st.dataframe(
                    df[["timestamp", "model", "prompt", "output"]],
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Could not display history as DataFrame: {e}")
                st.json(history)  # Fallback to JSON
        else:
            st.caption("No execution history recorded yet.")
        # Rollback functionality is complex and not implemented here.
        # It would involve resetting session_state to a previous checkpoint state.

    # --- Raw Message History (Debug) ---
    with st.expander("💬 Raw LangGraph Message History (Debug)"):
        messages = current_task_state.get("messages", [])
        if messages:
            for i, msg_obj in enumerate(messages):
                # Check if it's the BaseMessage placeholder or actual LangChain message
                if hasattr(msg_obj, "role") and hasattr(msg_obj, "content"):
                    role = msg_obj.role
                    content = msg_obj.content
                    tool_calls = getattr(msg_obj, "tool_calls", None)
                    st.markdown(f"**{i+1}. {role.capitalize()}**:")
                    st.markdown(content)
                    if tool_calls:
                        st.json(tool_calls)
                    st.caption(f"ID: {getattr(msg_obj, 'id', 'N/A')}")
                    st.divider()
                else:
                    # Fallback for unexpected message format
                    st.write(f"{i+1}. {str(msg_obj)}")
        else:
            st.caption("No message history recorded yet.")
