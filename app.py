# app.py
import streamlit as st
import uuid
from datetime import datetime
import time  # For simulating delays if needed
import traceback  # For detailed error logging

# Import necessary components from your pipeline structure
from langgraph_pipeline.graph_state import RivalryState  # Import the state definition

# Import the function to get the compiled graph (mock or real)
from langgraph_pipeline.build_graph import get_graph

# Import the display component
from components.output_display import display_outputs

# Import utility functions
from langgraph_pipeline.utils import format_timestamp, sanitize_filename

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
    # Removed use_mock=True as the function no longer accepts it
    try:
        st.session_state.graph = get_graph()
    except Exception as e:
        st.error(
            f"Fatal Error: Could not build LangGraph. Please check pipeline code and dependencies. Error: {e}"
        )
        st.stop()  # Stop the app if the graph can't be built


# --- Task Management Sidebar ---
st.sidebar.header("🚀 Create New Task")
with st.sidebar.form(key="new_task_form"):
    new_task_name = st.text_input(
        "Task Name (e.g., Landing Page Copy)", key="new_task_name_input"
    )
    base_prompt = st.text_area(
        "Base Prompt (Input for OpenAI)", height=150, key="base_prompt_input"
    )
    feedback_template_default = "OpenAI's draft scored {score}/10—can you elevate emotional impact? Here's the draft:\n\n{draft}"
    feedback_template = st.text_area(
        "Feedback Prompt Template for Gemini",
        value=feedback_template_default,
        height=100,
        key="feedback_template_input",
        help="Use {score} and {draft} placeholders.",
    )
    submit_button = st.form_submit_button(label="Start New Task")

    if submit_button:
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

            # --- Trigger LangGraph Pipeline Execution ---
            graph_to_run = st.session_state.graph  # Get graph from session state
            config = {
                "configurable": {"thread_id": task_id}
            }  # Use task_id as thread_id
            progress_placeholder = (
                st.empty()
            )  # Placeholder for status updates in the main area

            try:
                print(f"\n--- Starting Graph Execution for Task {task_id} ---")
                # Use the graph's stream method for step-by-step updates
                for step_update in graph_to_run.stream(
                    initial_task_state, config=config
                ):
                    # The key of the dict indicates the node that produced the update
                    # The value contains the state update
                    node_name = list(step_update.keys())[0]
                    update_data = step_update[node_name]

                    # Important: Update the task state in session_state IMMEDIATELY
                    if task_id in st.session_state.tasks:
                        # Ensure messages are handled correctly (appended)
                        if "messages" in update_data:
                            # Use the reducer logic (add_messages) implicitly via LangGraph state update
                            # For direct session_state update, manual append might be needed if not using LangGraph's state directly
                            # This assumes LangGraph handles the append via checkpointer/state mechanism
                            pass  # LangGraph's compile/invoke handles the message append via checkpointer

                        # Update other fields
                        st.session_state.tasks[task_id].update(
                            {k: v for k, v in update_data.items()}
                        )

                        status_message = f"Executing: {node_name}..."
                        if "status" in update_data:
                            status_message += f" Status -> {update_data['status']}"
                        progress_placeholder.info(status_message)
                        print(
                            f"Update from {node_name}: { {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in update_data.items()} }"
                        )  # Log update
                    else:
                        print(
                            f"Warning: Task {task_id} not found in session state during stream update."
                        )
                        st.warning(f"Task {task_id} disappeared during execution.")
                        break  # Stop processing if task disappears

                # Check final status after stream completes
                if task_id in st.session_state.tasks:
                    # Fetch the potentially updated state after the stream finishes
                    # final_state_snapshot = graph_to_run.get_state(config) # Get final state from checkpointer
                    # st.session_state.tasks[task_id] = final_state_snapshot.values # Update with definitive final state
                    final_status = st.session_state.tasks[task_id].get(
                        "status", "unknown"
                    )
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
                error_msg = f"An error occurred during pipeline execution: {e}\n{traceback.format_exc()}"
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
    # Sort tasks by creation time, newest first
    sorted_task_ids = sorted(
        st.session_state.tasks.keys(),
        key=lambda tid: st.session_state.tasks[tid].get("created_at", ""),
        reverse=True,
    )
    task_options = {
        tid: f"{st.session_state.tasks[tid].get('name', f'Task {i+1}')} ({format_timestamp(st.session_state.tasks[tid].get('created_at', ''))})"
        for i, tid in enumerate(sorted_task_ids)
    }

    # Determine the default index for the selectbox
    current_selection = st.session_state.get("selected_task_id")
    try:
        # Find the index of the currently selected task ID within the sorted list
        default_index = (
            sorted_task_ids.index(current_selection)
            if current_selection in sorted_task_ids
            else 0
        )
    except ValueError:
        default_index = (
            0  # Default to the first item if the selected ID is somehow invalid
        )

    selected_tid = st.sidebar.selectbox(
        "Select Task",
        options=sorted_task_ids,  # Use sorted IDs as options
        format_func=lambda tid: task_options.get(
            tid, "Unknown Task"
        ),  # Show name and date
        key="task_selector_dropdown",
        index=default_index,  # Set the default index
    )
    # Update the selected task ID in session state only if it changes
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
    # Optionally select the first task if none is selected and tasks exist
    if st.session_state.tasks:
        first_task_id = list(st.session_state.tasks.keys())[0]
        if st.session_state.selected_task_id != first_task_id:
            st.session_state.selected_task_id = first_task_id
            st.rerun()
else:
    # Display the selected task
    current_task_state = st.session_state.tasks[selected_task_id]
    task_name = current_task_state.get("name", "N/A")

    st.subheader(f"Viewing Task: {task_name}")
    # Display metadata
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        st.markdown(f"**Task ID:**")
        st.code(selected_task_id, language=None)
    with meta_col2:
        st.markdown(f"**Status:** `{current_task_state.get('status', 'N/A')}`")
    with meta_col3:
        st.markdown(
            f"**Created:** {format_timestamp(current_task_state.get('created_at', ''))}"
        )

    if current_task_state.get("status") == "error":
        st.error(
            f"Error Details: {current_task_state.get('error_message', 'Unknown error')}"
        )

    # --- Display Prompts ---
    with st.expander("View Prompts", expanded=False):
        st.markdown("**Original Prompt (for OpenAI):**")
        st.text(current_task_state.get("original_prompt", "N/A"))
        st.markdown("**Feedback Template (for Gemini):**")
        st.text(current_task_state.get("feedback_prompt_template", "N/A"))
        st.markdown("**Actual Prompt (for Gemini):**")
        st.text(current_task_state.get("gemini_input_prompt", "Not generated yet."))

    # --- Output Comparison Dashboard (Using Component) ---
    display_outputs(current_task_state)

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
        # Update the specific task's state
        st.session_state.tasks[selected_task_id]["composite_draft"] = edited_composite
        st.session_state.tasks[selected_task_id][
            "status"
        ] = "composite_done"  # Update status
        st.success("Composite draft saved!")
        # Short delay before rerun allows message to be seen
        time.sleep(0.5)
        st.rerun()

    # --- Export Options ---
    final_draft = current_task_state.get("composite_draft")
    # Show export button only if the composite draft has been explicitly saved (status is composite_done)
    if current_task_state.get("status") == "composite_done" and final_draft:
        st.download_button(
            label="Export as Markdown (.md)",
            data=final_draft,
            # Sanitize filename
            file_name=f"{sanitize_filename(task_name, 'draft')}_final.md",
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
            try:
                import pandas as pd

                df = pd.DataFrame(history)
                # Format timestamp for display
                if "timestamp" in df.columns:
                    df["timestamp_fmt"] = pd.to_datetime(df["timestamp"]).dt.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    # Optionally reorder columns
                    display_cols = ["timestamp_fmt", "model", "prompt", "output"]
                    df_display = df[[col for col in display_cols if col in df.columns]]
                else:
                    df_display = df
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            except ImportError:
                st.warning(
                    "Pandas not installed (`pip install pandas`). Displaying raw history."
                )
                st.json(history)  # Fallback to JSON
            except Exception as e:
                st.error(f"Could not display history as DataFrame: {e}")
                st.json(history)  # Fallback to JSON
        else:
            st.caption("No execution history recorded yet for this task.")
        # Rollback functionality is complex and not implemented here.
        # It would involve getting a previous checkpoint state and invoking the graph from there.

    # --- Raw Message History (Debug) ---
    with st.expander("💬 Raw LangGraph Message History (Debug)"):
        messages = current_task_state.get("messages", [])
        if messages:
            for i, msg_obj in enumerate(messages):
                try:
                    # Check if it's a BaseMessage object or a dict representation
                    if hasattr(msg_obj, "role") and hasattr(msg_obj, "content"):
                        role = msg_obj.role
                        content = msg_obj.content
                        tool_calls = getattr(msg_obj, "tool_calls", None)
                        msg_id = getattr(msg_obj, "id", "N/A")
                    elif isinstance(msg_obj, dict):
                        role = msg_obj.get("role", "unknown")
                        content = msg_obj.get("content", "")
                        tool_calls = msg_obj.get("tool_calls")
                        msg_id = msg_obj.get("id", "N/A")
                    else:
                        role = "unknown"
                        content = str(msg_obj)
                        tool_calls = None
                        msg_id = "N/A"

                    st.markdown(f"**{i+1}. {role.capitalize()}** (ID: `{msg_id}`):")
                    # Use st.code for potentially long/formatted content if needed, else st.markdown
                    if isinstance(content, str) and len(content) > 200:
                        st.text(
                            content
                        )  # Use text for very long content to avoid markdown issues
                    else:
                        st.markdown(
                            str(content) if content is not None else "*No Content*"
                        )

                    if tool_calls:
                        st.markdown("**Tool Calls:**")
                        st.json(tool_calls)
                    st.divider()
                except Exception as e:
                    st.error(f"Error displaying message {i+1}: {e}")
                    st.write(msg_obj)  # Show raw object on error
        else:
            st.caption("No message history recorded yet for this task.")
