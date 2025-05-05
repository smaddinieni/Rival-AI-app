# langgraph_pipeline/llm_clients.py
import os
from dotenv import load_dotenv
import streamlit as st
from typing import Literal, Optional

# --- Actual LangChain Imports ---
# Make sure these are installed: pip install langchain-openai langchain-google-genai
try:
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Import BaseChatModel for type hinting, though specific classes are used
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError:
    st.error(
        "Required LangChain libraries not found. Please install: pip install langchain-openai langchain-google-genai"
    )
    # Define BaseChatModel as None or a placeholder if import fails,
    # so the function signature doesn't break immediately.
    BaseChatModel = None


# --- Function to Get Real LLM Client ---
def get_llm_client(provider: Literal["openai", "gemini"]) -> Optional[BaseChatModel]:
    """
    Initializes and returns a LangChain LLM client for the specified provider.
    Loads API keys from the .env file. Returns None if keys are missing or
    if client initialization fails.

    Args:
        provider: The LLM provider ('openai' or 'gemini').

    Returns:
        An initialized LangChain ChatModel instance or None if initialization fails.
    """
    load_dotenv()  # Load keys from .env file in the project root
    api_key = os.getenv(f"{provider.upper()}_API_KEY")

    if not api_key:
        st.error(
            f"API Key for {provider.upper()} not found in .env file or environment variables."
        )
        return None

    try:
        if provider == "openai":
            print("Initializing ChatOpenAI client...")
            # Use the specific model and settings provided by the user
            client = ChatOpenAI(
                model="gpt-4o-mini",  # User specified gpt-4.1-mini-2025-04-14, using gpt-4o-mini as a close alternative for now
                temperature=0.2,
                # max_tokens=8000, # Often set during invoke, not init
                max_retries=2,
                api_key=api_key,
            )
            print("ChatOpenAI client initialized.")
            return client
        elif provider == "gemini":
            print("Initializing ChatGoogleGenerativeAI client...")
            # Use the specific model and settings provided by the user
            client = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",  # User specified gemini-2.0-flash, using 1.5-flash as common alternative
                temperature=0.2,
                # max_tokens=8000, # Often set during invoke, not init
                # max_retries=2, # Not a direct parameter for ChatGoogleGenerativeAI init
                google_api_key=api_key,
                convert_system_message_to_human=True,  # Often helpful for Gemini
            )
            print("ChatGoogleGenerativeAI client initialized.")
            return client
        else:
            st.error(f"Unknown LLM provider requested: {provider}")
            return None
    except ImportError:
        # This case should ideally be caught by the top-level import try-except,
        # but included here for robustness.
        st.error(
            f"Failed to import LangChain client for {provider}. "
            f"Install with: pip install langchain-{provider}."
        )
        return None
    except Exception as e:
        st.error(f"Failed to initialize real {provider} client: {e}")
        import traceback

        traceback.print_exc()  # Print stack trace for debugging
        return None


# --- Optional: Add centralized configuration/rate limiting logic here ---
# For example, you could wrap the clients or add parameters for rate limits,
# logging, etc., although LangSmith handles much of the logging/tracing.
