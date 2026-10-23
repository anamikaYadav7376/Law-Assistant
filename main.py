"""
DesignLaw AI — Clean Chat Interface.

A clean, document-grounded legal assistant for Indian industrial design registration
based strictly on the official CGPDTM Manual.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from rag_chain import (
    check_database_ready,
    answer_legal_question,
)

# Load environment variables from .env
load_dotenv()

# Page configuration - No sidebar
st.set_page_config(
    page_title="Law Assistant",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Clean, professional styling - Completely removes sidebar and provides high-contrast header
st.markdown("""
<style>
    /* Completely hide Streamlit sidebar and its toggle button */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Clean layout and typography */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 5rem;
        max-width: 800px;
    }
    
    /* High-contrast header card so white/light title never blends with background */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff !important;
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        font-size: 1rem;
        font-weight: 400;
        color: #cbd5e1 !important;
    }

    /* Modern, sleek Chat Bubbles - replaces bright default colors */
    [data-testid="stChatMessage"] {
        border-radius: 10px !important;
        padding: 0.95rem 1.15rem !important;
        margin-bottom: 0.85rem !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        background-color: rgba(30, 41, 59, 0.35) !important;
    }

    /* Replace the default red & yellow avatar background boxes */
    [data-testid="stChatMessage"] div[data-testid="chatAvatarIcon-user"],
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #2563eb !important; /* Premium Cobalt/Slate Blue instead of red */
        color: #ffffff !important;
        border-radius: 8px !important;
    }

    [data-testid="stChatMessage"] div[data-testid="chatAvatarIcon-assistant"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #0f766e !important; /* Elegant Legal Teal/Emerald instead of orange/yellow */
        color: #ffffff !important;
        border-radius: 8px !important;
    }

    /* Custom avatar emoji badge styling */
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarCustom"],
    [data-testid="stChatMessage"] div[data-testid="stChatMessageAvatar"] {
        background: rgba(51, 65, 85, 0.5) !important;
        border: 1px solid rgba(100, 116, 139, 0.4) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Define custom professional icons to avoid Streamlit's default red/yellow icons
USER_AVATAR = "👤"
AI_AVATAR = "⚖️"


def main():
    # Header: Clean Title & Subtitle in white/light color without restriction guidance
    st.markdown("""
    <div class="header-card">
        <div class="header-title">Law Assistant</div>
        <div class="header-subtitle">Ask questions about industrial design registration in India</div>
    </div>
    """, unsafe_allow_html=True)

    # Server-side API key check - never expose or request key in UI
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key.strip() in ("", "your_gemini_api_key_here"):
        st.error("Gemini API configuration is missing. Please configure the server environment.")
        return

    # Server-side database readiness check
    db_ready, db_msg = check_database_ready()
    if not db_ready:
        st.error(f"Vector database is not ready: {db_msg}")
        return

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display full chat history with custom avatars
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else AI_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Dynamic Chat input at the bottom
    user_query = st.chat_input("Ask any question about industrial design registration...")
    if user_query:
        clean_query = user_query.strip()
        if clean_query:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": clean_query})
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(clean_query)

            # Retrieve and generate answer with loading indicator
            with st.chat_message("assistant", avatar=AI_AVATAR):
                with st.spinner("Searching official manual and verifying evidence..."):
                    try:
                        result = answer_legal_question(clean_query)
                        response_text = result["full_text"]
                    except Exception as err:
                        response_text = f"An error occurred: {str(err)}"

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
