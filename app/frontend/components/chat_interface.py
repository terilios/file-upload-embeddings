import streamlit as st
import requests
from typing import List, Dict, Optional
from datetime import datetime
import os

from config.settings import settings

def format_message(message: Dict) -> None:
    """
    Format and display a chat message.
    
    Args:
        message: Dictionary containing message data
    """
    role = message["role"]
    content = message["content"]
    
    # Apply different styling based on role
    if role == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            
            # Display sources if available
            if "metadata" in message and message["metadata"].get("sources"):
                with st.expander("View Sources"):
                    for source in message["metadata"]["sources"]:
                        st.markdown("---")
                        st.markdown(f"**Excerpt:** {source['content']}")
                        st.markdown(f"**Relevance:** {source['score']:.2f}")
                        st.markdown(f"**Document ID:** {source['document_id']}")

def send_message(
    query: str,
    document_id: Optional[int] = None,
    session_id: Optional[int] = None
) -> Optional[Dict]:
    """
    Send a message to the backend API.
    
    Args:
        query: The user's question
        document_id: Optional ID of the current document
        session_id: Optional chat session ID
    
    Returns:
        Response data if successful, None otherwise
    """
    try:
        # Get backend URL from environment variable or use default
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        
        # Prepare request data
        data = {
            "query": query,
            "document_id": document_id,
            "session_id": session_id
        }
        
        # Send request to backend
        response = requests.post(
            f"{backend_url}{settings.API_V1_STR}/chat/query",
            json=data
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None

def load_chat_history(session_id: int) -> List[Dict]:
    """
    Load chat history from the backend.
    
    Args:
        session_id: ID of the chat session
    
    Returns:
        List of message dictionaries
    """
    try:
        # Get backend URL from environment variable or use default
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        
        response = requests.get(
            f"{backend_url}{settings.API_V1_STR}/chat/sessions/{session_id}"
        )
        
        if response.status_code == 200:
            session_data = response.json()
            return session_data.get("messages", [])
        return []
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
        return []

def render_chat_interface() -> None:
    """Render the main chat interface."""
    st.header("Chat with your Document")
    
    # Initialize chat history if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    # Display chat history
    for message in st.session_state.messages:
        format_message(message)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat history
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.utcnow().isoformat()
        }
        st.session_state.messages.append(user_message)
        format_message(user_message)
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = send_message(
                query=prompt,
                document_id=st.session_state.current_document["id"],
                session_id=st.session_state.session_id
            )
            
            if response:
                # Update session ID if needed
                if not st.session_state.session_id:
                    st.session_state.session_id = response["session_id"]
                
                # Add assistant response to chat history
                assistant_message = {
                    "role": "assistant",
                    "content": response["response"],
                    "metadata": {
                        "sources": response["sources"]
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                st.session_state.messages.append(assistant_message)
                format_message(assistant_message)
    
    # Chat controls
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.experimental_rerun()
    
    with col2:
        if st.session_state.messages:
            if st.button("Export Chat"):
                # Format chat history for export
                export_data = {
                    "session_id": st.session_state.session_id,
                    "document_id": st.session_state.current_document["id"],
                    "document_name": st.session_state.current_document["filename"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "messages": st.session_state.messages
                }
                
                # Create download button
                st.download_button(
                    label="Download Chat History",
                    data=str(export_data),
                    file_name=f"chat_history_{st.session_state.session_id}.json",
                    mime="application/json"
                )
