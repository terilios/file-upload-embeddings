import streamlit as st
from typing import Dict, Optional, Any
from datetime import datetime

def initialize_session_state() -> None:
    """Initialize all required session state variables."""
    # Document management
    if "current_document" not in st.session_state:
        st.session_state.current_document = None
    
    # Chat management
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    # Upload management
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []
    
    # Error management
    if "errors" not in st.session_state:
        st.session_state.errors = []

def update_current_document(document: Dict) -> None:
    """
    Update the current document in session state.
    
    Args:
        document: Document data dictionary
    """
    st.session_state.current_document = document
    
    # Add to upload history
    if "upload_history" in st.session_state:
        st.session_state.upload_history.append({
            "document": document,
            "timestamp": datetime.utcnow().isoformat()
        })

def clear_current_document() -> None:
    """Clear the current document from session state."""
    st.session_state.current_document = None
    st.session_state.messages = []
    st.session_state.session_id = None

def add_message(
    role: str,
    content: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Add a message to the chat history.
    
    Args:
        role: Message role ('user' or 'assistant')
        content: Message content
        metadata: Optional message metadata
    """
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if metadata:
        message["metadata"] = metadata
    
    if "messages" in st.session_state:
        st.session_state.messages.append(message)

def clear_chat_history() -> None:
    """Clear the chat history from session state."""
    st.session_state.messages = []
    st.session_state.session_id = None

def get_error_message() -> Optional[str]:
    """
    Get and clear the latest error message.
    
    Returns:
        Error message if exists, None otherwise
    """
    if st.session_state.errors:
        return st.session_state.errors.pop()
    return None

def add_error(error: str) -> None:
    """
    Add an error message to the error stack.
    
    Args:
        error: Error message to add
    """
    st.session_state.errors.append(error)

def get_session_data() -> Dict[str, Any]:
    """
    Get all relevant session data for export or debugging.
    
    Returns:
        Dictionary containing session data
    """
    return {
        "current_document": st.session_state.get("current_document"),
        "session_id": st.session_state.get("session_id"),
        "message_count": len(st.session_state.get("messages", [])),
        "upload_history": st.session_state.get("upload_history", []),
        "timestamp": datetime.utcnow().isoformat()
    }

def restore_session(session_data: Dict[str, Any]) -> None:
    """
    Restore session state from saved data.
    
    Args:
        session_data: Dictionary containing session data
    """
    if "current_document" in session_data:
        st.session_state.current_document = session_data["current_document"]
    
    if "session_id" in session_data:
        st.session_state.session_id = session_data["session_id"]
    
    if "messages" in session_data:
        st.session_state.messages = session_data["messages"]
    
    if "upload_history" in session_data:
        st.session_state.upload_history = session_data["upload_history"]

def handle_error(error: Exception) -> None:
    """
    Handle an error by adding it to the error stack and displaying it.
    
    Args:
        error: Exception to handle
    """
    error_message = str(error)
    add_error(error_message)
    st.error(f"Error: {error_message}")

def is_session_active() -> bool:
    """
    Check if there is an active chat session.
    
    Returns:
        True if there is an active session, False otherwise
    """
    return (
        st.session_state.current_document is not None and
        st.session_state.session_id is not None
    )
