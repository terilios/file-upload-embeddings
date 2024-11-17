import streamlit as st
import requests
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional, Dict, List

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from config.settings import settings
from app.frontend.components.file_upload import render_file_upload
from app.frontend.components.chat_interface import render_chat_interface
from app.frontend.components.metadata_display import render_metadata_display
from app.frontend.utils.state_management import initialize_session_state

def setup_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title=settings.PROJECT_NAME,
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
        <style>
        .main {
            max-width: 1200px;
            padding: 2rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
        .upload-section {
            padding: 2rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
            margin-bottom: 2rem;
        }
        .metadata-section {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 2rem;
        }
        .assistant-message {
            background-color: #f5f5f5;
            margin-right: 2rem;
        }
        .source-section {
            font-size: 0.9rem;
            color: #666;
            border-left: 3px solid #ccc;
            padding-left: 1rem;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the application header and description."""
    st.title("Document Q&A System")
    st.markdown("""
        Upload documents and ask questions about their content. 
        The system will provide answers based on the document context.
    """)

def main():
    """Main application entry point."""
    # Setup page configuration
    setup_page()
    
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Sidebar with document upload
    with st.sidebar:
        st.header("Document Management")
        uploaded_doc = render_file_upload()
        
        if uploaded_doc:
            st.session_state.current_document = uploaded_doc
            render_metadata_display(uploaded_doc)
    
    # Main chat interface
    if st.session_state.current_document:
        render_chat_interface()
    else:
        st.info("👈 Start by uploading a document in the sidebar")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with Streamlit • "
        f"API Status: {'🟢 Online' if check_api_health() else '🔴 Offline'}"
    )

def check_api_health() -> bool:
    """Check if the backend API is healthy."""
    try:
        # Always use the container name in Docker
        backend_url = os.getenv("BACKEND_URL", "http://backend:8000")
        
        # Add timeout and headers
        response = requests.get(
            f"{backend_url}/health",
            timeout=5,
            headers={"Accept": "application/json"}
        )
        
        if response.status_code == 200:
            health_data = response.json()
            return health_data.get("status") == "healthy"
        return False
        
    except Exception as e:
        print(f"API health check failed: {e}")
        return False

if __name__ == "__main__":
    main()
