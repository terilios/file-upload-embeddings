import streamlit as st
from typing import Dict, Optional
import requests
from datetime import datetime

from config.settings import settings

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def format_timestamp(timestamp: str) -> str:
    """
    Format timestamp in a readable format.
    
    Args:
        timestamp: ISO format timestamp
    
    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp

def get_document_stats(document_id: int) -> Optional[Dict]:
    """
    Get detailed statistics for a document.
    
    Args:
        document_id: ID of the document
    
    Returns:
        Dictionary containing document statistics
    """
    try:
        response = requests.get(
            f"http://localhost:8000{settings.API_V1_STR}/documents/{document_id}"
        )
        
        if response.status_code == 200:
            return response.json()
        return None
        
    except Exception as e:
        st.error(f"Error fetching document stats: {str(e)}")
        return None

def render_metadata_display(document: Dict) -> None:
    """
    Render document metadata and statistics.
    
    Args:
        document: Dictionary containing document data
    """
    st.subheader("Document Information")
    
    # Basic information
    with st.expander("📄 Basic Information", expanded=True):
        st.markdown(f"**Filename:** {document['filename']}")
        if 'file_size' in document:
            st.markdown(f"**Size:** {format_file_size(document['file_size'])}")
        if 'content_type' in document:
            st.markdown(f"**Type:** {document['content_type']}")
        if 'created_at' in document:
            st.markdown(f"**Uploaded:** {format_timestamp(document['created_at'])}")
    
    # Get detailed statistics
    stats = get_document_stats(document['id'])
    
    if stats:
        # Processing statistics
        with st.expander("📊 Processing Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Total Chunks",
                    value=stats.get('chunk_count', 0)
                )
            
            with col2:
                if 'metadata' in stats and 'token_count' in stats['metadata']:
                    st.metric(
                        label="Total Tokens",
                        value=stats['metadata']['token_count']
                    )
        
        # Document metadata
        if 'metadata' in stats and stats['metadata']:
            with st.expander("🔍 Document Metadata"):
                metadata = stats['metadata']
                
                # File information
                if 'file_extension' in metadata:
                    st.markdown(f"**Extension:** {metadata['file_extension']}")
                if 'processing_timestamp' in metadata:
                    st.markdown(
                        f"**Processed:** {format_timestamp(metadata['processing_timestamp'])}"
                    )
                
                # Custom metadata
                custom_metadata = {
                    k: v for k, v in metadata.items() 
                    if k not in ['file_extension', 'processing_timestamp']
                }
                
                if custom_metadata:
                    st.markdown("**Additional Metadata:**")
                    for key, value in custom_metadata.items():
                        st.markdown(f"- **{key}:** {value}")
        
        # Document preview
        if 'preview' in stats:
            with st.expander("👁️ Document Preview"):
                st.markdown("**First 500 characters:**")
                st.markdown(stats['preview'][:500] + "...")
    
    # Document actions
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Refresh Metadata"):
            st.experimental_rerun()
    
    with col2:
        if st.button("Delete Document"):
            if delete_document(document['id']):
                st.success("Document deleted successfully!")
                st.session_state.current_document = None
                st.experimental_rerun()

def delete_document(document_id: int) -> bool:
    """
    Delete a document.
    
    Args:
        document_id: ID of the document to delete
    
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        response = requests.delete(
            f"http://localhost:8000{settings.API_V1_STR}/documents/{document_id}"
        )
        return response.status_code == 200
    except:
        return False
