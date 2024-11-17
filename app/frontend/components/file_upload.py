import streamlit as st
import requests
from typing import Optional, Dict
import time
from pathlib import Path

from config.settings import settings

def validate_file(file) -> tuple[bool, str]:
    """
    Validate the uploaded file.
    
    Args:
        file: The uploaded file object
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file:
        return False, "No file uploaded"
    
    # Check file extension
    file_extension = Path(file.name).suffix.lower().replace('.', '')
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
    
    # Check file size
    if file.size > settings.MAX_CONTENT_LENGTH:
        max_size_mb = settings.MAX_CONTENT_LENGTH / (1024 * 1024)
        return False, f"File too large. Maximum size: {max_size_mb}MB"
    
    return True, ""

def upload_file(file) -> Optional[Dict]:
    """
    Upload file to backend API.
    
    Args:
        file: The file to upload
    
    Returns:
        Document metadata if successful, None otherwise
    """
    try:
        # Create upload form data
        files = {"file": file}
        
        # Upload file to backend
        with st.spinner("Processing document..."):
            response = requests.post(
                f"http://localhost:8000{settings.API_V1_STR}/documents/upload",
                files=files
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def render_file_upload() -> Optional[Dict]:
    """
    Render the file upload interface.
    
    Returns:
        Document metadata if upload successful, None otherwise
    """
    st.subheader("Upload Document")
    
    # File upload widget
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=list(settings.ALLOWED_EXTENSIONS),
        help=f"Supported formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
    )
    
    if uploaded_file:
        # Validate file
        is_valid, error_message = validate_file(uploaded_file)
        
        if not is_valid:
            st.error(error_message)
            return None
        
        # Display file info
        st.write("File Information:")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {uploaded_file.name}")
        with col2:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.write(f"**Size:** {size_mb:.2f}MB")
        
        # Upload button
        if st.button("Process Document"):
            # Show progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress while uploading
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
                status_text.text(f"Processing: {i + 1}%")
            
            # Upload file
            result = upload_file(uploaded_file)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if result:
                st.success("Document processed successfully!")
                
                # Display document info
                st.write("Document Details:")
                st.json(result)
                
                return result
    
    return None

def render_document_list():
    """Render the list of uploaded documents."""
    try:
        response = requests.get(
            f"http://localhost:8000{settings.API_V1_STR}/documents/list"
        )
        
        if response.status_code == 200:
            documents = response.json()
            
            if documents:
                st.subheader("Uploaded Documents")
                
                for doc in documents:
                    with st.expander(f"📄 {doc['filename']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**ID:** {doc['id']}")
                            st.write(f"**Type:** {doc['content_type']}")
                        
                        with col2:
                            st.write(f"**Uploaded:** {doc['created_at']}")
                            if doc.get('metadata'):
                                st.write("**Has Metadata:** Yes")
                        
                        # Delete button
                        if st.button("Delete", key=f"delete_{doc['id']}"):
                            if delete_document(doc['id']):
                                st.success("Document deleted!")
                                st.experimental_rerun()
            else:
                st.info("No documents uploaded yet")
                
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")

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
