from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import openai
from datetime import datetime

from app.database.vector_store import VectorStore
from app.document_processing.embeddings import generate_embeddings
from app.database.models import ChatSession, ChatMessage
from config.settings import settings
from app.backend.main import get_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    document_id: Optional[int] = None
    session_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[dict]
    session_id: int

async def generate_response(
    query: str,
    relevant_chunks: List[tuple],
    chat_history: List[dict] = None
) -> str:
    """
    Generate a response using OpenAI's API with relevant context.
    """
    # Prepare context from relevant chunks
    context = "\n\n".join([
        f"Content: {chunk.content}\nRelevance: {score:.2f}"
        for chunk, score in relevant_chunks
    ])
    
    # Prepare chat history context
    history_context = ""
    if chat_history:
        history_context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in chat_history[-5:]  # Include last 5 messages
        ])
    
    # Construct the prompt
    system_prompt = """You are a helpful assistant that answers questions based on the provided document context. 
    Always ground your answers in the given context and cite relevant sources. 
    If the context doesn't contain enough information to answer the question, say so."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Context:\n{context}\n\nChat History:\n{history_context}\n\nQuestion: {query}
        
        Please provide a clear and concise answer based on the context provided. Include relevant citations."""}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=settings.DEFAULT_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

@router.post("/query", response_model=ChatResponse)
async def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Process a query and return a response based on relevant documents.
    """
    try:
        vector_store = VectorStore(db)
        
        # Generate embedding for the query
        query_embedding = generate_embeddings(request.query)
        
        # Retrieve relevant chunks
        relevant_chunks = await vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=settings.TOP_K_RESULTS
        )
        
        if not relevant_chunks:
            return ChatResponse(
                response="I couldn't find any relevant information in the documents to answer your question. Could you please rephrase your question or provide more context?",
                sources=[],
                session_id=request.session_id or 0
            )
        
        # Get or create chat session
        session = None
        if request.session_id:
            session = db.query(ChatSession).get(request.session_id)
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail=f"Chat session {request.session_id} not found"
                )
        
        if not session:
            session = ChatSession(
                metadata={"created_at": str(datetime.utcnow())}
            )
            db.add(session)
            db.flush()
        
        # Get chat history
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in session.messages
        ]
        
        # Generate response
        response_text = await generate_response(
            query=request.query,
            relevant_chunks=relevant_chunks,
            chat_history=chat_history
        )
        
        # Store messages
        user_message = ChatMessage(
            session_id=session.id,
            role="user",
            content=request.query
        )
        assistant_message = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=response_text,
            metadata={
                "sources": [
                    {
                        "content": chunk.content,
                        "score": float(score),
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index
                    }
                    for chunk, score in relevant_chunks
                ]
            }
        )
        
        db.add(user_message)
        db.add(assistant_message)
        db.commit()
        
        # Prepare sources for response
        sources = [
            {
                "content": chunk.content,
                "score": float(score),
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index
            }
            for chunk, score in relevant_chunks
        ]
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            session_id=session.id
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@router.get("/sessions/{session_id}")
async def get_chat_session(
    session_id: int,
    db: Session = Depends(get_db)
):
    """
    Get chat session details and messages.
    """
    session = db.query(ChatSession).get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Chat session {session_id} not found"
        )
    
    messages = [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at,
            "metadata": msg.metadata
        }
        for msg in session.messages
    ]
    
    return {
        "session_id": session.id,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "metadata": session.metadata,
        "messages": messages
    }

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a chat session and its messages.
    """
    session = db.query(ChatSession).get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Chat session {session_id} not found"
        )
    
    db.delete(session)
    db.commit()
    
    return {"message": "Chat session deleted successfully"}
