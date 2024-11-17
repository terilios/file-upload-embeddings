from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from app.document_processing.embeddings import (
    generate_embeddings,
    compute_similarity,
    generate_hybrid_embedding
)
from config.settings import settings

class QueryClassifier:
    """Classify queries to determine the optimal retrieval strategy."""
    
    FACTUAL_PATTERNS = [
        r'\b(?:what|who|when|where|which|how many|how much)\b',
        r'\b(?:define|explain|tell me about)\b',
        r'\b(?:find|search|locate)\b'
    ]
    
    ANALYTICAL_PATTERNS = [
        r'\b(?:why|how|compare|analyze|evaluate|assess)\b',
        r'\b(?:relationship|difference|similarity|impact)\b',
        r'\b(?:pros|cons|advantages|disadvantages)\b'
    ]
    
    SUMMARY_PATTERNS = [
        r'\b(?:summarize|overview|brief|main points)\b',
        r'\b(?:key|important|significant)\b',
        r'\b(?:conclusion|takeaway)\b'
    ]
    
    @classmethod
    def classify_query(cls, query: str) -> str:
        """
        Classify the query type.
        Returns: 'factual', 'analytical', or 'summary'
        """
        query = query.lower()
        
        # Count matches for each pattern type
        factual_score = sum(1 for pattern in cls.FACTUAL_PATTERNS if re.search(pattern, query))
        analytical_score = sum(1 for pattern in cls.ANALYTICAL_PATTERNS if re.search(pattern, query))
        summary_score = sum(1 for pattern in cls.SUMMARY_PATTERNS if re.search(pattern, query))
        
        # Return the type with highest score
        scores = {
            'factual': factual_score,
            'analytical': analytical_score,
            'summary': summary_score
        }
        
        return max(scores.items(), key=lambda x: x[1])[0]

def retrieve_relevant_chunks(
    query: str,
    db: Session,
    document_id: Optional[int] = None,
    top_k: int = settings.TOP_K_RESULTS
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant document chunks using hybrid search strategy.
    
    Args:
        query: User query
        db: Database session
        document_id: Optional document ID to filter results
        top_k: Number of results to return
    
    Returns:
        List of relevant chunks with metadata
    """
    # Classify query type
    query_type = QueryClassifier.classify_query(query)
    
    # Adjust retrieval parameters based on query type
    if query_type == 'factual':
        similarity_threshold = settings.SIMILARITY_THRESHOLD
        chunk_limit = top_k
    elif query_type == 'analytical':
        similarity_threshold = settings.SIMILARITY_THRESHOLD * 0.8  # More lenient
        chunk_limit = top_k * 2  # Retrieve more chunks for analysis
    else:  # summary
        similarity_threshold = settings.SIMILARITY_THRESHOLD * 0.9
        chunk_limit = top_k * 1.5
    
    # Generate query embedding
    query_embedding = generate_embeddings(query)
    
    # Build base query
    base_query = """
        WITH ranked_chunks AS (
            SELECT 
                c.id,
                c.content,
                c.metadata,
                c.embedding <=> %s as similarity,
                ts_rank_cd(
                    to_tsvector('english', c.content),
                    plainto_tsquery('english', %s)
                ) as text_rank
            FROM chunks c
    """
    
    params = [query_embedding, query]
    
    # Add document filter if specified
    if document_id:
        base_query += " WHERE c.document_id = %s"
        params.append(document_id)
    
    # Complete query with hybrid ranking
    query_sql = base_query + f"""
        )
        SELECT 
            id,
            content,
            metadata,
            similarity,
            text_rank,
            (0.7 * (1 - similarity) + 0.3 * text_rank) as hybrid_score
        FROM ranked_chunks
        WHERE similarity < %s
        ORDER BY hybrid_score DESC
        LIMIT %s
    """
    
    params.extend([similarity_threshold, int(chunk_limit)])
    
    # Execute query
    results = db.execute(query_sql, params).fetchall()
    
    # Process results
    chunks = []
    for row in results:
        chunk = {
            "id": row[0],
            "content": row[1],
            "metadata": row[2],
            "similarity_score": 1 - row[3],  # Convert distance to similarity
            "text_rank": row[4],
            "hybrid_score": row[5]
        }
        chunks.append(chunk)
    
    # Post-process results based on query type
    if query_type == 'analytical':
        # For analytical queries, ensure context continuity
        chunks = ensure_context_continuity(chunks, db)
    elif query_type == 'summary':
        # For summary queries, prioritize diverse sections
        chunks = diversify_sections(chunks)
    
    return chunks[:top_k]

def ensure_context_continuity(
    chunks: List[Dict[str, Any]],
    db: Session
) -> List[Dict[str, Any]]:
    """
    Ensure context continuity by including adjacent chunks when necessary.
    """
    continuous_chunks = []
    added_ids = set()
    
    for chunk in chunks:
        if chunk["id"] in added_ids:
            continue
            
        continuous_chunks.append(chunk)
        added_ids.add(chunk["id"])
        
        # Get document_id from metadata
        doc_id = chunk["metadata"].get("document_id")
        if not doc_id:
            continue
        
        # Find adjacent chunks
        query = """
            SELECT id, content, metadata
            FROM chunks
            WHERE document_id = %s
            AND id != %s
            AND ABS(
                (metadata->>'start_idx')::int - 
                (%s)::int
            ) <= 1000
            ORDER BY (metadata->>'start_idx')::int
            LIMIT 2
        """
        
        adjacent = db.execute(
            query,
            (doc_id, chunk["id"], chunk["metadata"].get("start_idx", 0))
        ).fetchall()
        
        for adj in adjacent:
            if adj[0] not in added_ids:
                continuous_chunks.append({
                    "id": adj[0],
                    "content": adj[1],
                    "metadata": adj[2],
                    "similarity_score": chunk["similarity_score"] * 0.9,  # Slightly lower score
                    "text_rank": chunk["text_rank"] * 0.9,
                    "hybrid_score": chunk["hybrid_score"] * 0.9
                })
                added_ids.add(adj[0])
    
    return continuous_chunks

def diversify_sections(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure diversity in section representation for summary queries.
    """
    sections = {}
    
    # Group chunks by section
    for chunk in chunks:
        section = chunk["metadata"].get("section_header", "default")
        if section not in sections:
            sections[section] = []
        sections[section].append(chunk)
    
    # Select top chunks from each section
    diverse_chunks = []
    while sections and len(diverse_chunks) < len(chunks):
        for section in list(sections.keys()):
            if not sections[section]:
                del sections[section]
                continue
            
            # Add highest scoring chunk from this section
            section_chunks = sections[section]
            best_chunk = max(section_chunks, key=lambda x: x["hybrid_score"])
            diverse_chunks.append(best_chunk)
            section_chunks.remove(best_chunk)
    
    return diverse_chunks
