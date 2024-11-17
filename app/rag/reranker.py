from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from app.document_processing.embeddings import compute_similarity, generate_embeddings
from config.settings import settings

# Load spaCy model for text processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class RelevanceScorer:
    """Calculate relevance scores using multiple methods."""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english'
        )
    
    def semantic_similarity(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> List[float]:
        """Calculate semantic similarity using embeddings."""
        query_embedding = generate_embeddings(query)
        
        scores = []
        for chunk in chunks:
            if "embedding" in chunk:
                score = compute_similarity(query_embedding, chunk["embedding"])
            else:
                chunk_embedding = generate_embeddings(chunk["content"])
                score = compute_similarity(query_embedding, chunk_embedding)
            scores.append(score)
        
        return scores
    
    def keyword_similarity(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> List[float]:
        """Calculate keyword-based similarity using TF-IDF."""
        texts = [query] + [chunk["content"] for chunk in chunks]
        
        # Fit TF-IDF on all texts
        tfidf_matrix = self.tfidf.fit_transform(texts)
        
        # Calculate similarity between query and each chunk
        query_vector = tfidf_matrix[0:1]
        chunk_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vector, chunk_vectors)
        return similarities[0]
    
    def entity_overlap(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> List[float]:
        """Calculate named entity overlap score."""
        query_doc = nlp(query)
        query_entities = set([ent.text.lower() for ent in query_doc.ents])
        
        if not query_entities:
            return [0.0] * len(chunks)
        
        scores = []
        for chunk in chunks:
            chunk_doc = nlp(chunk["content"])
            chunk_entities = set([ent.text.lower() for ent in chunk_doc.ents])
            
            if chunk_entities:
                overlap = len(query_entities & chunk_entities)
                score = overlap / len(query_entities)
            else:
                score = 0.0
            
            scores.append(score)
        
        return scores

def rerank_results(
    query: str,
    chunks: List[Dict[str, Any]],
    weights: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    Rerank retrieved chunks using multiple scoring methods.
    
    Args:
        query: User query
        chunks: List of retrieved chunks with scores
        weights: Optional custom weights for different scoring methods
    
    Returns:
        Reranked list of chunks with updated scores
    """
    if not chunks:
        return []
    
    # Default weights if not provided
    if weights is None:
        weights = {
            "semantic": 0.4,
            "keyword": 0.3,
            "entity": 0.2,
            "original": 0.1
        }
    
    scorer = RelevanceScorer()
    
    # Calculate scores using different methods
    semantic_scores = scorer.semantic_similarity(query, chunks)
    keyword_scores = scorer.keyword_similarity(query, chunks)
    entity_scores = scorer.entity_overlap(query, chunks)
    
    # Get original scores (normalized)
    original_scores = [chunk.get("similarity_score", 0) for chunk in chunks]
    if original_scores:
        max_score = max(original_scores)
        if max_score > 0:
            original_scores = [score / max_score for score in original_scores]
    
    # Combine scores
    final_scores = []
    for i in range(len(chunks)):
        combined_score = (
            weights["semantic"] * semantic_scores[i] +
            weights["keyword"] * keyword_scores[i] +
            weights["entity"] * entity_scores[i] +
            weights["original"] * original_scores[i]
        )
        final_scores.append(combined_score)
    
    # Create reranked list
    reranked_chunks = []
    for chunk, score in zip(chunks, final_scores):
        chunk_copy = chunk.copy()
        chunk_copy["score"] = score
        reranked_chunks.append(chunk_copy)
    
    # Sort by final score
    reranked_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    # Apply post-processing
    reranked_chunks = post_process_results(query, reranked_chunks)
    
    return reranked_chunks

def post_process_results(
    query: str,
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Apply post-processing to reranked results.
    
    - Ensure context continuity
    - Remove redundant information
    - Boost diversity when appropriate
    """
    if not chunks:
        return []
    
    # Remove near-duplicate chunks
    unique_chunks = remove_duplicates(chunks)
    
    # Ensure context continuity
    continuous_chunks = ensure_context_continuity(unique_chunks)
    
    # Boost diversity for certain query types
    if should_boost_diversity(query):
        diverse_chunks = boost_diversity(continuous_chunks)
        return diverse_chunks
    
    return continuous_chunks

def remove_duplicates(
    chunks: List[Dict[str, Any]],
    similarity_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """Remove near-duplicate chunks based on content similarity."""
    if not chunks:
        return []
    
    unique_chunks = [chunks[0]]
    
    for chunk in chunks[1:]:
        is_duplicate = False
        chunk_embedding = generate_embeddings(chunk["content"])
        
        for unique_chunk in unique_chunks:
            unique_embedding = generate_embeddings(unique_chunk["content"])
            similarity = compute_similarity(chunk_embedding, unique_embedding)
            
            if similarity > similarity_threshold:
                is_duplicate = True
                # Keep the chunk with higher score
                if chunk["score"] > unique_chunk["score"]:
                    unique_chunks.remove(unique_chunk)
                    unique_chunks.append(chunk)
                break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
    
    return unique_chunks

def ensure_context_continuity(
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Ensure logical flow between chunks."""
    if not chunks:
        return []
    
    # Sort chunks by document and position
    chunks.sort(key=lambda x: (
        x["metadata"].get("document_id", 0),
        int(x["metadata"].get("start_idx", 0))
    ))
    
    # Adjust scores based on context
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i-1]
        curr_chunk = chunks[i]
        
        # If chunks are from same document and sequential
        if (prev_chunk["metadata"].get("document_id") == 
            curr_chunk["metadata"].get("document_id")):
            prev_end = int(prev_chunk["metadata"].get("end_idx", 0))
            curr_start = int(curr_chunk["metadata"].get("start_idx", 0))
            
            if 0 <= curr_start - prev_end <= 1000:  # Within 1000 chars
                # Boost score of sequential chunks
                curr_chunk["score"] *= 1.1
    
    # Resort by adjusted scores
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks

def should_boost_diversity(query: str) -> bool:
    """Determine if results should prioritize diversity."""
    diversity_indicators = [
        r'\b(?:different|various|multiple|diverse)\b',
        r'\b(?:compare|contrast|versus|vs)\b',
        r'\b(?:overview|summary|summarize)\b'
    ]
    
    return any(re.search(pattern, query, re.IGNORECASE) 
              for pattern in diversity_indicators)

def boost_diversity(
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Boost diversity in results by considering different sections/sources."""
    if not chunks:
        return []
    
    diverse_chunks = []
    seen_sections = set()
    
    # First pass: select chunks from different sections
    for chunk in chunks:
        section = chunk["metadata"].get("section_header", "")
        if section not in seen_sections:
            diverse_chunks.append(chunk)
            seen_sections.add(section)
    
    # Second pass: add remaining high-scoring chunks
    remaining_chunks = [c for c in chunks if c not in diverse_chunks]
    remaining_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    diverse_chunks.extend(remaining_chunks)
    
    return diverse_chunks
