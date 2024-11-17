from typing import List, Dict, Any, Optional, Set
import spacy
from spacy.tokens import Doc
import numpy as np
import logging
from datetime import datetime
import json

from app.document_processing.embeddings import generate_embeddings
from app.cache.redis_cache import RedisCache
from config.settings import settings

logger = logging.getLogger(__name__)

class QueryExpander:
    """Expand queries with synonyms and semantic terms."""
    
    def __init__(
        self,
        cache: Optional[RedisCache] = None,
        max_synonyms: int = 3,
        max_semantic_terms: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize query expander.
        
        Args:
            cache: Optional Redis cache instance
            max_synonyms: Maximum synonyms per term
            max_semantic_terms: Maximum semantic terms to add
            similarity_threshold: Minimum similarity for semantic terms
        """
        self.cache = cache or RedisCache()
        self.max_synonyms = max_synonyms
        self.max_semantic_terms = max_semantic_terms
        self.similarity_threshold = similarity_threshold
        
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Custom stopwords to exclude from expansion
        self.custom_stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on",
            "at", "to", "for", "with", "by", "from", "up", "about",
            "into", "over", "after"
        }
    
    async def expand_query(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Expand query with synonyms and semantic terms.
        
        Args:
            query: Original query string
            context: Optional context documents for semantic expansion
        
        Returns:
            Dictionary containing expanded query information
        """
        try:
            # Check cache first
            cache_key = f"expanded_query:{query}:{hash(str(context))}"
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Process query
            doc = self.nlp(query)
            
            # Get important terms
            terms = await self._extract_terms(doc)
            
            # Get synonyms for terms
            synonyms = await self._get_synonyms(terms)
            
            # Get semantic terms
            semantic_terms = await self._get_semantic_terms(
                query,
                terms,
                context
            )
            
            # Combine expansions
            expanded = await self._combine_expansions(
                query,
                terms,
                synonyms,
                semantic_terms
            )
            
            # Cache result
            await self.cache.set(
                cache_key,
                expanded,
                ttl=settings.CACHE_DEFAULT_TIMEOUT
            )
            
            return expanded
            
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            return {
                "original_query": query,
                "expanded_query": query,
                "terms": [],
                "synonyms": {},
                "semantic_terms": [],
                "metadata": {}
            }
    
    async def _extract_terms(self, doc: Doc) -> List[str]:
        """Extract important terms from query."""
        try:
            terms = []
            
            for token in doc:
                # Include tokens that are:
                # - Not stopwords (custom or spaCy)
                # - Not punctuation
                # - Nouns, verbs, or adjectives
                if (
                    not token.is_stop and
                    not token.is_punct and
                    token.text.lower() not in self.custom_stopwords and
                    token.pos_ in {"NOUN", "VERB", "ADJ"}
                ):
                    terms.append(token.text.lower())
            
            # Add noun phrases
            for chunk in doc.noun_chunks:
                if (
                    chunk.text.lower() not in self.custom_stopwords and
                    len(chunk) > 1  # Multi-word phrases only
                ):
                    terms.append(chunk.text.lower())
            
            return list(set(terms))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting terms: {str(e)}")
            return []
    
    async def _get_synonyms(
        self,
        terms: List[str]
    ) -> Dict[str, List[str]]:
        """Get synonyms for terms using WordNet."""
        try:
            synonyms = {}
            
            for term in terms:
                # Get synsets for term
                synsets = self.nlp.vocab.vectors_for_string(term)
                if not synsets:
                    continue
                
                # Get unique lemmas as synonyms
                term_synonyms = set()
                for synset in synsets[:self.max_synonyms]:
                    for lemma in synset:
                        if (
                            lemma.lower() != term and
                            lemma.lower() not in self.custom_stopwords
                        ):
                            term_synonyms.add(lemma.lower())
                
                if term_synonyms:
                    synonyms[term] = list(term_synonyms)
            
            return synonyms
            
        except Exception as e:
            logger.error(f"Error getting synonyms: {str(e)}")
            return {}
    
    async def _get_semantic_terms(
        self,
        query: str,
        terms: List[str],
        context: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Get semantically related terms."""
        try:
            semantic_terms = set()
            
            # Generate query embedding
            query_embedding = await generate_embeddings(query)
            
            # Get candidate terms from context if available
            candidates = set()
            if context:
                for doc in context:
                    doc_terms = await self._extract_terms(
                        self.nlp(doc["content"])
                    )
                    candidates.update(doc_terms)
            else:
                # Use pre-defined domain-specific terms
                candidates = await self._get_domain_terms()
            
            # Filter out original terms
            candidates = candidates - set(terms)
            
            # Get embeddings for candidates
            candidate_embeddings = {
                term: await generate_embeddings(term)
                for term in candidates
            }
            
            # Calculate similarities and select top terms
            similarities = []
            for term, embedding in candidate_embeddings.items():
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) *
                    np.linalg.norm(embedding)
                )
                if similarity >= self.similarity_threshold:
                    similarities.append((term, similarity))
            
            # Sort by similarity and take top terms
            similarities.sort(key=lambda x: x[1], reverse=True)
            semantic_terms.update(
                term for term, _ in similarities[:self.max_semantic_terms]
            )
            
            return list(semantic_terms)
            
        except Exception as e:
            logger.error(f"Error getting semantic terms: {str(e)}")
            return []
    
    async def _combine_expansions(
        self,
        query: str,
        terms: List[str],
        synonyms: Dict[str, List[str]],
        semantic_terms: List[str]
    ) -> Dict[str, Any]:
        """Combine different types of query expansion."""
        try:
            # Build expanded query
            expansion_parts = [query]  # Start with original query
            
            # Add synonyms
            for term, term_synonyms in synonyms.items():
                expansion_parts.extend(term_synonyms)
            
            # Add semantic terms
            expansion_parts.extend(semantic_terms)
            
            # Join unique terms
            expanded_query = " ".join(set(expansion_parts))
            
            return {
                "original_query": query,
                "expanded_query": expanded_query,
                "terms": terms,
                "synonyms": synonyms,
                "semantic_terms": semantic_terms,
                "metadata": {
                    "expansion_count": len(expansion_parts) - 1,
                    "synonym_count": sum(len(s) for s in synonyms.values()),
                    "semantic_term_count": len(semantic_terms),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error combining expansions: {str(e)}")
            return {
                "original_query": query,
                "expanded_query": query,
                "terms": terms,
                "synonyms": synonyms,
                "semantic_terms": semantic_terms,
                "metadata": {}
            }
    
    async def _get_domain_terms(self) -> Set[str]:
        """Get domain-specific terms for semantic expansion."""
        # This could be extended with domain-specific terminology
        return {
            # Technical terms
            "algorithm", "database", "interface", "protocol", "framework",
            "architecture", "implementation", "configuration", "deployment",
            "optimization", "scalability", "performance", "security",
            
            # Business terms
            "strategy", "analysis", "requirement", "specification", "solution",
            "integration", "workflow", "process", "management", "resource",
            
            # Document terms
            "document", "content", "format", "structure", "metadata",
            "reference", "section", "chapter", "paragraph", "annotation"
        }

# Global query expander instance
query_expander = QueryExpander()
