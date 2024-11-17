from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx
import re
from pathlib import Path
import logging
from datetime import datetime
import json

from app.database.vector_store import VectorStore
from app.cache.redis_cache import RedisCache
from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentGraph:
    """Manage cross-document references and relationships."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        cache: Optional[RedisCache] = None
    ):
        """
        Initialize document graph.
        
        Args:
            vector_store: Vector store instance
            cache: Optional Redis cache instance
        """
        self.vector_store = vector_store
        self.cache = cache or RedisCache()
        self.graph = nx.DiGraph()
        
        # Reference patterns
        self.patterns = {
            "url": r'https?://[^\s<>"]+|www\.[^\s<>"]+',
            "file": r'(?i)(?:see|refer to|check)\s+([\'"])(.*?\.(?:txt|pdf|doc|docx))\1',
            "section": r'(?i)(?:see|as shown in|described in)\s+(?:section|chapter)\s+[\d\.]+',
            "citation": r'\[([\d,\s]+)\]|\(\w+\s*(?:et al\.?)?,\s*\d{4}\)'
        }
    
    async def build_graph(
        self,
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        Build document reference graph.
        
        Args:
            documents: List of documents with content and metadata
        """
        try:
            # Clear existing graph
            self.graph.clear()
            
            # Add all documents as nodes
            for doc in documents:
                self.graph.add_node(
                    doc["id"],
                    title=doc.get("filename", ""),
                    type=doc.get("content_type", ""),
                    metadata=doc.get("metadata", {})
                )
            
            # Find and add references
            for doc in documents:
                references = await self._find_references(doc)
                for ref_type, ref_ids in references.items():
                    for ref_id in ref_ids:
                        if ref_id in self.graph:
                            self.graph.add_edge(
                                doc["id"],
                                ref_id,
                                type=ref_type,
                                timestamp=datetime.utcnow().isoformat()
                            )
            
            logger.info(
                f"Built document graph with {self.graph.number_of_nodes()} nodes "
                f"and {self.graph.number_of_edges()} edges"
            )
            
            # Cache graph structure
            await self._cache_graph()
            
        except Exception as e:
            logger.error(f"Error building document graph: {str(e)}")
            raise
    
    async def get_related_documents(
        self,
        document_id: int,
        max_depth: int = 2,
        ref_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get related documents up to specified depth.
        
        Args:
            document_id: Source document ID
            max_depth: Maximum traversal depth
            ref_types: Optional list of reference types to consider
        
        Returns:
            List of related documents with relationship info
        """
        try:
            # Check cache first
            cache_key = f"related_docs:{document_id}:{max_depth}:{str(ref_types)}"
            cached_results = await self.cache.get(cache_key)
            if cached_results:
                return cached_results
            
            if document_id not in self.graph:
                return []
            
            # Get subgraph within max_depth
            subgraph = nx.ego_graph(
                self.graph,
                document_id,
                radius=max_depth,
                undirected=True
            )
            
            # Filter by reference types if specified
            if ref_types:
                edges_to_remove = [
                    (u, v) for u, v, d in subgraph.edges(data=True)
                    if d["type"] not in ref_types
                ]
                subgraph.remove_edges_from(edges_to_remove)
            
            # Format results
            results = []
            for node in subgraph.nodes():
                if node != document_id:
                    # Get shortest path
                    path = nx.shortest_path(
                        subgraph,
                        document_id,
                        node,
                        weight=None
                    )
                    
                    # Get relationship info
                    relationship = []
                    for i in range(len(path) - 1):
                        edge_data = subgraph.edges[path[i], path[i+1]]
                        relationship.append({
                            "from_id": path[i],
                            "to_id": path[i+1],
                            "type": edge_data["type"],
                            "timestamp": edge_data["timestamp"]
                        })
                    
                    results.append({
                        "document_id": node,
                        "title": subgraph.nodes[node]["title"],
                        "type": subgraph.nodes[node]["type"],
                        "metadata": subgraph.nodes[node]["metadata"],
                        "distance": len(path) - 1,
                        "relationship": relationship
                    })
            
            # Sort by distance
            results.sort(key=lambda x: x["distance"])
            
            # Cache results
            await self.cache.set(
                cache_key,
                results,
                ttl=settings.CACHE_DEFAULT_TIMEOUT
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting related documents: {str(e)}")
            return []
    
    async def find_common_references(
        self,
        doc_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Find documents referenced by multiple source documents.
        
        Args:
            doc_ids: List of document IDs to analyze
        
        Returns:
            List of commonly referenced documents with counts
        """
        try:
            if not all(doc_id in self.graph for doc_id in doc_ids):
                return []
            
            # Get all referenced documents for each source
            referenced_docs: List[Set[int]] = []
            for doc_id in doc_ids:
                refs = set()
                for _, target in self.graph.edges(doc_id):
                    refs.add(target)
                referenced_docs.append(refs)
            
            # Find intersection
            common_refs = set.intersection(*referenced_docs)
            
            # Get reference details
            results = []
            for ref_id in common_refs:
                # Count references from source documents
                ref_count = sum(
                    1 for doc_id in doc_ids
                    if self.graph.has_edge(doc_id, ref_id)
                )
                
                results.append({
                    "document_id": ref_id,
                    "title": self.graph.nodes[ref_id]["title"],
                    "type": self.graph.nodes[ref_id]["type"],
                    "metadata": self.graph.nodes[ref_id]["metadata"],
                    "reference_count": ref_count,
                    "reference_types": [
                        self.graph.edges[doc_id, ref_id]["type"]
                        for doc_id in doc_ids
                        if self.graph.has_edge(doc_id, ref_id)
                    ]
                })
            
            # Sort by reference count
            results.sort(key=lambda x: x["reference_count"], reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error finding common references: {str(e)}")
            return []
    
    async def get_citation_graph(
        self,
        document_id: int
    ) -> Dict[str, Any]:
        """
        Get citation graph for visualization.
        
        Args:
            document_id: Source document ID
        
        Returns:
            Graph data in visualization format
        """
        try:
            if document_id not in self.graph:
                return {"nodes": [], "edges": []}
            
            # Get citation subgraph
            subgraph = nx.ego_graph(
                self.graph,
                document_id,
                radius=2,
                undirected=True
            )
            
            # Format for visualization
            nodes = []
            edges = []
            
            for node in subgraph.nodes():
                nodes.append({
                    "id": node,
                    "label": subgraph.nodes[node]["title"],
                    "type": subgraph.nodes[node]["type"],
                    "level": nx.shortest_path_length(
                        subgraph,
                        document_id,
                        node
                    )
                })
            
            for u, v, data in subgraph.edges(data=True):
                edges.append({
                    "from": u,
                    "to": v,
                    "type": data["type"]
                })
            
            return {
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            logger.error(f"Error getting citation graph: {str(e)}")
            return {"nodes": [], "edges": []}
    
    async def _find_references(
        self,
        document: Dict[str, Any]
    ) -> Dict[str, Set[int]]:
        """Find references in document content."""
        references = {
            "url": set(),
            "file": set(),
            "section": set(),
            "citation": set()
        }
        
        content = document["content"]
        
        # Find URLs
        urls = re.findall(self.patterns["url"], content)
        if urls:
            # Look up documents by URL in metadata
            for url in urls:
                doc_id = await self._find_document_by_metadata(
                    "url",
                    url
                )
                if doc_id:
                    references["url"].add(doc_id)
        
        # Find file references
        file_refs = re.findall(self.patterns["file"], content)
        if file_refs:
            for _, filename in file_refs:
                doc_id = await self._find_document_by_filename(filename)
                if doc_id:
                    references["file"].add(doc_id)
        
        # Find section references
        section_refs = re.findall(self.patterns["section"], content)
        if section_refs:
            # Look up documents by section reference
            for section in section_refs:
                doc_id = await self._find_document_by_metadata(
                    "section",
                    section
                )
                if doc_id:
                    references["section"].add(doc_id)
        
        # Find citations
        citations = re.findall(self.patterns["citation"], content)
        if citations:
            # Look up documents by citation
            for citation in citations:
                doc_id = await self._find_document_by_metadata(
                    "citation",
                    citation
                )
                if doc_id:
                    references["citation"].add(doc_id)
        
        return references
    
    async def _find_document_by_filename(
        self,
        filename: str
    ) -> Optional[int]:
        """Find document ID by filename."""
        for node, data in self.graph.nodes(data=True):
            if data["title"].lower() == filename.lower():
                return node
        return None
    
    async def _find_document_by_metadata(
        self,
        key: str,
        value: str
    ) -> Optional[int]:
        """Find document ID by metadata value."""
        for node, data in self.graph.nodes(data=True):
            if data["metadata"].get(key) == value:
                return node
        return None
    
    async def _cache_graph(self) -> None:
        """Cache graph structure."""
        try:
            # Convert graph to serializable format
            graph_data = {
                "nodes": [
                    {
                        "id": n,
                        "data": d
                    }
                    for n, d in self.graph.nodes(data=True)
                ],
                "edges": [
                    {
                        "from": u,
                        "to": v,
                        "data": d
                    }
                    for u, v, d in self.graph.edges(data=True)
                ]
            }
            
            await self.cache.set(
                "document_graph",
                graph_data,
                ttl=settings.CACHE_DEFAULT_TIMEOUT
            )
            
        except Exception as e:
            logger.error(f"Error caching graph: {str(e)}")

# Global document graph instance will be initialized with vector store
document_graph = None
