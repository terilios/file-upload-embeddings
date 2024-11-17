from typing import List, Dict, Any, Optional
import asyncio
import time
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

from app.database.vector_store import VectorStore
from app.document_processing.batch_processor import BatchProcessor
from app.rag.hybrid_searcher import HybridSearcher
from app.rag.document_graph import DocumentGraph
from app.rag.context_manager import ContextManager
from app.rag.query_expander import QueryExpander
from app.cache.redis_cache import RedisCache
from .test_data_generator import TestDataGenerator

logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Benchmark system performance metrics."""
    
    def __init__(self, db_session):
        """Initialize benchmark suite."""
        self.db_session = db_session
        self.cache = RedisCache()
        
        # Initialize components
        self.vector_store = VectorStore(db_session)
        self.batch_processor = BatchProcessor(self.vector_store)
        self.hybrid_searcher = HybridSearcher(self.vector_store, self.cache)
        self.document_graph = DocumentGraph(self.vector_store, self.cache)
        self.context_manager = ContextManager(self.cache)
        self.query_expander = QueryExpander(self.cache)
        
        # Initialize test data generator
        self.data_generator = TestDataGenerator()
        
        # Results storage
        self.results = {
            "document_processing": [],
            "search_performance": [],
            "graph_operations": [],
            "context_windows": [],
            "query_expansion": [],
            "cache_performance": [],
            "concurrent_operations": []
        }
    
    async def run_benchmarks(
        self,
        doc_count: int = 100,
        query_count: int = 50,
        concurrent_users: int = 10
    ) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Args:
            doc_count: Number of test documents
            query_count: Number of test queries
            concurrent_users: Number of simulated concurrent users
        
        Returns:
            Dictionary containing benchmark results
        """
        try:
            logger.info("Starting performance benchmarks")
            start_time = time.time()
            
            # Generate test data
            docs = self.data_generator.generate_documents(
                count=doc_count,
                include_code=True,
                include_references=True
            )
            queries = self.data_generator.generate_queries(
                count=query_count,
                complexity="complex"
            )
            
            # Run benchmarks
            await self._benchmark_document_processing(docs)
            await self._benchmark_search_performance(queries)
            await self._benchmark_graph_operations(docs)
            await self._benchmark_context_windows(docs, queries)
            await self._benchmark_query_expansion(queries)
            await self._benchmark_cache_performance(queries)
            await self._benchmark_concurrent_operations(
                docs,
                queries,
                concurrent_users
            )
            
            # Calculate summary statistics
            summary = self._generate_summary()
            
            duration = time.time() - start_time
            logger.info(f"Benchmarks completed in {duration:.2f} seconds")
            
            return {
                "results": self.results,
                "summary": summary,
                "metadata": {
                    "doc_count": doc_count,
                    "query_count": query_count,
                    "concurrent_users": concurrent_users,
                    "duration": duration,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error running benchmarks: {str(e)}")
            raise
    
    async def _benchmark_document_processing(
        self,
        docs: List[Dict[str, Any]]
    ) -> None:
        """Benchmark document processing performance."""
        logger.info("Benchmarking document processing")
        
        batch_sizes = [1, 10, 50, 100]
        for batch_size in batch_sizes:
            if batch_size > len(docs):
                continue
            
            batch = docs[:batch_size]
            start_time = time.time()
            
            processed = await self.batch_processor.process_documents(batch)
            
            duration = time.time() - start_time
            self.results["document_processing"].append({
                "batch_size": batch_size,
                "duration": duration,
                "docs_per_second": batch_size / duration
            })
    
    async def _benchmark_search_performance(
        self,
        queries: List[Dict[str, Any]]
    ) -> None:
        """Benchmark search performance."""
        logger.info("Benchmarking search performance")
        
        for query in queries:
            start_time = time.time()
            
            results = await self.hybrid_searcher.search(
                query["query"],
                top_k=10
            )
            
            duration = time.time() - start_time
            self.results["search_performance"].append({
                "query_type": query["type"],
                "duration": duration,
                "result_count": len(results)
            })
    
    async def _benchmark_graph_operations(
        self,
        docs: List[Dict[str, Any]]
    ) -> None:
        """Benchmark document graph operations."""
        logger.info("Benchmarking graph operations")
        
        # Benchmark graph building
        start_time = time.time()
        await self.document_graph.build_graph(docs)
        build_duration = time.time() - start_time
        
        # Benchmark traversal operations
        traversal_times = []
        for doc in docs[:10]:  # Sample 10 documents
            start_time = time.time()
            related = await self.document_graph.get_related_documents(
                doc["id"],
                max_depth=2
            )
            traversal_times.append(time.time() - start_time)
        
        self.results["graph_operations"].append({
            "doc_count": len(docs),
            "build_duration": build_duration,
            "avg_traversal_time": np.mean(traversal_times)
        })
    
    async def _benchmark_context_windows(
        self,
        docs: List[Dict[str, Any]],
        queries: List[Dict[str, Any]]
    ) -> None:
        """Benchmark context window operations."""
        logger.info("Benchmarking context windows")
        
        # Create test chunks
        chunks = []
        for doc in docs:
            chunks.extend([
                {
                    "content": f"Chunk {i} from {doc['id']}",
                    "embedding": np.random.rand(1536).tolist()
                }
                for i in range(5)
            ])
        
        for query in queries[:10]:  # Sample 10 queries
            start_time = time.time()
            
            windows = await self.context_manager.create_context_windows(
                chunks,
                np.random.rand(1536).tolist()  # Mock query embedding
            )
            
            duration = time.time() - start_time
            self.results["context_windows"].append({
                "chunk_count": len(chunks),
                "window_count": len(windows),
                "duration": duration
            })
    
    async def _benchmark_query_expansion(
        self,
        queries: List[Dict[str, Any]]
    ) -> None:
        """Benchmark query expansion."""
        logger.info("Benchmarking query expansion")
        
        for query in queries:
            start_time = time.time()
            
            expanded = await self.query_expander.expand_query(query["query"])
            
            duration = time.time() - start_time
            self.results["query_expansion"].append({
                "query_type": query["type"],
                "duration": duration,
                "expansion_count": len(expanded["semantic_terms"])
            })
    
    async def _benchmark_cache_performance(
        self,
        queries: List[Dict[str, Any]]
    ) -> None:
        """Benchmark cache performance."""
        logger.info("Benchmarking cache performance")
        
        # Clear cache
        await self.cache.clear()
        
        # First run (cache misses)
        miss_times = []
        for query in queries[:10]:
            start_time = time.time()
            await self.query_expander.expand_query(query["query"])
            miss_times.append(time.time() - start_time)
        
        # Second run (cache hits)
        hit_times = []
        for query in queries[:10]:
            start_time = time.time()
            await self.query_expander.expand_query(query["query"])
            hit_times.append(time.time() - start_time)
        
        self.results["cache_performance"].append({
            "avg_miss_time": np.mean(miss_times),
            "avg_hit_time": np.mean(hit_times),
            "speedup_factor": np.mean(miss_times) / np.mean(hit_times)
        })
    
    async def _benchmark_concurrent_operations(
        self,
        docs: List[Dict[str, Any]],
        queries: List[Dict[str, Any]],
        concurrent_users: int
    ) -> None:
        """Benchmark concurrent operations."""
        logger.info("Benchmarking concurrent operations")
        
        async def simulate_user():
            # Simulate user operations
            start_time = time.time()
            
            # Process a document
            doc = random.choice(docs)
            await self.batch_processor.process_documents([doc])
            
            # Perform a search
            query = random.choice(queries)
            expanded = await self.query_expander.expand_query(query["query"])
            results = await self.hybrid_searcher.search(
                expanded["expanded_query"]
            )
            
            return time.time() - start_time
        
        # Run concurrent user simulations
        tasks = [
            simulate_user()
            for _ in range(concurrent_users)
        ]
        
        durations = await asyncio.gather(*tasks)
        
        self.results["concurrent_operations"].append({
            "concurrent_users": concurrent_users,
            "avg_duration": np.mean(durations),
            "max_duration": max(durations),
            "min_duration": min(durations)
        })
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "document_processing": {
                "avg_docs_per_second": np.mean([
                    r["docs_per_second"]
                    for r in self.results["document_processing"]
                ])
            },
            "search_performance": {
                "avg_duration": np.mean([
                    r["duration"]
                    for r in self.results["search_performance"]
                ])
            },
            "graph_operations": {
                "avg_traversal_time": np.mean([
                    r["avg_traversal_time"]
                    for r in self.results["graph_operations"]
                ])
            },
            "context_windows": {
                "avg_duration": np.mean([
                    r["duration"]
                    for r in self.results["context_windows"]
                ])
            },
            "query_expansion": {
                "avg_duration": np.mean([
                    r["duration"]
                    for r in self.results["query_expansion"]
                ])
            },
            "cache_performance": {
                "avg_speedup": np.mean([
                    r["speedup_factor"]
                    for r in self.results["cache_performance"]
                ])
            },
            "concurrent_operations": {
                "avg_duration": np.mean([
                    r["avg_duration"]
                    for r in self.results["concurrent_operations"]
                ])
            }
        }

# Create benchmark runner function
async def run_benchmarks(
    db_session,
    doc_count: int = 100,
    query_count: int = 50,
    concurrent_users: int = 10
) -> Dict[str, Any]:
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark(db_session)
    return await benchmark.run_benchmarks(
        doc_count,
        query_count,
        concurrent_users
    )
