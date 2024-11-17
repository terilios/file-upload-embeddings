from typing import List, Dict, Any
import random
from pathlib import Path
import json
import numpy as np
from datetime import datetime, timedelta

class TestDataGenerator:
    """Generate test data for integration testing."""
    
    def __init__(self):
        """Initialize test data generator."""
        # Sample content templates
        self.technical_templates = [
            "The {system} implements {algorithm} for {task}.",
            "Using {technology} to optimize {process} performance.",
            "Implementation of {method} in the {component} subsystem.",
            "Analysis of {metric} in {context} scenarios.",
            "Evaluation of {approach} for {problem} solving."
        ]
        
        self.code_templates = [
            """
            def {function_name}({params}):
                \"""
                {docstring}
                \"""
                {code_body}
            """,
            """
            class {class_name}:
                def __init__(self, {params}):
                    {init_body}
                
                def {method_name}(self):
                    {method_body}
            """,
            """
            async def {async_function}({params}):
                try:
                    {async_body}
                except Exception as e:
                    {error_handling}
            """
        ]
        
        # Domain-specific terms
        self.technologies = [
            "PostgreSQL", "Redis", "Docker", "FastAPI", "Python",
            "Vector Database", "Neural Networks", "Machine Learning"
        ]
        
        self.algorithms = [
            "BM25", "Vector Similarity", "Cosine Distance",
            "Semantic Search", "Query Expansion", "Hybrid Retrieval"
        ]
        
        self.components = [
            "Database", "Cache", "API", "Frontend", "Backend",
            "Processing Pipeline", "Search Engine", "Monitoring System"
        ]
        
        self.metrics = [
            "Latency", "Throughput", "Accuracy", "Precision",
            "Recall", "F1 Score", "Cache Hit Rate", "Response Time"
        ]
    
    def generate_documents(
        self,
        count: int = 10,
        include_code: bool = True,
        include_references: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate test documents.
        
        Args:
            count: Number of documents to generate
            include_code: Whether to include code documents
            include_references: Whether to add cross-references
        
        Returns:
            List of test documents
        """
        documents = []
        doc_ids = list(range(1, count + 1))
        
        for doc_id in doc_ids:
            # Determine document type
            is_code = include_code and random.random() < 0.3
            
            if is_code:
                doc = self._generate_code_document(doc_id)
            else:
                doc = self._generate_technical_document(doc_id)
            
            # Add references if enabled
            if include_references and len(documents) > 0:
                doc["content"] += self._add_references(doc_ids, doc_id)
            
            documents.append(doc)
        
        return documents
    
    def generate_queries(
        self,
        count: int = 5,
        complexity: str = "medium"
    ) -> List[Dict[str, Any]]:
        """
        Generate test queries.
        
        Args:
            count: Number of queries to generate
            complexity: Query complexity (simple, medium, complex)
        
        Returns:
            List of test queries with expected results
        """
        queries = []
        
        for _ in range(count):
            if complexity == "simple":
                query = self._generate_simple_query()
            elif complexity == "complex":
                query = self._generate_complex_query()
            else:
                query = self._generate_medium_query()
            
            queries.append(query)
        
        return queries
    
    def _generate_technical_document(self, doc_id: int) -> Dict[str, Any]:
        """Generate technical document."""
        # Generate content from templates
        paragraphs = []
        for _ in range(random.randint(3, 7)):
            template = random.choice(self.technical_templates)
            paragraph = template.format(
                system=random.choice(self.components),
                algorithm=random.choice(self.algorithms),
                task=f"{random.choice(self.metrics)} optimization",
                technology=random.choice(self.technologies),
                process=random.choice(self.components).lower(),
                method=random.choice(self.algorithms),
                component=random.choice(self.components).lower(),
                metric=random.choice(self.metrics).lower(),
                context=random.choice(self.components).lower(),
                approach=random.choice(self.algorithms),
                problem=f"{random.choice(self.metrics).lower()} improvement"
            )
            paragraphs.append(paragraph)
        
        content = "\n\n".join(paragraphs)
        
        return {
            "id": doc_id,
            "content": content,
            "filename": f"technical_doc_{doc_id}.txt",
            "content_type": "text/plain",
            "file_size": len(content),
            "metadata": {
                "type": "technical",
                "created_at": self._random_date(),
                "version": "1.0"
            }
        }
    
    def _generate_code_document(self, doc_id: int) -> Dict[str, Any]:
        """Generate code document."""
        template = random.choice(self.code_templates)
        
        # Generate code content
        content = template.format(
            function_name=f"process_{random.choice(self.components).lower()}",
            class_name=f"{random.choice(self.components)}Processor",
            async_function=f"optimize_{random.choice(self.metrics).lower()}",
            params="data: Dict[str, Any]",
            docstring=f"Process {random.choice(self.components).lower()} data.",
            code_body="return processed_data",
            init_body="self.initialized = True",
            method_name=f"calculate_{random.choice(self.metrics).lower()}",
            method_body="return result",
            async_body="await process_data()",
            error_handling="logger.error(str(e))"
        )
        
        return {
            "id": doc_id,
            "content": content,
            "filename": f"code_{doc_id}.py",
            "content_type": "text/x-python",
            "file_size": len(content),
            "metadata": {
                "type": "code",
                "language": "python",
                "created_at": self._random_date(),
                "version": "1.0"
            }
        }
    
    def _add_references(
        self,
        doc_ids: List[int],
        current_id: int
    ) -> str:
        """Add references to other documents."""
        references = []
        
        # Add 1-3 references
        for _ in range(random.randint(1, 3)):
            ref_id = random.choice(doc_ids)
            if ref_id != current_id:
                ref_type = random.choice([
                    f"\n\nSee document_{ref_id} for more details.",
                    f"\n\nAs described in doc_{ref_id}.",
                    f"\n\nReference: [doc_{ref_id}]"
                ])
                references.append(ref_type)
        
        return "".join(references)
    
    def _generate_simple_query(self) -> Dict[str, Any]:
        """Generate simple query."""
        component = random.choice(self.components)
        return {
            "query": f"What is {component.lower()}?",
            "type": "simple",
            "expected_terms": [component.lower()],
            "metadata": {"complexity": "simple"}
        }
    
    def _generate_medium_query(self) -> Dict[str, Any]:
        """Generate medium complexity query."""
        component = random.choice(self.components)
        metric = random.choice(self.metrics)
        return {
            "query": f"How to improve {metric.lower()} in {component.lower()}?",
            "type": "medium",
            "expected_terms": [metric.lower(), component.lower()],
            "metadata": {"complexity": "medium"}
        }
    
    def _generate_complex_query(self) -> Dict[str, Any]:
        """Generate complex query."""
        component1 = random.choice(self.components)
        component2 = random.choice(self.components)
        algorithm = random.choice(self.algorithms)
        return {
            "query": f"Compare {algorithm} implementation between {component1.lower()} and {component2.lower()} for performance optimization",
            "type": "complex",
            "expected_terms": [
                algorithm.lower(),
                component1.lower(),
                component2.lower(),
                "performance",
                "optimization"
            ],
            "metadata": {"complexity": "complex"}
        }
    
    def _random_date(self) -> str:
        """Generate random date within last year."""
        days = random.randint(0, 365)
        date = datetime.now() - timedelta(days=days)
        return date.isoformat()

# Create global test data generator instance
test_data_generator = TestDataGenerator()
