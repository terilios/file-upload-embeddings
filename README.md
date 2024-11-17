# File Upload Embeddings System

## Strategic Overview

The File Upload Embeddings System is an enterprise-grade document intelligence platform that transforms how organizations process, understand, and retrieve information from their document repositories. By leveraging advanced vector embeddings and state-of-the-art language models, it bridges the gap between traditional document management systems and modern AI-powered information retrieval.

### Key Value Propositions

- **Intelligent Document Understanding**: Automatically extract and understand content from any document format, including complex tables, code snippets, and scanned documents.
- **Semantic Search & Discovery**: Go beyond keyword matching with context-aware search that understands the meaning behind queries.
- **Enterprise-Ready Architecture**: Built with scalability, security, and performance in mind, supporting high-throughput document processing and real-time retrieval.
- **Comprehensive Analytics**: Deep insights into document usage, search patterns, and system performance through integrated monitoring.

### Target Use Cases

1. **Enterprise Knowledge Management**
   - Centralize and make searchable all corporate documentation
   - Enable semantic search across multiple document formats
   - Maintain document relationships and cross-references

2. **Technical Documentation Systems**
   - Process and analyze code repositories
   - Extract insights from technical specifications
   - Enable context-aware code search and reference

3. **Research & Analysis Platforms**
   - Process academic papers and research documents
   - Extract and analyze data tables and figures
   - Enable cross-document citation and reference tracking

4. **Compliance & Legal Document Processing**
   - Automated document classification and analysis
   - Extract and track key clauses and terms
   - Maintain audit trails and document lineage

## Features

### Document Processing
- ✅ Intelligent chunking with adaptive sizes
- ✅ Vector embeddings generation
- ✅ Table extraction from PDFs
- ✅ OCR for images and scanned documents
- ✅ Code parsing with syntax highlighting
- ✅ Batch processing with parallel execution

### Advanced Retrieval
- ✅ Hybrid search (BM25 + Vector Similarity)
- ✅ Cross-document references
- ✅ Dynamic context windows
- ✅ Query expansion
- ✅ Document graph analysis

### System Features
- ✅ Redis caching
- ✅ Connection pooling
- ✅ Query optimization
- ✅ Performance monitoring
- ✅ Comprehensive logging

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- PostgreSQL 13+
- Redis 6+
- At least 4GB RAM
- 10GB free disk space

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/terilios/file-upload-embeddings.git
   cd file-upload-embeddings
   ```

2. Create a .env file:
   ```env
   OPENAI_API_KEY=your_api_key_here
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=file_upload_embeddings
   REDIS_URL=redis://redis:6379/0
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Access the interfaces:
   - Frontend: http://localhost:8501
   - API Documentation: http://localhost:8000/docs
   - Monitoring Dashboard: http://localhost:3000

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   pytest
   ```

4. Start services individually:
   ```bash
   # Backend
   uvicorn app.backend.main:app --reload --port 8000

   # Frontend
   streamlit run app/frontend/main.py
   ```

## Project Structure

```
.
├── app/
│   ├── backend/          # FastAPI application
│   ├── frontend/         # Streamlit interface
│   ├── database/         # Database models and operations
│   ├── document_processing/  # Document handling
│   ├── cache/           # Caching implementation
│   ├── monitoring/      # Logging and metrics
│   └── rag/            # Retrieval and generation
├── config/             # Configuration files
├── scripts/            # Utility scripts
├── tests/             # Test suite
└── monitoring/        # Monitoring configuration
```

## Usage Guide

### Document Upload
1. Navigate to the frontend interface
2. Use the sidebar to upload documents
3. Supported formats:
   - Text files (.txt)
   - PDFs (.pdf)
   - Word documents (.docx)
   - Images (.jpg, .png)
   - Code files (various extensions)

### Search and Retrieval
1. Enter your query in the chat interface
2. The system will:
   - Expand your query with relevant terms
   - Perform hybrid search
   - Analyze document relationships
   - Create dynamic context windows
   - Return relevant results with citations

### Monitoring
1. Access the Grafana dashboard
2. View metrics for:
   - System performance
   - Query latency
   - Cache hit rates
   - Resource utilization

## Configuration

### Database Settings
```python
# config/settings.py
POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'file_upload_embeddings',
    'pool_size': 20,
    'max_overflow': 10
}
```

### Cache Settings
```python
# config/settings.py
REDIS_CONFIG = {
    'url': 'redis://localhost:6379/0',
    'pool_size': 20,
    'timeout': 300
}
```

### Processing Settings
```python
# config/settings.py
PROCESSING_CONFIG = {
    'chunk_size': 1000,
    'overlap': 200,
    'batch_size': 50,
    'max_workers': 4
}
```

## Performance Optimization

### Connection Pooling
- Database connections are pooled
- Redis connections are pooled
- Connection lifecycle is managed

### Caching Strategy
- Embeddings are cached
- Query results are cached
- Document metadata is cached
- Cache invalidation is automatic

### Query Optimization
- BM25 for lexical search
- Vector similarity for semantic search
- Results are combined with weights
- Indexes are optimized

## Testing

### Unit Tests
```bash
pytest tests/test_unit/
```

### Integration Tests
```bash
pytest tests/test_integration/
```

### Performance Tests
```bash
pytest tests/test_integration/test_performance_benchmark.py
```

## Monitoring

### Metrics
- Request latency
- Query performance
- Cache hit rates
- Resource utilization
- Error rates

### Logging
- Structured JSON logs
- Request/response logging
- Error tracking
- Performance metrics

### Dashboards
- System overview
- Search performance
- Document processing
- Cache performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## About Terilios

[Terilios](https://terilios.com) specializes in developing cutting-edge AI and machine learning solutions for enterprise document processing and information retrieval. Our File Upload Embeddings System represents our commitment to pushing the boundaries of what's possible in document intelligence and semantic search technology.
