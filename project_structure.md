file-upload-embeddings/
├── app/
│ ├── frontend/
│ │ ├── **init**.py
│ │ ├── main.py # Streamlit application entry point
│ │ ├── components/ # Reusable Streamlit components
│ │ │ ├── **init**.py
│ │ │ ├── file_upload.py
│ │ │ ├── chat_interface.py
│ │ │ └── metadata_display.py
│ │ └── utils/
│ │ ├── **init**.py
│ │ └── state_management.py
│ │
│ ├── backend/
│ │ ├── **init**.py
│ │ ├── main.py # FastAPI application entry point
│ │ ├── api/
│ │ │ ├── **init**.py
│ │ │ ├── routes.py
│ │ │ └── endpoints/
│ │ │ ├── **init**.py
│ │ │ ├── document.py
│ │ │ └── chat.py
│ │ └── core/
│ │ ├── **init**.py
│ │ └── config.py
│ │
│ ├── document_processing/
│ │ ├── **init**.py
│ │ ├── chunking.py # Adaptive document chunking logic
│ │ ├── embeddings.py # Multi-vector embedding generation
│ │ └── metadata.py # Metadata extraction and enrichment
│ │
│ ├── rag/
│ │ ├── **init**.py
│ │ ├── retriever.py # Document retrieval strategies
│ │ ├── reranker.py # Result reranking logic
│ │ └── answer_fusion.py # Response generation and fusion
│ │
│ └── database/
│ ├── **init**.py
│ ├── models.py # SQLAlchemy/Pydantic models
│ ├── crud.py # Database operations
│ └── vector_store.py # pgvector operations
│
├── tests/
│ ├── **init**.py
│ ├── test_frontend/
│ ├── test_backend/
│ ├── test_document_processing/
│ └── test_rag/
│
├── scripts/
│ ├── setup_db.py # Database initialization
│ └── generate_test_data.py # Test data generation
│
├── config/
│ ├── **init**.py
│ └── settings.py # Global configuration settings
│
├── .env # Environment variables (existing)
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── docker-compose.yml # Container configuration
