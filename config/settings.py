import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# API Configuration
API_V1_STR = "/api/v1"
PROJECT_NAME = "File Upload Embeddings"

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 8  # 8 days

# Determine if running in Docker
IN_DOCKER = os.getenv('IN_DOCKER', 'false').lower() == 'true'

# Database configuration
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "postgres")  # Use container name by default
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "file_upload_embeddings")

# Database URLs
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Document Processing Settings
CHUNK_SIZE_MAPPING: Dict[str, int] = {
    "email": 300,
    "report": 500,
    "technical": 800,
    "default": 500
}

CHUNK_OVERLAP_MAPPING: Dict[str, int] = {
    "email": 50,
    "report": 100,
    "technical": 150,
    "default": 100
}

# Vector Store Settings
VECTOR_DIMENSION = 1536  # OpenAI embedding dimension
SIMILARITY_THRESHOLD = 0.5  # Lowered from 0.7 for better recall

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Frontend Configuration
STREAMLIT_THEME = {
    "primaryColor": "#FF4B4B",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F0F2F6",
    "textColor": "#262730",
    "font": "sans serif"
}

# Redis Cache Settings
CACHE_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_DEFAULT_TIMEOUT = int(os.getenv("CACHE_DEFAULT_TIMEOUT", 3600))  # 1 hour
REDIS_POOL_SIZE = int(os.getenv("REDIS_POOL_SIZE", 10))
REDIS_POOL_TIMEOUT = int(os.getenv("REDIS_POOL_TIMEOUT", 20))

# Cache Configuration
CACHE_TYPE = "redis"

# SQLAlchemy Configuration
SQLALCHEMY_POOL_SIZE = int(os.getenv("SQLALCHEMY_POOL_SIZE", "5"))
SQLALCHEMY_MAX_OVERFLOW = int(os.getenv("SQLALCHEMY_MAX_OVERFLOW", "10"))
SQLALCHEMY_POOL_TIMEOUT = int(os.getenv("SQLALCHEMY_POOL_TIMEOUT", "30"))

# API Rate Limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hour

# File Upload Settings
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {
    "txt", "pdf", "doc", "docx", 
    "csv", "xls", "xlsx", "json",
    "md", "rst"
}

# RAG Configuration
DEFAULT_TEMPERATURE = 0.7
MAX_TOKENS = 2000
TOP_K_RESULTS = 5

class Settings:
    PROJECT_NAME: str = PROJECT_NAME
    API_V1_STR: str = API_V1_STR
    SECRET_KEY: str = SECRET_KEY
    ACCESS_TOKEN_EXPIRE_MINUTES: int = ACCESS_TOKEN_EXPIRE_MINUTES
    DEBUG: bool = DEBUG
    IN_DOCKER: bool = IN_DOCKER
    DATABASE_URL: str = DATABASE_URL
    POSTGRES_USER: str = POSTGRES_USER
    POSTGRES_PASSWORD: str = POSTGRES_PASSWORD
    POSTGRES_SERVER: str = POSTGRES_SERVER
    POSTGRES_PORT: str = POSTGRES_PORT
    POSTGRES_DB: str = POSTGRES_DB
    OPENAI_API_KEY: Optional[str] = OPENAI_API_KEY
    AZURE_OPENAI_API_KEY: Optional[str] = AZURE_OPENAI_API_KEY
    AZURE_OPENAI_API_BASE: Optional[str] = AZURE_OPENAI_API_BASE
    AZURE_OPENAI_API_VERSION: Optional[str] = AZURE_OPENAI_API_VERSION
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = AZURE_OPENAI_DEPLOYMENT_NAME
    MAX_CONTENT_LENGTH: int = MAX_CONTENT_LENGTH
    ALLOWED_EXTENSIONS: set = ALLOWED_EXTENSIONS
    VECTOR_DIMENSION: int = VECTOR_DIMENSION
    SIMILARITY_THRESHOLD: float = SIMILARITY_THRESHOLD
    CHUNK_SIZE_MAPPING: Dict[str, int] = CHUNK_SIZE_MAPPING
    CHUNK_OVERLAP_MAPPING: Dict[str, int] = CHUNK_OVERLAP_MAPPING
    DEFAULT_TEMPERATURE: float = DEFAULT_TEMPERATURE
    MAX_TOKENS: int = MAX_TOKENS
    TOP_K_RESULTS: int = TOP_K_RESULTS
    SQLALCHEMY_POOL_SIZE: int = SQLALCHEMY_POOL_SIZE
    SQLALCHEMY_MAX_OVERFLOW: int = SQLALCHEMY_MAX_OVERFLOW
    SQLALCHEMY_POOL_TIMEOUT: int = SQLALCHEMY_POOL_TIMEOUT
    CACHE_TYPE: str = CACHE_TYPE
    CACHE_REDIS_URL: str = CACHE_REDIS_URL
    CACHE_DEFAULT_TIMEOUT: int = CACHE_DEFAULT_TIMEOUT
    REDIS_POOL_SIZE: int = REDIS_POOL_SIZE
    REDIS_POOL_TIMEOUT: int = REDIS_POOL_TIMEOUT

settings = Settings()
