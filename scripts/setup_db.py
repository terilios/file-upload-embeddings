import sys
from pathlib import Path
import sqlalchemy as sa
from sqlalchemy import create_engine, text

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config.settings import settings
from app.database.models import Base

def setup_database():
    """Set up the database with required extensions and tables."""
    # Create engine without database name to connect to PostgreSQL server
    engine_url = sa.engine.URL.create(
        drivername="postgresql",
        username=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        host=settings.POSTGRES_SERVER,
        port=settings.POSTGRES_PORT
    )
    
    engine = create_engine(str(engine_url))
    
    # Create database if it doesn't exist
    with engine.connect() as conn:
        conn.execute(text("commit"))  # Close any existing transaction
        
        # Check if database exists
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :database"),
            {"database": settings.POSTGRES_DB}
        )
        
        if not result.scalar():
            conn.execute(text("commit"))
            conn.execute(text(f'CREATE DATABASE "{settings.POSTGRES_DB}"'))
    
    # Connect to the created database
    engine = create_engine(settings.DATABASE_URL)
    
    with engine.connect() as conn:
        # Create pgvector extension if it doesn't exist
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        # Create index on embeddings for similarity search
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
            ON document_chunks 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """))
        
        conn.execute(text("commit"))

def reset_database():
    """Reset the database by dropping and recreating it."""
    # Create engine without database name
    engine_url = sa.engine.URL.create(
        drivername="postgresql",
        username=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        host=settings.POSTGRES_SERVER,
        port=settings.POSTGRES_PORT
    )
    
    engine = create_engine(str(engine_url))
    
    with engine.connect() as conn:
        conn.execute(text("commit"))
        
        # Terminate all connections to the database
        conn.execute(text(f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{settings.POSTGRES_DB}'
            AND pid <> pg_backend_pid();
        """))
        
        # Drop database if it exists
        conn.execute(text("commit"))
        conn.execute(text(f'DROP DATABASE IF EXISTS "{settings.POSTGRES_DB}"'))
    
    # Create fresh database
    setup_database()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database setup script")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database (WARNING: This will delete all data)"
    )
    
    args = parser.parse_args()
    
    if args.reset:
        print("Resetting database...")
        reset_database()
        print("Database reset complete.")
    else:
        print("Setting up database...")
        setup_database()
        print("Database setup complete.")
