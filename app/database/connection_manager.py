from typing import Optional, Dict
from contextlib import contextmanager
import time
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
import redis
from redis.connection import ConnectionPool
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages database and Redis connection pools."""
    
    def __init__(self):
        """Initialize connection pools."""
        self._db_engine = None
        self._session_factory = None
        self._redis_pool = None
        self._redis_client = None
        self._health_check_interval = 30  # seconds
        self._last_health_check = 0
        self.setup()
    
    def setup(self):
        """Set up connection pools."""
        self._setup_database()
        self._setup_redis()
    
    def _setup_database(self):
        """Configure database connection pool."""
        self._db_engine = create_engine(
            settings.DATABASE_URL,
            poolclass=QueuePool,
            pool_size=getattr(settings, 'SQLALCHEMY_POOL_SIZE', 5),
            max_overflow=getattr(settings, 'SQLALCHEMY_MAX_OVERFLOW', 10),
            pool_timeout=getattr(settings, 'SQLALCHEMY_POOL_TIMEOUT', 30),
            pool_pre_ping=True  # Enable connection health checks
        )
        
        # Set up connection event listeners
        event.listen(self._db_engine, 'connect', self._on_connect)
        event.listen(self._db_engine, 'checkout', self._on_checkout)
        event.listen(self._db_engine, 'checkin', self._on_checkin)
        
        # Create session factory
        self._session_factory = scoped_session(
            sessionmaker(
                bind=self._db_engine,
                expire_on_commit=False
            )
        )
    
    def _setup_redis(self):
        """Configure Redis connection pool."""
        self._redis_pool = ConnectionPool.from_url(
            settings.CACHE_REDIS_URL,
            max_connections=getattr(settings, 'REDIS_POOL_SIZE', 10),
            socket_timeout=getattr(settings, 'REDIS_POOL_TIMEOUT', 30),
            socket_keepalive=True,
            health_check_interval=self._health_check_interval
        )
        
        self._redis_client = redis.Redis(
            connection_pool=self._redis_pool,
            decode_responses=True
        )
    
    def _on_connect(self, dbapi_connection, connection_record):
        """Handle database connection creation."""
        logger.info("New database connection created")
    
    def _on_checkout(self, dbapi_connection, connection_record, connection_proxy):
        """Handle database connection checkout."""
        # Perform connection health check if needed
        current_time = time.time()
        if current_time - self._last_health_check > self._health_check_interval:
            try:
                # Test connection with simple query
                connection_proxy.scalar("SELECT 1")
                self._last_health_check = current_time
            except Exception as e:
                logger.error(f"Connection health check failed: {str(e)}")
                raise DisconnectionError("Connection is invalid")
    
    def _on_checkin(self, dbapi_connection, connection_record):
        """Handle database connection return to pool."""
        pass
    
    @contextmanager
    def get_db_session(self):
        """
        Get a database session from the pool.
        
        Yields:
            SQLAlchemy session
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_redis_client(self) -> redis.Redis:
        """
        Get Redis client with connection pool.
        
        Returns:
            Redis client instance
        """
        return self._redis_client
    
    async def check_health(self) -> Dict[str, bool]:
        """
        Check health of all connections.
        
        Returns:
            Dictionary with health status
        """
        status = {
            "database": False,
            "redis": False
        }
        
        # Check database
        try:
            with self.get_db_session() as session:
                session.execute("SELECT 1")
                status["database"] = True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
        
        # Check Redis
        try:
            self._redis_client.ping()
            status["redis"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
        
        return status
    
    def close(self):
        """Close all connections."""
        if self._db_engine:
            self._db_engine.dispose()
        
        if self._redis_pool:
            self._redis_pool.disconnect()
    
    async def wait_for_connections(self, timeout: int = 30) -> bool:
        """
        Wait for connections to be available.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if connections are available, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = await self.check_health()
            if all(status.values()):
                return True
            time.sleep(1)
        return False
    
    @property
    def pool_status(self) -> Dict:
        """
        Get current pool status.
        
        Returns:
            Dictionary with pool statistics
        """
        return {
            "database": {
                "size": self._db_engine.pool.size(),
                "checkedin": self._db_engine.pool.checkedin(),
                "overflow": self._db_engine.pool.overflow(),
                "checkedout": self._db_engine.pool.checkedout()
            },
            "redis": {
                "size": self._redis_pool.max_connections,
                "active": len(self._redis_pool._in_use_connections),
                "available": len(self._redis_pool._available_connections)
            }
        }

# Global connection manager instance
connection_manager = ConnectionManager()

def get_db_session():
    """Dependency for FastAPI to get database session."""
    with connection_manager.get_db_session() as session:
        yield session

def get_redis_client():
    """Dependency for FastAPI to get Redis client."""
    return connection_manager.get_redis_client()
