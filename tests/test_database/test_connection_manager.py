import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import redis
from sqlalchemy.exc import SQLAlchemyError

from app.database.connection_manager import ConnectionManager, connection_manager
from config.settings import settings

@pytest.fixture
async def test_connection_manager():
    """Create a test connection manager instance."""
    manager = ConnectionManager()
    yield manager
    manager.close()

@pytest.mark.asyncio
async def test_database_connection_pool(test_connection_manager):
    """Test database connection pooling."""
    # Test getting multiple sessions
    sessions = []
    for _ in range(5):
        with test_connection_manager.get_db_session() as session:
            sessions.append(session)
            # Verify session works
            result = session.execute("SELECT 1").scalar()
            assert result == 1
    
    # Check pool status
    status = test_connection_manager.pool_status
    assert "database" in status
    assert status["database"]["size"] <= settings.SQLALCHEMY_POOL_SIZE

@pytest.mark.asyncio
async def test_redis_connection_pool(test_connection_manager):
    """Test Redis connection pooling."""
    # Get Redis client
    redis_client = test_connection_manager.get_redis_client()
    assert isinstance(redis_client, redis.Redis)
    
    # Test Redis operations
    await redis_client.set("test_key", "test_value")
    value = await redis_client.get("test_key")
    assert value == "test_value"
    
    # Check pool status
    status = test_connection_manager.pool_status
    assert "redis" in status
    assert status["redis"]["size"] == settings.REDIS_POOL_SIZE

@pytest.mark.asyncio
async def test_health_check(test_connection_manager):
    """Test connection health checks."""
    # Initial health check
    status = await test_connection_manager.check_health()
    assert status["database"] is True
    assert status["redis"] is True
    
    # Test health check after operations
    with test_connection_manager.get_db_session() as session:
        session.execute("SELECT 1")
    
    redis_client = test_connection_manager.get_redis_client()
    await redis_client.ping()
    
    status = await test_connection_manager.check_health()
    assert status["database"] is True
    assert status["redis"] is True

@pytest.mark.asyncio
async def test_connection_lifecycle(test_connection_manager):
    """Test connection lifecycle management."""
    # Test session cleanup
    with test_connection_manager.get_db_session() as session:
        session.execute("SELECT 1")
    
    # Verify session is closed
    status = test_connection_manager.pool_status
    assert status["database"]["checkedout"] == 0
    
    # Test Redis connection cleanup
    redis_client = test_connection_manager.get_redis_client()
    await redis_client.ping()
    
    # Close all connections
    test_connection_manager.close()
    
    # Verify connections are closed
    status = test_connection_manager.pool_status
    assert status["database"]["checkedout"] == 0
    assert status["redis"]["active"] == 0

@pytest.mark.asyncio
async def test_concurrent_connections(test_connection_manager):
    """Test handling concurrent connections."""
    async def run_db_operation():
        with test_connection_manager.get_db_session() as session:
            session.execute("SELECT 1")
            await asyncio.sleep(0.1)  # Simulate work
    
    # Run multiple concurrent operations
    tasks = [run_db_operation() for _ in range(10)]
    await asyncio.gather(*tasks)
    
    # Verify pool handled concurrent connections
    status = test_connection_manager.pool_status
    assert status["database"]["checkedout"] == 0
    assert status["database"]["size"] <= settings.SQLALCHEMY_POOL_SIZE

@pytest.mark.asyncio
async def test_error_handling(test_connection_manager):
    """Test error handling in connection manager."""
    # Test database error handling
    with pytest.raises(SQLAlchemyError):
        with test_connection_manager.get_db_session() as session:
            session.execute("INVALID SQL")
    
    # Verify session is properly cleaned up
    status = test_connection_manager.pool_status
    assert status["database"]["checkedout"] == 0
    
    # Test Redis error handling
    redis_client = test_connection_manager.get_redis_client()
    with pytest.raises(redis.RedisError):
        await redis_client.execute_command("INVALID")
    
    # Verify Redis connection is still valid
    assert await redis_client.ping()

@pytest.mark.asyncio
async def test_wait_for_connections(test_connection_manager):
    """Test waiting for connections to be available."""
    # Test successful wait
    success = await test_connection_manager.wait_for_connections(timeout=5)
    assert success
    
    # Test timeout
    test_connection_manager.close()
    success = await test_connection_manager.wait_for_connections(timeout=1)
    assert not success

@pytest.mark.asyncio
async def test_connection_reuse(test_connection_manager):
    """Test connection reuse in pool."""
    # Get initial pool status
    initial_status = test_connection_manager.pool_status
    
    # Run multiple operations
    for _ in range(5):
        with test_connection_manager.get_db_session() as session:
            session.execute("SELECT 1")
    
    # Verify connections were reused
    final_status = test_connection_manager.pool_status
    assert final_status["database"]["size"] == initial_status["database"]["size"]

@pytest.mark.asyncio
async def test_pool_overflow(test_connection_manager):
    """Test handling pool overflow."""
    async def run_long_operation():
        with test_connection_manager.get_db_session() as session:
            session.execute("SELECT pg_sleep(0.5)")
    
    # Try to exceed pool size
    tasks = [run_long_operation() for _ in range(settings.SQLALCHEMY_POOL_SIZE + 5)]
    await asyncio.gather(*tasks)
    
    # Verify pool handled overflow
    status = test_connection_manager.pool_status
    assert status["database"]["overflow"] <= settings.SQLALCHEMY_MAX_OVERFLOW

@pytest.mark.asyncio
async def test_global_connection_manager():
    """Test global connection manager instance."""
    assert connection_manager is not None
    
    # Test database connection
    with connection_manager.get_db_session() as session:
        result = session.execute("SELECT 1").scalar()
        assert result == 1
    
    # Test Redis connection
    redis_client = connection_manager.get_redis_client()
    assert await redis_client.ping()
