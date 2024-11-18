-- Install required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

-- Optimize PostgreSQL settings for vector operations
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '8GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET max_parallel_workers = '8';
ALTER SYSTEM SET max_parallel_workers_per_gather = '4';
ALTER SYSTEM SET max_parallel_maintenance_workers = '4';
ALTER SYSTEM SET work_mem = '128MB';
ALTER SYSTEM SET random_page_cost = '1.1';

-- Reload configuration
SELECT pg_reload_conf();
