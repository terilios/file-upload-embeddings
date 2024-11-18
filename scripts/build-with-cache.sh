#!/bin/bash

# Enable BuildKit for better caching
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Create cache directories if they don't exist
mkdir -p /tmp/.buildx-cache
mkdir -p /tmp/.buildx-cache-new

# Pull existing images to use cache
echo "Pulling existing images for cache..."
docker-compose pull

# Build with cache
echo "Building services with cache..."
docker-compose build \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain

# Move cache
echo "Moving cache..."
rm -rf /tmp/.buildx-cache
mv /tmp/.buildx-cache-new /tmp/.buildx-cache

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
timeout 300 bash -c 'until docker-compose ps | grep -q "(healthy)"; do echo "Waiting for services to be healthy..."; sleep 5; done'

echo "Services are up and running!"

# Show service status
docker-compose ps
