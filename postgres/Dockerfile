FROM postgres:15-bullseye

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    postgresql-server-dev-15 \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install pgrx
RUN cargo install --locked cargo-pgrx --version 0.12.5

# Initialize pgrx
RUN cargo pgrx init --pg15 pg_config

# Install pgvector
RUN cd /tmp \
    && git clone https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install

# Install pgvectorscale
RUN cd /tmp \
    && git clone https://github.com/timescale/pgvectorscale.git \
    && cd pgvectorscale/pgvectorscale \
    && cargo pgrx install --release

# Cleanup
RUN apt-get purge -y --auto-remove \
    build-essential \
    curl \
    git \
    pkg-config \
    postgresql-server-dev-15 \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /root/.cargo \
    && rm -rf /root/.rustup
