# 1. Start from a Debian-slim uv image (pre-installed uv binaries)
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy only lockfile and pyproject to install deps first
COPY pyproject.toml uv.lock ./

# 4. Sync dependencies into a fresh .venv using your lockfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# 5. Copy the rest of your source
COPY src/ ./src

# 6. (Optional) Re-sync just the project itself for editable installs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

RUN ls -1 /app/.venv/bin

# 7. Put the virtual-env binaries on the PATH
ENV PATH="/app/.venv/bin:$PATH"

# 8. Start your FastAPI app — uv will activate .venv and run uvicorn
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
