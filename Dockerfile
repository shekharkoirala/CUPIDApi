FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

COPY uv.lock .
COPY .python-version .
COPY pyproject.toml .

# RUN uv pip install --no-cache --system .
RUN uv venv && uv pip install --python .venv/bin/python --system uv && .venv/bin/uv sync


COPY . .


# Set venv python as default
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv pip install pip
RUN uv run uv run python -m spacy download en_core_web_sm

# Create directories with proper permissions
RUN mkdir -p /app/data /app/mlreports /app/reports && \
    chmod -R 777 /app/data /app/mlreports /app/reports

EXPOSE 8000

# CMD ["/bin/bash", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port 8000"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
