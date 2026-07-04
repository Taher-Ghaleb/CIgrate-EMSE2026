FROM python:3.11-slim

WORKDIR /replication

# System deps for YAML linting (optional) and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: regenerate figures from bundled results
CMD ["python", "scripts/analysis/generate_paper_figures.py"]
