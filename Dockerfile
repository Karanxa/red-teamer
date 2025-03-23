FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Kubernetes support
RUN pip install --no-cache-dir kubernetes

# Copy the application code
COPY . .

# Install the redteamer package in development mode
RUN pip install -e .

# Create directory for configuration
RUN mkdir -p /etc/redteamer

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Define the entry point
ENTRYPOINT ["redteamer"]
CMD ["--help"] 