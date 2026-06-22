FROM python:3.11-slim

WORKDIR /app

# Install system dependencies pdfplumber needs
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first
# (separate layer — cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and index
COPY src/ ./src/
COPY faiss_index/ ./faiss_index/

# Default working directory for running src files
WORKDIR /app/src

# Default command runs the API
# Override in docker-compose for streamlit
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
