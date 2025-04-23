# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal, as Ollama API doesn't require heavy libraries)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY config.json .
COPY scaler.joblib .
COPY .env .
COPY .streamlit/config.toml .streamlit/config.toml

RUN chmod -R 644 .streamlit/*

# Expose port for Streamlit
EXPOSE 8501

# Set environment variable to ensure Streamlit runs on host 0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]