# Use Python 3.8 with CUDA support
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports for different AI services
EXPOSE 8081 8082 8083

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ML_QR_SERVICE_PORT=8081
ENV ML_VOICE_TEXT_SERVICE_PORT=8082
ENV ML_VIDEO_SERVICE_PORT=8083

# Start script to run all AI services
COPY start_ai_services.sh .
RUN chmod +x start_ai_services.sh

# Start all AI services
CMD ["./start_ai_services.sh"] 