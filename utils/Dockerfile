# Base Node.js image
FROM node:20-slim AS node_base

# Set working directory
WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment and install Python dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci

# Copy Python requirements
COPY pyproject.toml ./

# Install Python dependencies
COPY . .
RUN pip3 install -e .

# Build the application
RUN npm run build

# Expose ports
EXPOSE 5000 8000 8100

# Start the application
CMD ["npm", "start"]