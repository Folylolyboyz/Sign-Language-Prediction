# Base image for Python backend
FROM python:3.8.10-slim as backend

# Set the working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx curl

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js for frontend
RUN curl -sL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs

# Install frontend dependencies
WORKDIR /app/Frontend
RUN npm install

# Expose ports
EXPOSE 5173

# Run backend and build frontend together
CMD sh -c "cd /app/Frontend && npm run dev & cd /app && python Inference/inference_api.py"
