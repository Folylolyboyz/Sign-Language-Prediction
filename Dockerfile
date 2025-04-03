FROM python:3.8.10-slim as backend

# Set the working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx curl

# Install Python dependencies
RUN pip install -r requirements.txt

CMD sh -c "cd /app && python Inference/inference_api.py"
