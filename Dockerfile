FROM python:3.8.10-slim as backend

# Set the working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install Python dependencies
RUN pip install -r requirements_server.txt

CMD sh -c "cd /app && python Inference/inference_api.py"
