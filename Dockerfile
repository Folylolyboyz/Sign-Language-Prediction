FROM python:3.8.10-slim as backend

# Set the working directory
WORKDIR /app

COPY requirements_server.txt .
RUN pip install -r requirements_server.txt

# Copy the entire project
WORKDIR /app
COPY . .

# Install Python dependencies

CMD sh -c "cd /app && python Inference/inference_NN_api.py"
