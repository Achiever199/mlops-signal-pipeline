FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and data
COPY run.py .
COPY config.yaml .
COPY data.csv .

# Run the pipeline
CMD ["python", "run.py"]
