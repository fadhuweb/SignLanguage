# Use slim Python base image
FROM python:3.10-slim

# Avoid cache bloat
ENV TRANSFORMERS_CACHE=/tmp
ENV TORCH_HOME=/tmp

WORKDIR /app

# Copy and install only what's needed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Run your app using Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
