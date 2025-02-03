# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script into the container
COPY train_model.py .

# Expose port 8080 (optional if you're running a server)
EXPOSE 8080

# Run the script when the container starts
CMD ["python", "train_model.py"]
