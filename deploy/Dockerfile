FROM python:3.10-slim

WORKDIR /app

# Copy the requirements.txt first to take advantage of Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files (app.py, model.pkl)
COPY . .

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Start the application
CMD ["python", "app.py"]
