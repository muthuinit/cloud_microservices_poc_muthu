# Use the official Scikit-learn image from Vertex AI
FROM us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-0:latest

# Copy the training script and requirements
COPY train_model.py /train_model.py
COPY requirements.txt /requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt

# Set the entrypoint
ENTRYPOINT ["python", "/train_model.py"]