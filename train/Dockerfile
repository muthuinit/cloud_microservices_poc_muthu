# Use the official Scikit-learn image from Vertex AI
FROM gcr.io/deeplearning-platform-release/sklearn-cpu.2-6:latest

# Copy the training script and requirements
COPY train_model.py /train_model.py
COPY requirements.txt /requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt

# Set the entrypoint
ENTRYPOINT ["python", "/train_model.py"]