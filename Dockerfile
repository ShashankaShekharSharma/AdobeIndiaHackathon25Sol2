FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential && apt-get clean

# Install Python dependencies
RUN pip install pymupdf numpy scikit-learn

# Copy the script into the container
COPY pdf_analyzer.py .

# Set the default command to run the analyzer
CMD ["python", "pdf_analyzer.py"]
