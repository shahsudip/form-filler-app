# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies, including Tesseract OCR and Japanese language pack
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-jpn \
    tesseract-ocr-nep \
    fonts-ipafont-gothic \
    fonts-lohit-deva \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose port (Render sets $PORT dynamically)
EXPOSE 10000

# Run uvicorn when the container launches, dynamically binding to $PORT (or 8000)
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
