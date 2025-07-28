# 1. Use lightweight Python
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# 4. Install system dependencies for pdfplumber
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your entire project (models + nltk_data + code)
COPY src ./src

# 7. Set NLTK data environment path
ENV NLTK_DATA=/app/src/nltk_data

# 8. Set default command
CMD ["python", "src/main.py", "/app/input/input.json"]
