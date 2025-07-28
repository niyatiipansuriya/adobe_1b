# 🧠 PDF Heading Extractor & Intelligent Document Analyzer

A robust NLP-powered pipeline to extract, rank, and summarize key sections and subsections from PDFs based on a user's persona and task using sentence embeddings and cosine similarity.

---

## 🚀 Features

- Extracts hierarchical section headings using regex and heuristics.
- Identifies and ranks the most relevant document sections for a persona-task pair.
- Uses `SentenceTransformer` embeddings and cosine similarity for semantic ranking.
- Summarizes top sections into refined subsections.
- Fully containerized using Docker.
- Offline-compatible with local model and NLTK data.

---

## 🧠 Approach

### 1. **PDF Text Extraction**:
   - Uses `pdfplumber` to extract full-text content from all pages.
   - Regex detects heading structures like:
     - Numbered sections (e.g., `1. Introduction`)
     - Title Case headings (e.g., `Getting Started`)
     - ALL CAPS headings (e.g., `BACKGROUND AND OBJECTIVES`)

### 2. **Section Identification**:
   - Groups paragraphs under each heading.
   - Discards overly short or noisy segments.

### 3. **Semantic Relevance Scoring**:
   - Concatenates persona + job description to create a context.
   - Embeds sections and compares with the context using cosine similarity.
   - Ranks and filters most relevant sections.

### 4. **Subsection Summarization**:
   - Extracts top paragraphs from the top-3 ranked documents.
   - Uses semantic closeness to the job-persona context for selection.

---

## 📁 Project Structure

adobe__1B/
├── PDFs/ # Place input PDFs here
├── src/
│ ├── models/ # SentenceTransformer model
│ ├── nltk_data/ # Offline NLTK assets
│ ├── download_assets.py # Script to download models & data
│ └── main.py # Main script to run the pipeline
├── input.json # Input config with persona & job
├── requirements.txt # Python dependencies
├── Dockerfile # Docker container definition
├── .dockerignore
└── README.md # This file

yaml
Copy
Edit

---

## 📦 Dependencies

Listed in `requirements.txt`:

- `pdfplumber`
- `nltk`
- `numpy`
- `sentence-transformers`
- `scikit-learn`

---

## 🐳 Docker Setup

### 1. **Build the Docker Image**

bash
docker build -t pdf-heading-extractor .
2. Prepare Input
Place all your .pdf files in the PDFs/ folder.

Fill input.json with the format:

json
Copy
Edit
{
  "challenge_info": {
    "description": "Analyze educational documents for hiring"
  },
  "persona": {
    "role": "Education Researcher"
  },
  "job_to_be_done": {
    "task": "Find insights about school innovation"
  },
  "documents": [
    { "filename": "doc1.pdf" },
    { "filename": "doc2.pdf" }
  ]
}
3. Run the Container
bash
Copy
Edit
docker run --rm -v "$(pwd)/PDFs:/app/PDFs" -v "$(pwd)/input.json:/app/input.json" -v "$(pwd)/output:/app/output" pdf-heading-extractor
✅ The output will be saved as challenge1b_output.json in the output/ folder.

🔄 Offline Mode
To avoid redownloading models every time:

Run:

```
Copy
Edit
python src/download_assets.py
This downloads:
```

SentenceTransformer model to src/models/

NLTK data to src/nltk_data/

main.py will load from local folders.

📤 Output Format
json
Copy
Edit
{
  "metadata": {
    "input_documents": [...],
    "persona": "...",
    "job_to_be_done": "...",
    "processing_timestamp": "..."
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "page_number": 3,
      "section_title": "Innovation Strategies",
      "importance_rank": 1
    }
  ],
  "sub_section_analysis": [
    {
      "document": "doc1.pdf",
      "page_number": 3,
      "refined_text": "Subsection content relevant to the task..."
    }
  ]
}
🧪 Sample Run (Without Docker)
```
Copy
Edit
python src/main.py input.json
```
