import json
import re
import time
from datetime import datetime
from typing import List, Dict
import argparse
import os

import nltk
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Model Loading and Configuration (Online Version) ---
# This version will download models from the internet on the first run.
# Ensure you are connected to the internet for the initial run.
# --- Model Loading and Configuration (Offline Version) ---
# This version loads models from local directories.
# Ensure you have already run the 'download_assets.py' script.


# Point to the local NLTK data folder
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.insert(0, nltk_data_path)

# Point to the local Sentence Transformer model folder
model_path = os.path.join(os.getcwd(), 'models', 'all-MiniLM-L6-v2')
model = SentenceTransformer(model_path)


# --- Core Functions (No changes needed here) ---

def extract_sections(pdf_path: str) -> List[Dict]:
    """
    Extracts sections from a PDF using a more robust regex for titles.
    This version processes the entire document text at once to correctly
    handle sections that span across page breaks.
    """
    sections = []
    
    # A more robust regex to find different title formats.
    # It looks for:
    # 1. Numbered headings (e.g., "1. Introduction", "2.1. Results")
    # 2. Words in Title Case (e.g., "A Guide to the Cities")
    # 3. Lines in ALL CAPS (e.g., "NIGHTLIFE AND ENTERTAINMENT")
    title_regex = r'^(?P<title>(\d+[\.\d\s]*[A-Z].*)|([A-Z][a-z]+\s?){2,}|([A-Z\s-]{5,}))\n'

    with pdfplumber.open(pdf_path) as pdf:
        # Combine text from all pages into a single string
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        if not full_text:
            return []

        # Find all potential titles in the document
        matches = list(re.finditer(title_regex, full_text, re.MULTILINE))

        for i, match in enumerate(matches):
            title = match.group('title').strip()

            # Filter out titles that are too long (likely full sentences)
            if len(title.split()) > 10:
                continue

            start_of_content = match.end()
            # The section ends where the next title begins
            end_of_content = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
            
            section_text = full_text[start_of_content:end_of_content].strip()

            # Filter out empty or very short sections
            if not section_text or len(section_text) < 100:
                continue
                
            # Determine the page number of the section title
            page_number = 1
            char_count = 0
            for page_num, page in enumerate(pdf.pages, 1):
                page_text_len = len(page.extract_text() or "") + 1 # +1 for newline
                if char_count + page_text_len >= match.start():
                    page_number = page_num
                    break
                char_count += page_text_len

            sections.append({
                'page': page_number,
                'title': title,
                'text': section_text
            })
            
    return sections

def rank_sections(sections: List[Dict], persona: str, job: str):
    context = persona + ". " + job
    context_emb = model.encode([context])
    for section in sections:
        section_emb = model.encode([section['title'] + " " + section['text']])
        section['score'] = float(cosine_similarity(context_emb, section_emb)[0][0])
    return sorted(sections, key=lambda x: x['score'], reverse=True)

def extract_subsections(section, context_emb):
    subsections = []
    paras = section['text'].split('\n\n')
    for para in paras:
        if len(para.strip()) < 50: continue
        emb = model.encode([para])
        score = float(cosine_similarity(context_emb, emb)[0][0])
        subsections.append({
            'document': section['document'],
            'page': section['page'],
            'refined_text': para.strip(),
            'score': score,
        })
    return sorted(subsections, key=lambda x: x['score'], reverse=True)[:3]

def process_documents(pdf_paths, persona, job):
    document_scores = {}
    all_sections_by_doc = {}
    
    print("-> Step 1: Processing and scoring sections within each document...")
    for pdf_path in pdf_paths:
        try:
            doc_name = os.path.basename(pdf_path)
            sections = extract_sections(pdf_path)
            if not sections:
                continue

            # **FIX IS HERE**: Add the document name to each section
            for section in sections:
                section['document'] = doc_name
            # **END OF FIX**

            ranked_doc_sections = rank_sections(sections, persona, job)
            all_sections_by_doc[doc_name] = ranked_doc_sections
            # Use a fallback score of 0 if a document has no high-scoring sections
            document_scores[doc_name] = np.mean([s.get('score', 0) for s in ranked_doc_sections[:5]])

        except Exception as e:
            print(f"Warning: Could not process {pdf_path}. Error: {e}")

    sorted_documents = sorted(document_scores.items(), key=lambda item: item[1], reverse=True)

    print("\n-> Step 2: Identifying top sections from the most relevant documents...")
    top_sections = []
    
    if len(sorted_documents) > 0:
        top_doc_name = sorted_documents[0][0]
        top_sections.extend(all_sections_by_doc.get(top_doc_name, [])[:2])
        
    if len(sorted_documents) > 1:
        second_doc_name = sorted_documents[1][0]
        top_sections.extend(all_sections_by_doc.get(second_doc_name, [])[:2])

    if len(sorted_documents) > 2:
        third_doc_name = sorted_documents[2][0]
        top_sections.extend(all_sections_by_doc.get(third_doc_name, [])[:1])

    top_sections = sorted(top_sections, key=lambda x: x.get('score', 0), reverse=True)

    context = persona + ". " + job
    context_emb = model.encode([context])
    sub_section_results = []
    for sec in top_sections:
        subs = extract_subsections(sec, context_emb)
        sub_section_results.extend(subs)

    output = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in pdf_paths],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": sec.get('document', 'Unknown'),
                "page_number": sec.get('page', 0),
                "section_title": sec.get('title', 'Untitled'),
                "importance_rank": idx + 1
            } for idx, sec in enumerate(top_sections)
        ],
        "sub_section_analysis": [
            {
                "document": sub.get('document', 'Unknown'),
                "page_number": sub.get('page', 0),
                "refined_text": sub.get('refined_text', '')
            } for sub in sub_section_results
        ]
    }
    return output
# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Document Analyst")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
        exit()

    input_dir = os.path.dirname(args.input_file)
    pdf_folder_name = "PDFs" # Assuming the PDF folder is named "PDFs"
    pdf_folder_path = os.path.join(input_dir, pdf_folder_name)

    list_of_pdfs = [os.path.join(pdf_folder_path, doc['filename']) for doc in input_data['documents']]
    user_persona = input_data['persona']['role']
    job_to_be_done = input_data['job_to_be_done']['task']

    print(f"🚀 Starting document processing for: {input_data['challenge_info']['description']}")
    
    final_output = process_documents(
        pdf_paths=list_of_pdfs,
        persona=user_persona,
        job=job_to_be_done
    )

    output_filename = "challenge1b_output.json"
    output_path = os.path.join(input_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"\n✅ Processing Complete. Results saved to '{output_path}'")