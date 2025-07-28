# download_assets.py
from sentence_transformers import SentenceTransformer
import nltk
import os

# 1. Download and save the Sentence Transformer model
model_name = 'all-MiniLM-L6-v2'
model_save_path = os.path.join('models', model_name)

print(f"Downloading model '{model_name}' to '{model_save_path}'...")
model = SentenceTransformer(model_name)
model.save(model_save_path)
print("Model download complete.")

# 2. Download and save the NLTK 'punkt' tokenizer
nltk_data_path = 'nltk_data'
print(f"Downloading NLTK 'punkt' data to '{nltk_data_path}'...")
nltk.download('punkt', download_dir=nltk_data_path)
print("NLTK data download complete.")
print("\n✅ All assets downloaded successfully.")