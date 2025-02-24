from unidecode import unidecode
from sentence_transformers import SentenceTransformer
import hashlib
import re

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

def extract_all_numbers_and_combinations(text):
    pattern = r'(\d+)([a-zA-Zа-яА-Я])'
    
    text = re.sub(pattern, r'\1/\2', text)
    
    combinations = re.findall(r'\d+/\w+|\d+', text)

    return combinations

def clean_text(text):
    try:
        return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    except UnicodeDecodeError:
        return text

def generate_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def generate_embedding(text):
    return model.encode(text)

def normalize_text(text):
    normalized_text = unidecode(text)
    return normalized_text.strip()