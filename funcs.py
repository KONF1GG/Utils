import torch
from unidecode import unidecode
from transformers import AutoTokenizer, AutoModel
import hashlib
import re
from torch import Tensor

model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


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

def generate_embedding(texts):
    batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict = {key: value.to(device) for key, value in batch_dict.items()}
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    # print(embeddings)

    embeddings = embeddings.detach().cpu().numpy()
    # embeddings = embeddings.astype(np.float32)
    return embeddings

def normalize_text(text):
    normalized_text = unidecode(text)
    return normalized_text.strip()
