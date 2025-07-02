"""
Модуль содержит набор утилитарных функций для обработки текста,
генерации эмбеддингов, очистки GPU памяти и работы с временными файлами.
Переменные:
- model: загруженная модель 'intfloat/multilingual-e5-large'.
- tokenizer: токенизатор для модели 'intfloat/multilingual-e5-large'.
- device: устройство для выполнения вычислений ('cuda' или 'cpu').
Примечание:
Некоторые функции предполагают использование GPU, если оно доступно.
"""
import os
import gc
import logging
import hashlib
import re
from pathlib import Path
from contextlib import contextmanager

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from unidecode import unidecode

model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')

# model_base_path = "/root/.cache/huggingface/hub/models--intfloat--multilingual-e5-large/snapshots/0dc5580a448e4284468b8909bae50fa925907bc5"
# model = AutoModel.from_pretrained(model_base_path)
# tokenizer = AutoTokenizer.from_pretrained(model_base_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.warning(device)

# model = model.to(device)

@contextmanager
def use_device(local_model, target_device):
    """Контекстный менеджер для временного переноса модели на указанное устройство."""
    original_device = next(local_model.parameters()).device
    if original_device != target_device:
        local_model.to(target_device)
    try:
        yield
    finally:
        if original_device != target_device:
            local_model.to(original_device)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    """Выполняет усреднение скрытых состояний модели с учетом маски внимания."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def extract_all_numbers_and_combinations(text):
    """
    Извлекает все числа и их комбинации с буквами из текста.
    """
    pattern = r'(\d+)([a-zA-Zа-яА-Я])'
    text = re.sub(pattern, r'\1/\2', text)
    combinations = re.findall(r'\d+/\w+|\d+', text)
    return combinations

def clean_text(text):
    """
    Очищает текст от некорректных символов Unicode.
    """
    try:
        return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    except UnicodeDecodeError:
        return text

def generate_hash(text: str) -> str:
    """
    Генерирует SHA-256 хэш для заданного текста.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def generate_embedding(texts):
    """
    Генерирует эмбеддинги для списка текстов.
    """
    batch_dict = tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt')
    batch_dict = {key: value.to(device) for key, value in batch_dict.items()}
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    embeddings = embeddings.detach().cpu().numpy()
    return embeddings

def normalize_text(text):
    """
    Нормализует текст, удаляя акценты и пробелы.
    """
    normalized_text = unidecode(text)
    return normalized_text.strip()

def clear_gpu_memory():
    """Очистка видеопамяти."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def cleanup_temp_dir(temp_dir: Path):
    """Очищает временную директорию от старых файлов"""
    for file in temp_dir.glob("temp_users_*.json"):
        try:
            os.remove(file)
        except OSError:
            pass
