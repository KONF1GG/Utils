"""
Схемы коллекций и параметры индексов для работы с Milvus.
Определяет структуры данных для адресов, промтов и wiki, а также параметры поиска и индексации.
"""

from pymilvus import DataType, FieldSchema

# -------------------- Address Collection --------------------
address_schema = [
    FieldSchema(name='hash', dtype=DataType.VARCHAR, is_primary=True, max_length=255),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name='house_id', dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name='flat', dtype=DataType.VARCHAR, max_length=20),
]

address_index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 8, "efConstruction": 150}
}

address_search_params = {
    "metric_type": "L2",
    "params": {"ef": 200, "nprobe": 10}
}

# -------------------- Category Collection  --------------------
# category_schema = [
#     FieldSchema(name='hash', dtype=DataType.VARCHAR, is_primary=True, max_length=255),
#     FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=512),
#     FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=255),
# ]
#
# category_index_params = {
#     "index_type": "HNSW",
#     "metric_type": "COSINE",
#     "params": {"M": 16, "efConstruction": 300}
# }
#
# category_search_params = {
#     "metric_type": "COSINE",
#     "params": {"ef": 200, "nprobe": 10}
# }

# -------------------- Prompt Collection --------------------
promt_schema = [
    FieldSchema(name='hash', dtype=DataType.VARCHAR, is_primary=True, max_length=255),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=10000),
    FieldSchema(name='name', dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name='params', dtype=DataType.VARCHAR, max_length=255),
]

promt_index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 16, "efConstruction": 300}
}

promt_search_params = {
    "metric_type": "L2",
    "params": {"ef": 200, "nprobe": 10}
}

# -------------------- Wiki Collection --------------------
wiki_schema = [
    FieldSchema(name='hash', dtype=DataType.VARCHAR, is_primary=True, max_length=255),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=50000),
]

wiki_index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 300}
}

wiki_search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 200, "nprobe": 10}
}
