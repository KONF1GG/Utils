import os
from dotenv import dotenv_values

dotenv_values()

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

REDIS_HOST= os.getenv('REDIS_HOST')
REDIS_PORT= os.getenv('REDIS_PORT')
REDIS_PASSWORD= os.getenv('REDIS_PASSWORD')
REDIS_LOGIN= os.getenv('REDIS_LOGIN')