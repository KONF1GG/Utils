"""
Модуль config.py предназначен для загрузки конфигурационных переменных окружения,
используемых для подключения к различным сервисам (Milvus, Redis, MySQL, PostgreSQL).
Он считывает значения из файла .env и предоставляет их в виде переменных.
"""

import os

from dotenv import dotenv_values

dotenv_values()

TOKEN = os.getenv('TOKEN')
API_KEY=os.getenv('API_KEY')

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

REDIS_HOST= os.getenv('REDIS_HOST')
REDIS_PORT= os.getenv('REDIS_PORT')
REDIS_PASSWORD= os.getenv('REDIS_PASSWORD')
REDIS_LOGIN= os.getenv('REDIS_LOGIN')

HOST_MYSQL= os.getenv('HOST_MYSQL')
PORT_MYSQL= os.getenv('PORT_MYSQL')
USER_MYSQL= os.getenv('USER_MYSQL')
PASSWORD_MYSQL= os.getenv('PASSWORD_MYSQL')
DB_MYSQL= os.getenv('DB_MYSQL')

POSTGRES_USER=os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD=os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB=os.getenv('POSTGRES_DB')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')

API_KEY = os.getenv('API_KEY')

mysql_config = {
    'host': HOST_MYSQL,
    'port': PORT_MYSQL,
    'user': USER_MYSQL,
    'password': PASSWORD_MYSQL,
    'database': DB_MYSQL
}


postgres_config = {
    'host': POSTGRES_HOST,
    'port': POSTGRES_PORT,
    'user': POSTGRES_USER,
    'password': POSTGRES_PASSWORD,
    'database': POSTGRES_DB
}
