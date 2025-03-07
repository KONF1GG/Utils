import json
import logging
import asyncio
from milvus_schemas import address_schema, address_index_params, address_search_params
from database import Milvus
import config
import redis.asyncio as redis
from tqdm.asyncio import tqdm

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

MAX_CONCURRENT_REQUESTS = 5  
BATCH_SIZE = 1000  

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def get_batch_from_redis(r, cursor, batch_size=10000):
    """Получение порции данных из Redis."""
    cursor, keys = await r.scan(cursor, match="login:*", count=batch_size)
    result = []

    for key in keys:
        value = await r.json().get(key)
        if value is not None:
            try:
                value = json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError:
                pass  
            result.append({"key": key, "value": value})

    return cursor, result

async def insert_addresses_to_milvus(batch_data, milvus_db: Milvus):
    """Вставка порции данных в Milvus."""
    print(milvus_db.get_data_count())

    formatted_data = []
    for entry in batch_data:
        address = entry.get('value').get('address') 
        key = entry.get('key')
        house_id = entry.get('value').get('houseId')
        if address == None or house_id == None:
            continue
        
        formatted_data.append({
            'hash': key,
            'text': address,
            'house_id': house_id
        })

    milvus_db.insert_data(formatted_data, additional_field='house_id')
    logger.info(f"{len(batch_data)} адресов успешно загружены в Milvus.")

async def process_redis_to_milvus():
    """Асинхронная обработка данных из Redis и вставка в Milvus порциями."""
    r = redis.from_url(
        f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
        password=config.REDIS_PASSWORD,
        decode_responses=True
    )
    logger.info("Инициализация соединения с Milvus.")
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Address', address_schema, address_index_params, address_search_params)
    milvus_db.drop_collection()

    milvus_db.init_collection()

    cursor = 0 
    tasks = []  

    while True:
        cursor, batch_data = await get_batch_from_redis(r, cursor, batch_size=10000)
        
        if not batch_data:
            break 
        
        task = asyncio.create_task(insert_addresses_to_milvus(batch_data, milvus_db))
        tasks.append(task)


        if len(tasks) >= MAX_CONCURRENT_REQUESTS:
            await asyncio.gather(*tasks) 
            tasks = [] 

    if tasks:
        await asyncio.gather(*tasks)

    milvus_db.connection_close()

async def main():
    await process_redis_to_milvus()

# asyncio.run(main())

async def get_unique_keys_with_prefix(host='localhost', port=6379, db=0, pattern='login:*', count=10000):
    """
    Асинхронная функция для получения всех уникальных ключей с заданным префиксом.

    :param host: Хост Redis (по умолчанию 'localhost').
    :param port: Порт Redis (по умолчанию 6379).
    :param db: Номер базы данных Redis (по умолчанию 0).
    :param pattern: Шаблон для поиска ключей (по умолчанию 'login:*').
    :param count: Количество ключей за одну итерацию (по умолчанию 10000).
    :return: Множество уникальных ключей.
    """
    # Подключение к Redis
    r = redis.from_url(
        f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
        password=config.REDIS_PASSWORD,
        decode_responses=True
    )

    # Инициализация
    cursor = 0
    keys = set()

    while True:
        cursor, partial_keys = await r.scan(cursor, match=pattern, count=count)
        print(f"Found {len(partial_keys)} keys in this iteration.")
        
        keys.update(partial_keys)
        
        if cursor == 0:
            break

    await r.close()

    return keys

async def main():
    unique_keys = await get_unique_keys_with_prefix(pattern='login:*', count=10000)
    
    print(f"Total unique keys: {len(unique_keys)}")
    print("Keys:")
    # for key in unique_keys:
    #     print(key) 

if __name__ == "__main__":
    asyncio.run(main())

# def insert_categories_to_milvus():
#     logger.info("Инициализация соединения с Milvus.")
#     milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Categories', category_schema, category_index_params, category_search_params)
#     milvus_db.init_collection()
#     categories = [
#         "Отсутствие интернета",
#         "Оплата/Баланс",
#         "Личный кабинет",
#         "Оборудование",
#         "Телевидение",
#         "Тарифы",
#         "Подключение",
#         "Видеонаблюдение",
#         "Домофония",
#         "Сервисный выезд",
#         "Неопределено/Категория неопределена",
#         "Расторжение договора",
#         "Переезд",
#         "Приветствие",
#         "Скорость/Проверка скорости",
#         "Повышение стоимости"
#     ]

#     formatted_data = []
#     for category in categories:
#         formatted_data.append({
#             'hash': funcs.generate_hash(category), 
#             'text': category,
#         })
        
#     milvus_db.insert_data(formatted_data, additional_fields={}) 
#     logger.info("Категории успешно загружены в Milvus.")
#     milvus_db.connection_close()
