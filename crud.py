import json
import logging
import asyncio
import stat
import re

from torch import embedding
import config
from database import Milvus, MySQL, PostgreSQL
import funcs
from pyschemas import Page, Search2ResponseData, SearchResponseData
from fastapi import HTTPException
from milvus_schemas import address_schema, address_index_params, address_search_params, promt_schema, promt_index_params, promt_search_params, wiki_index_params, wiki_schema, wiki_search_params
from database import Milvus
import config
import redis.asyncio as redis
from tqdm import tqdm

from pyschemas import PromtModel 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

MAX_CONCURRENT_REQUESTS = 5  
BATCH_SIZE = 1000  

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

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
    r = redis.from_url(
        f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
        password=config.REDIS_PASSWORD,
        decode_responses=True
    )

    cursor = 0
    keys = set()

    with tqdm(desc="Получение ключей из Redis", unit=" ключей") as pbar:
        while True:
            cursor, partial_keys = await r.scan(cursor, match=pattern, count=count)
            pbar.update(len(partial_keys))
            
            keys.update(partial_keys)
            if cursor == 0:
                break

    await r.aclose()

    return keys

async def insert_addresses_to_milvus(data, milvus_db: Milvus, batch_size=10000):
    """
    Вставляет данные в Milvus пакетами.
    
    :param data: Список данных для вставки.
    :param milvus_db: Объект Milvus.
    :param batch_size: Размер пакета (по умолчанию 10 000 элементов).
    """
    total_batches = (len(data) + batch_size - 1) // batch_size 

    with tqdm(total=total_batches, desc="Вставка данных в Milvus", unit=" пакетов") as pbar:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            logger.info(f"Вставляю пакет {i // batch_size + 1} из {total_batches}")
            try:
                formatted_data = []
                for entry in batch:
                    data_json = entry[0]
                    address = data_json.get('address') 
                    key = data_json.get('login')
                    house_id = data_json.get('houseId')
                    flat = data_json.get('flat', '')
                    if address == None or house_id == None:
                        continue
                    
                    formatted_data.append({
                        'hash': key,
                        'text': 'passage: ' + address,
                        'house_id': house_id,
                        'flat': flat
                    })

                milvus_db.insert_data(formatted_data, additional_fields=['house_id', 'flat'], batch_size=16)
            except Exception as e:
                logger.error(f"Ошибка при вставке пакета {i // batch_size + 1}: {e}")
                raise
            pbar.update(1)

    milvus_db.create_index()


async def insert_promts_to_milvus(data: list[PromtModel], milvus_db: Milvus):
    formatted_data = []
    print('Formatting data')
    for entry in data:
        hash = entry.get('id')
        name = entry.get('name')
        template = entry.get('template')
        params = entry.get('params')

        formatted_data.append({
            'hash': hash,
            'text': 'passage: ' + template,
            'name': name,
            'params': params
        })
    print('Inserting data')
    milvus_db.insert_data(formatted_data, additional_fields=['name', 'params'], batch_size=1)
    milvus_db.create_index()


async def insert_addresses_from_redis_to_milvus():
    # Получаем уникальные ключи с префиксом
    try:
        unique_keys = list(await get_unique_keys_with_prefix(pattern='login:*', count=10000))
        r = redis.from_url(
            f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
    except Exception as e:
        logger.error(f"Ошибка при получении ключей из Redis: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при получении ключей из Redis")
    

    logger.info("Инициализация соединения с Milvus.")
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Address', address_schema, address_index_params, address_search_params)
    # milvus_db.drop_collection()

    milvus_db.init_collection()
    result = []

    batch_size = 1024
    total_keys = len(unique_keys)

    with tqdm(total=total_keys, desc="Обработка ключей", unit=" ключей") as pbar:
        for i in range(0, total_keys, batch_size):
            batch_keys = unique_keys[i:i + batch_size]  

            values = await r.json().mget(batch_keys, path="$")

            result.extend(values)
            pbar.update(len(batch_keys))  


        await insert_addresses_to_milvus(result, milvus_db, batch_size=10000)



async def insert_promts_from_redis_to_milvus():
    try:
        r = redis.from_url(
            f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
    except Exception as e:
        logger.error(f"Ошибка при подключении к Redis: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при подключении к Redis")

    try:
        result = await r.json().get('scheme:vector')
    except Exception as e:
        logger.error(f"Ошибка при получении схемы из Redis: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при получении схемы")
    logger.info("Инициализация соединения с Milvus.")
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Promts', promt_schema, promt_index_params, promt_search_params)
    milvus_db.init_collection()

    print('Inserting promts to Milvus')
    await insert_promts_to_milvus(result, milvus_db)



def insert_wiki_data():
    try:
        mysql_db = MySQL(**config.mysql_config)
        pages = mysql_db.get_pages_data()
    except Exception as e:
        print(f"Ошибка при подключении к MySQL или получении данных: {e}")
        return
    finally:
        if 'mysql_db' in locals():
            mysql_db.connection_close()

    if not pages:
        print("Не удалось получить данные из MySQL.")
        return

    try:
        postgres_db = PostgreSQL(**config.postgres_config)

        postgres_db.cursor.execute(
            "DELETE FROM frida_storage WHERE isExstra != TRUE;"
        )

        for page in pages:
            page_model = Page(title=page[0], text=page[1].strip(), book_slug=page[2], page_slug=page[3], book_name=page[4] if page[4] else '')
            url = f'http://wiki.freedom1.ru:8080/books/{page_model.book_slug}/page/{page_model.page_slug}'
            page_model.text = re.sub(r'(\r\n)+', '\r\n', page_model.text)
            page_hash = funcs.generate_hash(page_model.text)
            if len(page_model.text) < 20:
                continue


            clean_text_value = funcs.clean_text(page_model.text)

            try:
                postgres_db.cursor.execute(
                    """
                    INSERT INTO frida_storage (hash, book_name, title, text, url)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (hash) DO NOTHING;
                    """,
                    (page_hash, page_model.book_name, page_model.title, clean_text_value, url)
                )

            except Exception as e:
                print(f"Ошибка при вставке данных для страницы {page_model.title}: {e}")
                continue 

        postgres_db.connection.commit()
        return True

    except Exception as e:
        print(f"Ошибка при обработке данных в PostgreSQL: {e}")
        if 'postgres_db' in locals():
            postgres_db.connection.rollback()

    finally:
        if 'postgres_db' in locals():
            postgres_db.connection_close()


def insert_all_data_from_postgres_to_milvus():
    postgres_db = PostgreSQL(**config.postgres_config)
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Frida_bot_data', wiki_schema, wiki_index_params, wiki_search_params)
    milvus_db.init_collection()

    data = postgres_db.get_data_for_vector_db()
    data_list = []
    for topic in data:
        hash = topic[0]
        book_name = topic[1] if topic[1] else ''
        title = topic[2]
        textTitleLess = topic[3]
        text = 'passage: ' + book_name + '\n' + title + ' ' + textTitleLess
        data_list.append({'hash': hash, 'text': text, 'textTitleLess': textTitleLess})
        
    milvus_db.insert_data(data_list)
    milvus_db.create_index()
    duplicates = milvus_db.clean_similar_vectors()
    deleted_count = 0
    if duplicates:
        deleted_count = postgres_db.delete_items_by_hashs(duplicates)
    data_count = postgres_db.get_count()
    postgres_db.connection_close()
    milvus_db.connection_close()
    return data_count, deleted_count

def add_new_topic(title, text, user_id):
    try:
        postgres_db = PostgreSQL(**config.postgres_config)
        text_hash = funcs.generate_hash(text)
        postgres_db.insert_new_topic(text_hash, title, text, user_id)
        milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Frida_bot_data')
        milvus_db.insert_data([{'hash': text_hash, 'text': title+text, 'textTitleLess': text}])
        milvus_db.collection.flush()
        milvus_db.collection.load()
        postgres_db.connection_close()
        milvus_db.connection_close()
        return True
    except Exception as e:
        return e

async def upload_data_wiki_data_to_milvus():
    insert_wiki_data()
    insert_all_data_from_postgres_to_milvus()


# if __name__ == "__main__":
#     asyncio.run(insert_promts_from_redis_to_milvus())

async def search_milvus_and_prep_data(text, user_id) -> SearchResponseData:
    postgres_db = PostgreSQL(**config.postgres_config)
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Frida_bot_data', wiki_schema, wiki_index_params, wiki_search_params)
    milvus_db.collection.load()
    try:
        milvus_response = milvus_db.search(text)
        milvus_db.connection_close()
        hashs = []
        for result in milvus_response:
            for item in result:
                hash_value = item.id
                distance_value = item.distance 
                hashs.append(hash_value)        
                print(f"ID: {hash_value}, Distance: {distance_value}")

        contexts = postgres_db.get_topics_texts_by_hashs(tuple(hashs))
        result_string = "История вашего диалога: "
        message_hostory = postgres_db.get_history(user_id)
        for i, msg in enumerate(message_hostory, 1):
            query = msg[2]
            response = msg[3]

            result_string += f"{i}) Зпрос пользователя: {query} | Твой ответ: {response} "
        combined_context = ""

        for i, (book_name, text, url) in enumerate(contexts, start=1):
            book_name = book_name if book_name else ''
            combined_context += f" Контекст {i}: {book_name + ' ' + text}  URL: {url}"\
        
        return SearchResponseData(combined_context=combined_context, chat_history=result_string, hashs=hashs)
    
    finally:
        milvus_db.data_release()
        milvus_db.connection_close()
        

async def search_milvus(text) -> Search2ResponseData:
    postgres_db = PostgreSQL(**config.postgres_config)
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Frida_bot_data', wiki_schema, wiki_index_params, wiki_search_params)
    milvus_db.collection.load()

    try:
        milvus_response = milvus_db.search(text)
        milvus_db.connection_close()
        hashs = []
        for result in milvus_response:
            for item in result:
                hash_value = item.id
                distance_value = item.distance 
                hashs.append(hash_value)        
                # print(f"ID: {hash_value}, Distance: {distance_value}")

        contexts = postgres_db.get_topics_texts_by_hashs(tuple(hashs))

        combined_context = ''
        for i, (book_name, text, url) in enumerate(contexts, start=1):
            book_name = book_name if book_name else ''
            combined_context += f" Контекст {i}: {book_name + ' ' + text}  URL: {url}"\
            
        return Search2ResponseData(combined_context=combined_context, hashs=hashs)

    finally:
        milvus_db.data_release()
        milvus_db.connection_close()
