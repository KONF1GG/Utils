"""
Модуль содержит утилиты для работы с базами данных Redis, Milvus, MySQL и PostgreSQL.
Содержит функции для обработки, вставки, поиска и переноса данных между хранилищами.
"""

import asyncio
import logging
import re

from fastapi import HTTPException
import psycopg2
from pymilvus import SearchFuture, SearchResult
from tqdm import tqdm
import redis.asyncio as redis

import config
import funcs
from database import Milvus, MySQL, PostgreSQL
from milvus_schemas import (
    address_schema, address_index_params, address_search_params,
    promt_schema, promt_index_params, promt_search_params,
    wiki_index_params, wiki_schema, wiki_search_params
)
from pyschemas import Page, PromtModel, Search2ResponseData, SearchResponseData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

MAX_CONCURRENT_REQUESTS = 5
BATCH_SIZE = 1000

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def get_unique_keys_with_prefix(pattern='login:*', count=10000):
    """
    Асинхронная функция для получения всех уникальных ключей с заданным префиксом из Redis.

    :param pattern: Шаблон ключей для поиска.
    :param count: Количество ключей для обработки за один запрос.
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
            logger.info("Вставляю пакет %d из %d", i // batch_size + 1, total_batches)
            try:
                formatted_data = []
                for entry in batch:
                    data_json = entry[0]
                    address = data_json.get('address')
                    key = data_json.get('login')
                    house_id = data_json.get('houseId')
                    flat = data_json.get('flat', '')
                    if address is None or house_id is None:
                        continue
                    formatted_data.append({
                        'hash': key,
                        'text': 'passage: ' + address,
                        'house_id': house_id,
                        'flat': flat
                    })

                milvus_db.insert_data(
                    formatted_data,
                    additional_fields=['house_id', 'flat'],
                    batch_size=16
                )
            except psycopg2.Error as e:
                logger.error("Ошибка при вставке пакета %d: %s", i // batch_size + 1, e)
                raise
            pbar.update(1)

    milvus_db.create_index()


async def insert_promts_to_milvus(data: list[PromtModel], milvus_db: Milvus):
    """
    Вставляет промты в Milvus.

    :param data: Список объектов PromtModel для вставки.
    :param milvus_db: Объект Milvus.
    """
    formatted_data = []
    logger.info('Форматирование данных для вставки')
    for entry in data:
        formatted_data.append({
            'hash': entry.id,
            'text': 'passage: ' + entry.template,
            'name': entry.name,
            'params': entry.params
        })
    logger.info('Вставка данных в Milvus')
    milvus_db.insert_data(formatted_data, additional_fields=['name', 'params'], batch_size=1)
    milvus_db.create_index()


async def insert_addresses_from_redis_to_milvus():
    """
    Извлекает адреса из Redis и вставляет их в Milvus.
    """
    try:
        unique_keys = list(await get_unique_keys_with_prefix(pattern='login:*', count=10000))
        r = redis.from_url(
            f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
    except (TypeError, ValueError) as e:
        logger.error("Ошибка при получении ключей из Redis: %s", e)
        raise HTTPException(status_code=400, detail="Ошибка при получении ключей из Redis") from e
    except Exception as e:
        logger.error("Ошибка при получении ключей из Redis: %s", e)
        raise HTTPException(status_code=500, detail="Ошибка при получении ключей из Redis") from e

    logger.info("Инициализация соединения с Milvus.")
    milvus_db = Milvus(
        config.MILVUS_HOST,
        config.MILVUS_PORT,
        'Address',
        address_schema,
        address_index_params,
        address_search_params
    )

    milvus_db.init_collection()
    result = []

    batch_size = 1024
    total_keys = len(unique_keys)

    with tqdm(total=total_keys, desc="Обработка ключей", unit=" ключей") as pbar:
        for i in range(0, total_keys, batch_size):
            batch_keys = unique_keys[i:i + batch_size]
            values = r.json().mget(batch_keys, path="$")
            result.extend(values)
            pbar.update(len(batch_keys))

        await insert_addresses_to_milvus(result, milvus_db, batch_size=10000)


async def insert_promts_from_redis_to_milvus():
    """
    Извлекает промты из Redis и вставляет их в Milvus.
    """
    try:
        r = redis.from_url(
            f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
    except Exception as e:
        logger.error("Ошибка при подключении к Redis: %s", e)
        raise HTTPException(status_code=500, detail="Ошибка при подключении к Redis") from e

    try:
        result = r.json().get('scheme:vector')
        if result is None:
            result = []
        promt_models = []
        for item in result:
            if isinstance(item, PromtModel):
                promt_models.append(item)
            elif isinstance(item, dict):
                required_keys = {"id", "name", "template", "params"}
                if required_keys.issubset(item.keys()):
                    promt_models.append(PromtModel(
                        id=item["id"],
                        name=item["name"],
                        template=item["template"],
                        params=item["params"]
                    ))
                else:
                    logger.warning("Item missing required keys: %s", item)
            else:
                logger.warning("Unexpected item type: %s - %s", type(item), item)
    except Exception as e:
        logger.error("Ошибка при получении схемы из Redis: %s", e)
        raise HTTPException(status_code=500, detail="Ошибка при получении схемы") from e

    logger.info("Инициализация соединения с Milvus.")
    milvus_db = Milvus(
        config.MILVUS_HOST,
        config.MILVUS_PORT,
        'Promts',
        promt_schema,
        promt_index_params,
        promt_search_params
    )
    milvus_db.init_collection()

    logger.info('Вставка промтов в Milvus')
    await insert_promts_to_milvus(promt_models, milvus_db)


def insert_wiki_data():
    """
    Извлекает данные из MySQL и вставляет их в PostgreSQL.
    """
    try:
        # Подключение к MySQL и получение данных
        mysql_db = MySQL(**config.mysql_config)
        pages = mysql_db.get_pages_data()
    except Exception as e:
        logger.error("Неизвестная ошибка при работе с MySQL: %s", e)
        return False
    finally:
        if 'mysql_db' in locals():
            mysql_db.connection_close()

    if not pages:
        logger.warning("Не удалось получить данные из MySQL.")
        return False

    try:
        # Подключение к PostgreSQL и обработка данных
        postgres_db = PostgreSQL(**config.postgres_config)

        postgres_db.cursor.execute(
            "DELETE FROM frida_storage WHERE isExstra != TRUE;"
        )

        for page in pages:
            try:
                if len(page.get('page_text')) < 20:
                    continue
                page_model = Page(
                    title=page.get('page_name'),
                    text=page.get('page_text').strip(),
                    book_slug=page.get('book_slug'),
                    page_slug=page.get('page_slug'),
                    book_name=page.get('chapter_name')
                )
                url = f'http://wiki.freedom1.ru:8080/books/{page_model.book_slug}/page/{page_model.page_slug}'
                page_model.text = re.sub(r'(\r\n)+', '\r\n', page_model.text)
                page_hash = funcs.generate_hash(page_model.text)


                clean_text_value = funcs.clean_text(page_model.text)

                postgres_db.cursor.execute(
                    """
                    INSERT INTO frida_storage (hash, book_name, title, text, url)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (hash) DO NOTHING;
                    """,
                    (page_hash, page_model.book_name, page_model.title, clean_text_value, url)
                )
            except psycopg2.Error as e:
                logger.error("Ошибка при вставке данных для страницы %s: %s", page_model.title, e)
                postgres_db.connection.rollback()
                continue
            except Exception as e:
                logger.error("Неизвестная ошибка при обработке страницы: %s", e)
                continue

        postgres_db.connection.commit()
        return True

    except psycopg2.Error as e:
        logger.error("Ошибка при обработке данных в PostgreSQL: %s", e)
        if 'postgres_db' in locals():
            postgres_db.connection.rollback()
        return False
    except Exception as e:
        logger.error("Неизвестная ошибка при работе с PostgreSQL: %s", e)
        return False
    finally:
        if 'postgres_db' in locals():
            postgres_db.connection_close()


def insert_all_data_from_postgres_to_milvus():
    """
    Извлекает данные из PostgreSQL и вставляет их в Milvus.
    """
    postgres_db = PostgreSQL(**config.postgres_config)
    milvus_db = Milvus(
        config.MILVUS_HOST,
        config.MILVUS_PORT,
        'Frida_bot_data',
        wiki_schema,
        wiki_index_params,
        wiki_search_params
    )
    milvus_db.init_collection()

    data = postgres_db.get_data_for_vector_db()
    data_list = []
    for topic in data:
        topic_hash = topic[0]
        book_name = topic[1] if topic[1] else ''
        title = topic[2]
        text_title_less = topic[3]
        text = 'passage: ' + book_name + '\n' + title + ' ' + text_title_less
        data_list.append({'hash': topic_hash, 'text': text, 'textTitleLess': text_title_less})

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
    """
    Добавляет новую тему в PostgreSQL и Milvus.

    :param title: Заголовок темы.
    :param text: Текст темы.
    :param user_id: ID пользователя.
    :return: True, если успешно, иначе ошибка.
    """
    try:
        postgres_db = PostgreSQL(**config.postgres_config)
        text_hash = funcs.generate_hash(text)
        postgres_db.insert_new_topic(text_hash, title, text, user_id)
        milvus_db = Milvus(
            config.MILVUS_HOST,
            config.MILVUS_PORT,
            'Frida_bot_data',
            wiki_schema,
            wiki_index_params,
            wiki_search_params
        )
        milvus_db.insert_data([{'hash': text_hash, 'text': title + text, 'textTitleLess': text}])
        milvus_db.collection.flush()
        milvus_db.collection.load()
        postgres_db.connection_close()
        milvus_db.connection_close()
        return True
    except Exception as e:
        return e


async def upload_data_wiki_data_to_milvus():
    """
    Загружает данные из MySQL в PostgreSQL и затем в Milvus.
    """

    logger.info('Выгрузка данных WIKI')
    insert_wiki_data()
    insert_all_data_from_postgres_to_milvus()


async def search_milvus_and_prep_data(text, user_id) -> SearchResponseData:
    """
    Выполняет поиск в Milvus и подготавливает данные для ответа.

    :param text: Текст запроса.
    :param user_id: ID пользователя.
    :return: Объект SearchResponseData.
    """
    postgres_db = PostgreSQL(**config.postgres_config)
    milvus_db = Milvus(
        config.MILVUS_HOST,
        config.MILVUS_PORT,
        'Frida_bot_data',
        wiki_schema,
        wiki_index_params, 
        wiki_search_params)
    milvus_db.collection.load()
    try:
        response = milvus_db.search(text)
        if response is None:
            raise ValueError("Milvus search() вернул None")

        if isinstance(response, SearchFuture):
            response = response.result()

        if not isinstance(response, SearchResult):
            raise TypeError(f"Неподдерживаемый тип ответа от Milvus: {type(response)}")

        hashs = []
        for hits in response:
            for hit in hits:
                hash_val = hit.entity.get("hash")
                if hash_val:
                    hashs.append(hash_val)

        contexts = postgres_db.get_topics_texts_by_hashs(tuple(hashs))
        result_string = "История вашего диалога: "
        message_history = postgres_db.get_history(user_id)
        for i, msg in enumerate(message_history, 1):
            query = msg[2]
            response = msg[3]
            result_string += f"{i}) Запрос пользователя: {query} | Твой ответ: {response} "

        combined_context = ""
        for i, (book_name, text, url) in enumerate(contexts, start=1):
            book_name = book_name if book_name else ''
            combined_context += f" Контекст {i}: {book_name + ' ' + text}  URL: {url}"

        return SearchResponseData(
            combined_context=combined_context,
            chat_history=result_string,
            hashs=hashs
            )
    finally:
        milvus_db.data_release()
        milvus_db.connection_close()


async def search_milvus(text) -> Search2ResponseData:
    """
    Выполняет поиск в Milvus и возвращает контекст без истории диалога.

    :param text: Текст запроса.
    :return: Объект Search2ResponseData.
    """
    postgres_db = PostgreSQL(**config.postgres_config)
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Frida_bot_data', wiki_schema, wiki_index_params, wiki_search_params)
    try:
        response = milvus_db.search(text)
        if response is None:
            raise ValueError("Milvus search() вернул None")

        if isinstance(response, SearchFuture):
            response = response.result()

        if not isinstance(response, SearchResult):
            raise TypeError(f"Неподдерживаемый тип ответа от Milvus: {type(response)}")

        hashs = []
        for hits in response:
            for hit in hits:
                hash_val = hit.entity.get("hash")
                if hash_val:
                    hashs.append(hash_val)

        contexts = postgres_db.get_topics_texts_by_hashs(tuple(hashs))
        combined_context = ""
        for i, (book_name, text, url) in enumerate(contexts, start=1):
            book_name = book_name if book_name else ''
            combined_context += f" Контекст {i}: {book_name + ' ' + text}  URL: {url}"

        return Search2ResponseData(combined_context=combined_context, hashs=hashs)
    finally:
        milvus_db.data_release()
        milvus_db.connection_close()
