"""
Этот модуль определяет маршруты FastAPI для управления подсказками 
чат-бота с использованием векторной базы данных Milvus.

Маршруты:
    - GET /v1/promt: Поиск промтов чат-бота по строке запроса с помощью 
    поиска по векторному сходству.
    - POST /v1/promts: Добавление нового промта в базу данных Milvus.
    - POST /upload_promts_data: Загрузка данных промтов из Redis в Milvus.
    - GET /promts_count: Получение общего количества промтов, хранящихся в Milvus.

Зависимости:
    - config: Настройки конфигурации для подключения к Milvus.
    - crud: CRUD-операции для промтов.
    - database.Milvus: Интерфейс базы данных Milvus.
    - milvus_schemas: Схемы и параметры для коллекций и поиска в Milvus.
    - pyschemas: Pydantic-модели для ответов API.

"""
from typing import Dict, List
from fastapi import APIRouter, HTTPException
import config
import crud
from database import Milvus
from dependencies import RedisDependency
from milvus_schemas import promt_schema, promt_index_params, promt_search_params
from pyschemas import Count, PromtModel, StatusResponse

router = APIRouter()


@router.get('/v1/promt', response_model=List[PromtModel], tags=["ChatBot promts"])
async def get_promt_by_query(query: str):
    """Получает промты чат-бота из Milvus по текстовому запросу."""
    try:
        milvus_db = Milvus(
            config.MILVUS_HOST,
            config.MILVUS_PORT,
            'Promts',
            promt_schema,
            promt_index_params,
            promt_search_params
        )
        milvus_db.collection.load()
        result = milvus_db.search(query, ['name', 'text'], limit=3)
        promts_list = []
        hits = result[0]
        for hit in hits:
            entity = hit.fields
            promt_id = entity.get('hash', '')
            name = entity.get('name', '')
            template = entity.get('text', '')[9:]
            params = entity.get('params', '')
            if hit.distance < 0.42:
                promts_list.append(PromtModel(id=promt_id,
                                               name=name,
                                               template=template,
                                               params=params))
        if promts_list:
            return promts_list
        else:
            raise HTTPException(status_code=404, detail="Promts not found")
    finally:
        milvus_db.data_release()
        milvus_db.connection_close()

@router.post('/v1/promts', response_model=StatusResponse, tags=["ChatBot promts"])
async def insert_promts_to_milvus(data: Dict):
    """Вставляет новую промт в Milvus."""
    milvus_db = None
    try:
        milvus_db = Milvus(
            config.MILVUS_HOST,
            config.MILVUS_PORT,
            'Promts',
            promt_schema,
            promt_index_params,
            promt_search_params
        )
        promt_model = PromtModel(**data)
        await crud.insert_promts_to_milvus([promt_model], milvus_db)
        return StatusResponse(status='success')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if milvus_db:
            milvus_db.connection_close()

@router.post('/upload_promts_data', response_model=StatusResponse, tags=["ChatBot promts"])
async def upload_promts_data(redis: RedisDependency):
    """Загружает данные промтов из Redis в Milvus."""
    try:
        await crud.insert_promts_from_redis_to_milvus(redis)
        return StatusResponse(status='success')
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Internal server error during data upload") from e

@router.get('/promts_count', response_model=Count, tags=["ChatBot promts"])
async def get_promts_count():
    """Получает общее количество промтов, хранящихся в Milvus."""
    try:
        milvus_db = Milvus(
            config.MILVUS_HOST,
            config.MILVUS_PORT,
            'Promts', promt_schema,
            promt_index_params,
            promt_search_params)
        address_count = milvus_db.get_data_count()
        return Count(count=address_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail='Error during get data from milvus') from e
    finally:
        milvus_db.connection_close()
