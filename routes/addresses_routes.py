"""
Маршруты для работы с адресами в Milvus.

Этот модуль содержит набор маршрутов FastAPI для взаимодействия с коллекцией адресов в Milvus.
Реализованы следующие эндпоинты:

- GET /v1/address: Поиск адресов по текстовому запросу.
- POST /v1/addresses: Вставка списка адресов в Milvus.
- POST /upload_address_data: Загрузка адресов из Redis в Milvus.
- GET /addresses_count: Получение количества адресов в Milvus.

Каждый маршрут обрабатывает возможные ошибки и возвращает соответствующие HTTP-ответы.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from torch import cuda

import config
import crud

from milvus_schemas import address_schema, address_index_params, address_search_params
from database import Milvus
from pyschemas import AddressModel, Count, StatusResponse


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get('/v1/address', response_model=List[AddressModel], tags=["ChatBot addresses"])
async def get_address_from_text(query: str):
    """Получает адреса из Milvus по текстовому запросу."""
    milvus_db = None
    try:
        milvus_db = Milvus(
            config.MILVUS_HOST,
            config.MILVUS_PORT,
            'Address',
            address_schema,
            address_index_params,
            address_search_params
        )

        milvus_db.collection.load()
        result = milvus_db.search(query, ['text', 'house_id', 'flat'], limit=10)
        addresses_list = []

        for hit in result[0]:
            entity = hit.fields
            login = entity.get('hash', '')
            address = entity.get('text', '')[9:]
            house_id = entity.get('house_id', '')
            flat = entity.get('flat', '')
            flat = None if flat == 'None' else flat

            addresses_list.append(AddressModel(login=login, address=address, houseId=house_id))
        if addresses_list:
            return addresses_list

        logger.warning("Address not found for query: %s", query)

        raise HTTPException(status_code=404, detail="Address not found")
    except Exception as e:
        logger.error("Error in get_address_from_text: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if milvus_db:
            milvus_db.data_release()
            milvus_db.connection_close()

@router.post('/v1/addresses', response_model=StatusResponse, tags=["ChatBot addresses"])
async def insert_addresses_to_milvus(data: List[List]):
    """Вставляет адреса в Milvus."""
    milvus_db = None
    try:
        logger.info("Inserting addresses: %s", data)
        milvus_db = Milvus(
                config.MILVUS_HOST,
                config.MILVUS_PORT,
                'Address',
                address_schema,
                address_index_params,
                address_search_params)
        await crud.insert_addresses_to_milvus(data, milvus_db)
        return StatusResponse(status='success')
    except Exception as e:
        logger.error("Error in insert_addresses_to_milvus: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if milvus_db:
            milvus_db.connection_close()

@router.post('/upload_address_data', response_model=StatusResponse, tags=["ChatBot addresses"])
async def upload_address_data():
    """Загружает адреса из Redis в Milvus."""
    try:
        logger.info("Uploading address data from Redis to Milvus")
        await crud.insert_addresses_from_redis_to_milvus()
        return StatusResponse(status='success')
    except Exception as e:
        logger.error("Internal server error during data upload: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during data upload") from e

@router.get('/addresses_count', response_model=Count, tags=["ChatBot addresses"])
async def get_address_count():
    """Получает количество адресов в Milvus."""
    milvus_db = None
    try:
        logger.info("Checking CUDA availability: %s", cuda.is_available())
        milvus_db = Milvus(config.MILVUS_HOST,
         config.MILVUS_PORT, 'Address', address_schema, address_index_params, address_search_params)
        address_count = milvus_db.get_data_count()
        return Count(count=address_count)
    except Exception as e:
        logger.error("Error during get data from milvus: %s", e)
        raise HTTPException(status_code=500, detail='Error during get data from milvus') from e
    finally:
        if milvus_db:
            milvus_db.connection_close()
