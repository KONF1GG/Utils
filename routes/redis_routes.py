"""
Маршрут:
    GET /all_users_from_redis
Описание:
    - Получает все данные пользователей из Redis.
    - Возвращает файл пользователю.
Возвращает:
    FileResponse: JSON-файл со всеми пользователями.
"""

import json

import os
from pathlib import Path
from typing import List
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import crud
from dependencies import RedisDependency
from funcs import cleanup_temp_dir
from pyschemas import RedisAddressModel, RedisAddressModelResponse

router = APIRouter()


@router.get("/all_users_from_redis", response_class=FileResponse, tags=["Redis"])
async def get_all_users_data_from_redis(redis: RedisDependency):
    """Получает все данные пользователей из Redis и возвращает их в виде JSON-файла."""
    temp_dir = Path("temp_files")
    temp_dir.mkdir(exist_ok=True)

    cleanup_temp_dir(temp_dir)

    temp_filename = None

    try:
        unique_keys = list(await crud.get_unique_keys_with_prefix())

        result = []
        batch_size = 1024

        for i in range(0, len(unique_keys), batch_size):
            batch_keys = unique_keys[i : i + batch_size]
            values = await redis.json().mget(batch_keys, path="$")

            cleaned_batch = []
            for item in values:
                if item and isinstance(item, list) and len(item) > 0:
                    cleaned_batch.append(item[0])

            result.extend(cleaned_batch)

        temp_filename = str(temp_dir / f"temp_users_{uuid.uuid4().hex}.json")

        with open(temp_filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        if not os.path.exists(temp_filename):
            raise HTTPException(500, detail="Failed to create temporary file")

        return FileResponse(
            path=temp_filename,
            media_type="application/json",
            filename="users_data.json",
        )

    except Exception as e:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(500, detail=str(e)) from e


@router.get(
    "/redis_addresses", tags=["Redis"], response_model=RedisAddressModelResponse
)
async def get_addresses(query_address: str, redis: RedisDependency):
    """Получает все адреса из Redis"""
    try:
        from redis.commands.search.query import Query

        query = Query(query_address.lower()).paging(0, 40)
        addresses = await redis.ft("idx:adds").search(query)

        if not addresses.docs:
            raise HTTPException(status_code=404, detail="No addresses found")

        addresses_models = []
        for doc in addresses.docs:
            data = json.loads(doc.json)
            if data["territoryId"] is not None:
                addresses_models.append(
                    RedisAddressModel(
                        id=data["id"],
                        address=data.get("addressShort") or data.get("title", ""),
                        territory_id=data["territoryId"],
                        territory_name=data["territory"],
                    )
                )
        return RedisAddressModelResponse(addresses=addresses_models)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/redis_address_by_id", tags=["Redis"], response_model=RedisAddressModel)
async def get_address_by_id(address_id: str, redis: RedisDependency):
    """Получает адрес по ID из Redis"""
    try:
        address_result = await redis.json().get(f"adds:{address_id}")
        if address_result is None:
            raise HTTPException(status_code=404, detail="Address not found")

        if isinstance(address_result, str):
            address_data = json.loads(address_result)
        else:
            address_data = address_result

        return RedisAddressModel(
            id=address_data["id"],
            address=address_data.get("addressShort") or address_data.get("title", ""),
            territory_id=address_data["territoryId"],
            territory_name=address_data["territory"],
            conn_type=address_data.get("conn_type")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/redis_tariffs", tags=["Redis"])
async def get_tariffs(territory_id: str, redis: RedisDependency):
    """Получает тарифы для конкретного territory_id из Redis"""
    try:
        tariffs_result = await redis.json().get(f"terrtar:{territory_id}")
        if tariffs_result is None:
            raise HTTPException(status_code=404, detail="No tariffs found")
        if isinstance(tariffs_result, str):
            import json

            tariffs_result = json.loads(tariffs_result)
        return tariffs_result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
