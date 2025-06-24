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
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import redis.asyncio as redis

import config
import crud
from funcs import cleanup_temp_dir

router = APIRouter()

@router.get('/all_users_from_redis', response_class=FileResponse)
async def get_all_users_data_from_redis():
    """Получает все данные пользователей из Redis и возвращает их в виде JSON-файла."""
    r = None
    temp_dir = Path("temp_files")
    temp_dir.mkdir(exist_ok=True)

    cleanup_temp_dir(temp_dir)

    temp_filename = None

    try:
        unique_keys = list(await crud.get_unique_keys_with_prefix())
        r = redis.from_url(
            f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )

        result = []
        batch_size = 1024

        for i in range(0, len(unique_keys), batch_size):
            batch_keys = unique_keys[i:i + batch_size]
            values = r.json().mget(batch_keys, path="$")

            cleaned_batch = []
            for item in values:
                if item and isinstance(item, list) and len(item) > 0:
                    cleaned_batch.append(item[0])

            result.extend(cleaned_batch)

        temp_filename = str(temp_dir / f"temp_users_{uuid.uuid4().hex}.json")

        with open(temp_filename, 'w', encoding='utf-8') as f:
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

    finally:
        if r:
            await r.aclose()
