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
    r = None
    temp_dir = Path("temp_files")
    temp_dir.mkdir(exist_ok=True)
    
    # Очищаем папку перед обработкой нового запроса
    cleanup_temp_dir(temp_dir)
    
    temp_filename = None
    
    try:
        # Получаем данные из Redis
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
            values = await r.json().mget(batch_keys, path="$")
            
            cleaned_batch = []
            for item in values:
                if item and isinstance(item, list) and len(item) > 0:
                    cleaned_batch.append(item[0])
            
            result.extend(cleaned_batch)
        
        # Создаем временный файл
        temp_filename = str(temp_dir / f"temp_users_{uuid.uuid4().hex}.json")
        
        # Записываем данные с правильной кодировкой
        with open(temp_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        if not os.path.exists(temp_filename):
            raise HTTPException(500, detail="Failed to create temporary file")
        
        # Возвращаем файл с гарантированным удалением после отправки
        return FileResponse(
            path=temp_filename,
            media_type="application/json",
            filename="users_data.json",
        )
    
    except Exception as e:
        # Удаляем временный файл при ошибке
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(500, detail=str(e))
    
    finally:
        if r:
            await r.aclose()