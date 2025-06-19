from typing import List
from fastapi import APIRouter, HTTPException
from mistralai import Dict
from starlette.responses import JSONResponse
import logging

import config
from database import PostgreSQL
from pyschemas import UserData

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/auth", tags=["Frida"])
async def check_and_add_user(data: UserData):
    try:
        postgres = PostgreSQL(**config.postgres_config)

        if postgres.user_exists(data.user_id):
            logger.info(f"User {data.user_id} already exists.")
            return JSONResponse(
                status_code=200,
                content={"status": "exists", "message": "User already exists."}
            )

        postgres.add_user_to_db(
            data.user_id,
            data.username,
            data.firstname,
            data.lastname
        )
        logger.info(f"User {data.user_id} added to database.")
        return JSONResponse(
            status_code=201,
            content={"status": "created", "message": "User successfully added."}
        )

    except Exception as e:
        logger.exception(f"Failed to check/add user {data.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get('/v1/admins', response_model=List[Dict[str, str]], tags=["Frida"])
async def get_all_admins():
    """
    Получает список всех администраторов системы.
    
    Returns:
        List[Dict[str, str]]: Список словарей с информацией об администраторах,
        где каждый словарь содержит ключи 'user_id' и 'username'.
        
    Raises:
        HTTPException: Если произошла ошибка при работе с базой данных
    """
    postgres = None
    try:
        postgres = PostgreSQL(**config.postgres_config)
        admins = postgres.get_admins()
        
        # Преобразуем список кортежей в список словарей
        formatted_admins = [
            {"user_id": user_id, "username": username}
            for user_id, username in admins
        ]
        
        return formatted_admins
        
    except Exception as e:
        logger.error(f"Error fetching admins: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Не удалось получить список администраторов"
        )
    finally:
        if postgres:
            postgres.connection_close()