"""
Модуль содержит маршруты для аутентификации пользователей и управления администраторами.

Маршруты:
- POST /v1/auth: Проверяет наличие пользователя и добавляет его в базу данных, если он отсутствует.
- GET /v1/admins: Возвращает список всех администраторов системы.

Исключения:
- HTTPException с кодом 500 при ошибках работы с базой данных или других внутренних ошибках.
"""
from typing import List

import logging

from fastapi import APIRouter, HTTPException
from mistralai import Dict
from starlette.responses import JSONResponse

import config
from database import PostgreSQL
from pyschemas import UserData

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/auth", tags=["Frida"])
async def check_and_add_user(data: UserData):
    """Проверяет наличие пользователя и добавляет его в базу данных, если он отсутствует."""
    try:
        postgres = PostgreSQL(**config.postgres_config)

        if postgres.user_exists(data.user_id):
            logger.info("User %s already exists.", data.user_id)
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

        logger.info("User %s added to database.", data.user_id)
        return JSONResponse(
            status_code=201,
            content={"status": "created", "message": "User successfully added."}
        )

    except Exception as e:
        logger.exception("Failed to check/add user %s: %s", data.user_id, e)
        raise HTTPException(status_code=500, detail="Internal server error") from e

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
        formatted_admins = [
            {"user_id": user_id, "username": username}
            for user_id, username in admins
        ]
        return formatted_admins
    except Exception as e:
        logger.error("Error fetching admins: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Не удалось получить список администраторов"
        ) from e
    finally:
        if postgres:
            postgres.connection_close()
