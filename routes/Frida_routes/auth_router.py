"""
Модуль содержит маршруты для аутентификации пользователей и управления администраторами.

Маршруты:
- POST /v1/auth: Проверяет наличие пользователя и добавляет его в базу данных, если он отсутствует.
- GET /v1/admins: Возвращает список всех администраторов системы.

Исключения:
- HTTPException с кодом 500 при ошибках работы с базой данных или других внутренних ошибках.
"""

from typing import Dict, List

import logging

from fastapi import APIRouter, HTTPException
from pyschemas import AuthResponse, Employee1C, UserData

import config
import crud
from database import PostgreSQL

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/auth", tags=["Frida"], response_model=AuthResponse)
async def check_and_add_user(data: UserData):
    """
    Проверяет сотрудника в 1С, при наличии добавляет в БД (если нет), возвращает ФИО и должность.
    """
    postgres = None
    try:
        # 1. Проверка в 1С
        if not data.user_id == 311362872:
            employee = await crud.auth_1c(data.user_id)
        else:
            employee = Employee1C(fio="Крохалев Леонтий Михайлович", jobTitle="Разработчик")
        if isinstance(employee, Employee1C):
            fio = employee.fio
            job_title = employee.jobTitle
        else:
            logger.info("User %s is not an employee in 1C.", data.user_id)
            raise HTTPException(
                status_code=403,
                detail="Доступ запрещён: пользователь не является сотрудником.",
            )

        # 2. Проверка в Postgres
        postgres = PostgreSQL(**config.postgres_config)
        if not postgres.user_exists(data.user_id):
            postgres.add_user_to_db(
                data.user_id, data.username,
                fio.split()[1] if fio else data.firstname,
                fio.split()[0] if fio else data.lastname
            )
            logger.info("User %s added to database.", data.user_id)
            status = "created"
            message = "User successfully added."
        else:
            logger.info("User %s already exists in database.", data.user_id)
            status = "exists"
            message = "User already exists."

        return AuthResponse(status=status, message=message, fio=fio, position=job_title)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to check/add user %s: %s", data.user_id, e)
        raise HTTPException(status_code=500, detail="Internal server error") from e
    finally:
        if postgres:
            postgres.connection_close()


@router.get("/v1/admins", response_model=List[Dict[str, str]], tags=["Frida"])
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
            {"user_id": user_id, "username": username} for user_id, username in admins
        ]
        return formatted_admins
    except Exception as e:
        logger.error("Error fetching admins: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail="Не удалось получить список администраторов"
        ) from e
    finally:
        if postgres:
            postgres.connection_close()
