"""
Маршруты для поиска и загрузки данных в Milvus.

Этот модуль реализует следующие эндпоинты FastAPI:
- /v1/mlv_search: Поиск в Milvus с учетом истории пользователя.
- /v2/mlv_search: Поиск в Milvus без учета истории пользователя.
- /v1/upload_wiki_data: Загрузка данных из базы wiki в Milvus (только для администраторов).
- /v1/add_topic: Добавление новой темы в базу данных PostgreSQL и Milvus.

Зависимости:
- crud: функции для работы с Milvus и базой данных.
- config: настройки подключения к PostgreSQL.
- database: класс для работы с PostgreSQL.
- pyschemas: схемы для валидации и сериализации данных.

"""

import logging
from fastapi import APIRouter, Body, HTTPException, Query, status, Depends

import crud
from config import postgres_config

from database import PostgreSQL
from pyschemas import AddTopicRequest, Search2ResponseData, SearchParams, SearchResponseData

router = APIRouter()
logger = logging.getLogger(__name__)


def get_search_params(
    user_id: int = Query(...), text: str = Query(...)
) -> SearchParams:
    """Получает параметры поиска."""
    return SearchParams(user_id=user_id, text=text)


@router.get("/v1/mlv_search", response_model=SearchResponseData, tags=["Milvus"])
async def search_endpoint_with_history(
    params: SearchParams = Depends(get_search_params),
):
    """Поиск в Milvus с историей."""
    try:
        return await crud.search_milvus_and_prep_data(params.text, params.user_id)
    except Exception as e:
        logger.error("Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v2/mlv_search", response_model=Search2ResponseData, tags=["Milvus"])
async def search_endpoint(text: str = Query(...)):
    """Поиск в Milvus без истории."""
    try:
        return await crud.search_milvus(text)
    except Exception as e:
        logger.error("Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/v1/upload_wiki_data",
    responses={
        200: {"description": "Данные успешно загружены"},
        403: {"description": "Пользователь не является администратором"},
        500: {"description": "Ошибка сервера при обработке запроса"},
    },
    tags=["Milvus"],
)
async def upload_wiki_data_from_mysqldb_to_milvus(user_data: dict = Body(...)):
    """Загрузка данных из базы wiki в Milvus."""
    try:
        user_id = user_data.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="user_id is required in request body",
            )

        postgres = PostgreSQL(**postgres_config)

        # Проверка прав администратора
        if not postgres.check_user_is_admin(user_id):
            logger.warning(
                "User %s attempted to upload wiki data without admin rights", user_id
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Только администраторы могут загружать данные wiki",
            )

        # Загрузка данных
        wiki_response = crud.insert_wiki_data()
        if not wiki_response:
            logger.error("Failed to insert wiki data")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ошибка при загрузке данных из базы wiki",
            )

        # Перенос данных в Milvus
        milvus_data_count, deleted_data_count = (
            crud.insert_all_data_from_postgres_to_milvus()
        )

        # Формирование ответа
        response_message = ""
        if deleted_data_count:
            response_message = f"Обнаружено и удалено {deleted_data_count} дубликатов. "
        response_message += f"Текущее количество записей в базе: {milvus_data_count}"

        logger.info("Successfully uploaded wiki data. %s", response_message)

        return {
            "status": "success",
            "message": response_message,
            "data": {
                "total_records": milvus_data_count,
                "duplicates_removed": deleted_data_count,
            },
        }

    except HTTPException:
        raise  # Пробрасываем уже обработанные HTTP исключения
    except Exception as e:
        logger.error("Unexpected error during wiki data upload: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Произошла непредвиденная ошибка при обработке запроса",
        ) from e
    finally:
        # Закрытие соединения с БД
        postgres.connection_close()



@router.post(
    "/v1/add_topic",
    responses={
        200: {"description": "Тема успешно добавлена"},
        400: {"description": "Ошибка валидации данных"},
        500: {"description": "Ошибка сервера при добавлении темы"},
    },
    tags=["Milvus"],
)
async def add_topic_route(data: AddTopicRequest):
    """
    Добавляет новую тему в базу данных PostgreSQL и Milvus.
    Ожидает в теле запроса: {"title": str, "text": str, "user_id": int}
    """
    try:
        result = crud.add_new_topic(data.title, data.text, data.user_id)
        if result is True:
            return {"status": "success", "message": "Тема успешно добавлена"}
        else:
            logger.error("Ошибка при добавлении темы: %s", result)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка при добавлении темы: {result}",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error during topic addition: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Произошла непредвиденная ошибка при добавлении темы",
        ) from e
