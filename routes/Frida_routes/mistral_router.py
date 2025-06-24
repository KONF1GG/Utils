"""
Маршруты для взаимодействия с моделью Mistral.

Этот модуль содержит маршруты FastAPI для отправки запросов к модели Mistral
и получения ответов. Основной маршрут "/v1/mistral" принимает данные запроса
в формате MistralRequest, отправляет их в модель и возвращает результат в виде
MistralResponse. В случае ошибок возвращается HTTP 500 с подробным описанием.

Функции:
    get_mistral_response(request_data: MistralRequest): 
    Асинхронно отправляет запрос к модели Mistral и возвращает ответ.

Зависимости:
    - get_mistral: функция для взаимодействия с моделью Mistral
    - MistralRequest, MistralResponse: схемы данных для запроса и ответа
"""

import logging
from fastapi import APIRouter, HTTPException

from mistral import get_mistral
from pyschemas import MistralRequest, MistralResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/v1/mistral",
    response_model=MistralResponse,
    summary="Получить ответ от модели Mistral",
    description=(
        "Отправляет запрос к модели Mistral с специализированным промтом "
        "для фриды и возвращает ответ"
    ),
    tags=["Mistral"]
)
async def get_mistral_response(
    request_data: MistralRequest,
):
    """Получает ответ от модели Mistral."""
    try:
        response_text = await get_mistral(
            request_data.text,
            request_data.combined_context,
            request_data.chat_history
        )
        return MistralResponse(mistral_response=response_text)

    except Exception as e:
        logger.error("Error in get_mistral_response: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Internal server error",
                "error": str(e)
            }
        ) from e
