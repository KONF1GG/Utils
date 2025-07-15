"""
Маршруты для взаимодействия с AI.

Этот модуль содержит маршруты FastAPI для отправки запросов к модели AI
и получения ответов. Основной маршрут "/v1/ai" принимает данные запроса
в формате AIRequest, отправляет их в модель и возвращает результат в виде
AIResponse. В случае ошибок возвращается HTTP 500 с подробным описанием.

Функции:
    get_ai_response(request_data: AIRequest): 
    Асинхронно отправляет запрос к модели AI и возвращает ответ.

Зависимости:
    - get_ai: функция для взаимодействия с моделью AI
    - AIRequest, AIResponse: схемы данных для запроса и ответа
"""

import logging
from fastapi import APIRouter, HTTPException

from ai import get_ai
from pyschemas import AIRequest, AIResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/v1/ai",
    response_model=AIResponse,
    summary="Получить ответ от модели AI",
    description=(
        "Отправляет запрос к модели AI с специализированным промтом "
        "для фриды и возвращает ответ"
    ),
    tags=["AI"]
)
async def get_ai_response(
    request_data: AIRequest,
):
    """Получает ответ от модели AI."""
    print(request_data)
    try:
        logger.info("Received AI model: %s", request_data.model)
        response_text = await get_ai(
            request_data.text,
            request_data.combined_context,
            request_data.chat_history,
            request_data.input_type,
            request_data.model,
        )
        return AIResponse(ai_response=response_text)

    except Exception as e:
        logger.error("Error in get_ai_response: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Internal server error",
                "error": str(e)
            }
        ) from e
