from typing import Literal
from fastapi import APIRouter, HTTPException
import logging

from mistral import get_mistral
from pyschemas import MistralRequest, MistralResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/v1/mistral",
    response_model=MistralResponse,
    summary="Получить ответ от модели Mistral",
    description="Отправляет запрос к модели Mistral с специализированным промтом для фриды и возвращает ответ",
    tags=["Mistral"]
)
async def get_mistral_response(
    request_data: MistralRequest,
):
    try:
        response_text = await get_mistral(
            request_data.text,
            request_data.combined_context,
            request_data.chat_history
        )
        return MistralResponse(mistral_response=response_text)
    
    except Exception as e:
        logger.error(f"Error in get_mistral_response: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Internal server error",
                "error": str(e)
            }
        )