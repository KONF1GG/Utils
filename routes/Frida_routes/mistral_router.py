from typing import Literal
from fastapi import APIRouter, HTTPException
import logging

from mistral import mistral
from pyschemas import MistralResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/mistral", response_model=MistralResponse)
async def get_mistral_response(text: str, combined_context: str, chat_history: str, input_type: Literal['voice', 'csv', 'text'] = 'text'):
    try:
        response_text = await mistral(text, combined_context, chat_history)
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