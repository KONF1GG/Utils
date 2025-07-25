"""Маршруты для логирования сообщений в базу данных Frida."""

import logging

from fastapi import APIRouter, HTTPException

import config
from database import PostgreSQL
from pyschemas import LoggData, StatusResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/log", tags=["Frida"])
async def log_to_frida_db(data: LoggData) -> StatusResponse:
    """Логирует сообщение в базу данных Frida."""
    postgres = None
    try:
        postgres = PostgreSQL(**config.postgres_config)
        postgres.log_message(
            data.user_id,
            data.query,
            data.ai_response,
            data.status == 1,
            data.hashes,
            data.category,
        )

        return StatusResponse(status="success")
    except Exception as e:
        logger.exception("Failed to log message %s: %s", data.query, e)
        raise HTTPException(status_code=500, detail="Internal server error") from e
    finally:
        if postgres:
            postgres.connection_close()
