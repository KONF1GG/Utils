from fastapi import APIRouter, HTTPException
from starlette.responses import JSONResponse
import logging

import config
from database import PostgreSQL
from pyschemas import LoggData, StatusResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/log", tags=["Frida"])
async def log_to_frida_db(data: LoggData) -> StatusResponse:
    try:
        postgres = PostgreSQL(**config.postgres_config)
        postgres.log_message(data.user_id, data.query, data.mistral_response, True if data.status == 1 else False, data.hashes)

        return StatusResponse(status='success')
    except Exception as e:
        logger.exception(f"Failed to logg messege {data.query}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        postgres.connection_close()
