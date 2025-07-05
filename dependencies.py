
from typing import Annotated, Any, AsyncGenerator
from redis.asyncio import Redis, from_url
from fastapi import Depends

import config


async def get_redis_connection() -> AsyncGenerator[Redis, None]:
    """Получение подключения к Redis."""
    connection = from_url(
            f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
    try:
        yield connection
    finally:
        await connection.aclose()


RedisDependency = Annotated[Any, Depends(get_redis_connection)]