"""
Модуль для управления жизненным циклом FastAPI приложения с использованием планировщика задач
APScheduler.

Функции:
- lifespan(app: FastAPI): Контекстный менеджер для запуска и остановки задач планировщика.

Задачи:
- insert_promts_from_redis_to_milvus: Загрузка данных из Redis в Milvus.
- upload_data_wiki_data_to_milvus: Загрузка данных из Wiki в Milvus.

Планировщик:
- Задачи выполняются ежедневно в 03:00.
"""
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from crud import insert_promts_from_redis_to_milvus, upload_data_wiki_data_to_milvus
from redis import from_url
import config 

scheduler = AsyncIOScheduler()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекстный менеджер для запуска и остановки задач планировщика."""
    logger.info('START')
    redis = from_url(
        f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
        password=config.REDIS_PASSWORD,
        decode_responses=True
    )
    try:
        scheduler.add_job(
            insert_promts_from_redis_to_milvus,
            trigger=CronTrigger(hour=3, minute=0),
            args=[redis],
        )
        scheduler.add_job(
            upload_data_wiki_data_to_milvus,
            trigger=CronTrigger(hour=3, minute=0)
        )
        scheduler.start()
        yield
    finally:
        scheduler.shutdown()
        await redis.aclose()
        logger.info('STOP')
