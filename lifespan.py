from contextlib import asynccontextmanager
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from crud import insert_promts_from_redis_to_milvus, upload_data_wiki_data_to_milvus

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('START')
    scheduler.add_job(
        insert_promts_from_redis_to_milvus,
        trigger=CronTrigger(hour=3, minute=0),
    )
    scheduler.add_job(
        upload_data_wiki_data_to_milvus,
        trigger=CronTrigger(hour=3, minute=0)
    )
    scheduler.start()
    yield
    scheduler.shutdown()
    print('STOP')
