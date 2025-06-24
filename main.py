"""
Главная точка входа для приложения VECTOR API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from lifespan import lifespan

# Routes
from routes.addresses_routes import router as address_router
from routes.promts_routes import router as prompts_router
from routes.redis_routes import router as redis_router

from routes.Frida_routes.auth_router import router as auth_router
from routes.Frida_routes.milvus_router import router as milvus_router
from routes.Frida_routes.mistral_router import router as mistral_router
from routes.Frida_routes.logger_router import router as log_router

app = FastAPI(
    title="VECTOR API",
    version="1.1.0",
    lifespan=lifespan
)

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(address_router)
app.include_router(prompts_router)
app.include_router(redis_router)
app.include_router(auth_router)
app.include_router(milvus_router)
app.include_router(mistral_router)
app.include_router(log_router)

if __name__ == '__main__':
    uvicorn.run('main:app', reload=True, host='0.0.0.0', port=8000)
