from fastapi import FastAPI
import uvicorn
import lifespan
from fastapi.middleware.cors import CORSMiddleware
from lifespan import lifespan

from routes.addresses_routes import router as address_router
from routes.promts_routes import router as promts_router
from routes.redis_routes import router as redis_router


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
app.include_router(promts_router)
app.include_router(redis_router)

if __name__ == '__main__':
    uvicorn.run('main:app', reload=True, host='0.0.0.0', port=8080)