from multiprocessing import resource_tracker
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import crud
from database import Milvus
import config
from fastapi import FastAPI, HTTPException
from typing import List
import uvicorn
import lifespan
from milvus_schemas import address_schema, address_index_params, address_search_params, promt_schema, promt_index_params, promt_search_params
from torch import cuda
from fastapi.middleware.cors import CORSMiddleware
from lifespan import lifespan
from pyschemas import Count, ResponseAddress, ResponsePromt, StatusResponse
from fastapi.encoders import jsonable_encoder

app = FastAPI(
    title="VECTOR API",
    version="1.0.0",
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

@app.get('/v1/address', response_model=List[ResponseAddress])
async def get_address_from_text(query: str):
    try:
        milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Address', address_schema, address_index_params, address_search_params)
        milvus_db.collection.load()
        result = milvus_db.search(query, ['text', 'house_id', 'flat'], limit=10)
        addresses_list = []
        for hit in result[0]:
            entity = hit.fields
            login = entity.get('hash', '')
            address = entity.get('text', '')[9:]
            house_id = entity.get('house_id', '')
            flat = entity.get('flat', '')
            flat = None if flat == 'None' else flat

            addresses_list.append(ResponseAddress(login=login, address=address, houseId=house_id))        
        if addresses_list:
            return addresses_list
        else:
            raise HTTPException(status_code=404, detail="Address not found")  
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
    finally:
        milvus_db.data_release()
        milvus_db.connection_close()

@app.get('/v1/promt', response_model=List[ResponsePromt])
async def get_promt_by_query(query: str):
    try:
        milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Promts', promt_schema, promt_index_params, promt_search_params)
        milvus_db.collection.load()
        result = milvus_db.search(query, ['name', 'text'], limit=3)
        promts_list = []
        for hit in result[0]:
            entity = hit.fields
            id = entity.get('hash', '')
            name = entity.get('name', '')
            template = entity.get('text', '')[9:]
            params = entity.get('params', '')

            promts_list.append(ResponsePromt(id=id, name=name, template=template, params=params))        
        if promts_list:
            return promts_list
        else:
            raise HTTPException(status_code=404, detail="Promts not found")  
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
    finally:
        milvus_db.data_release()
        milvus_db.connection_close()
    
@app.post('/upload_address_data', response_model=StatusResponse)
async def upload_address_data():
    try:
        await crud.insert_addresses_from_redis_to_milvus()
        return StatusResponse(status='success')
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during data upload")
    
@app.post('/upload_promts_data', response_model=StatusResponse)
async def upload_promts_data():
    try:
        await crud.insert_promts_from_redis_to_milvus()
        return StatusResponse(status='success')
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during data upload")

@app.get('/addresses_count', response_model=Count)
async def get_address_count():
    try:
        print(cuda.is_available())
        milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Address', address_schema, address_index_params, address_search_params)
        address_count = milvus_db.get_data_count()
        return Count(count=address_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail='Error during get data from milvus')
    finally:
        milvus_db.connection_close()

@app.get('/all_users_from_redis', response_model=list[dict])
async def get_all_users_data_from_redis():
    r = None
    try:
        unique_keys = list(await crud.get_unique_keys_with_prefix())
        r = redis.from_url(
            f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
        
        result = []
        batch_size = 1024
        
        for i in range(0, len(unique_keys), batch_size):
            batch_keys = unique_keys[i:i + batch_size]
            values = await r.json().mget(batch_keys, path="$")
            
            # Чистим данные
            cleaned_batch = []
            for item in values:
                if item and isinstance(item, list) and len(item) > 0:
                    cleaned_batch.append(item[0])
            
            result.extend(cleaned_batch)
        
        return jsonable_encoder(result)
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    
    finally:
        if r:
            await r.aclose()
    


if __name__ == '__main__':
    uvicorn.run('main:app', reload=True, host='0.0.0.0')