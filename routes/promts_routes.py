from typing import Dict, List
from fastapi import APIRouter, HTTPException
import config
import crud
from database import Milvus
from milvus_schemas import promt_schema, promt_index_params, promt_search_params
from pyschemas import Count, PromtModel, StatusResponse

router = APIRouter()


@router.get('/v1/promt', response_model=List[PromtModel])
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

            promts_list.append(PromtModel(id=id, name=name, template=template, params=params))        
        if promts_list:
            return promts_list
        else:
            raise HTTPException(status_code=404, detail="Promts not found")  
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
    finally:
        milvus_db.data_release()
        milvus_db.connection_close()

@router.post('/upload_promts_data', response_model=StatusResponse)
async def upload_promts_data():
    try:
        await crud.insert_promts_from_redis_to_milvus()
        return StatusResponse(status='success')
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during data upload")

@router.get('/promts_count', response_model=Count)
async def get_promts_count():
    try:
        milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Promts', promt_schema, promt_index_params, promt_search_params)
        address_count = milvus_db.get_data_count()
        return Count(count=address_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail='Error during get data from milvus')
    finally:
        milvus_db.connection_close()

@router.post('/v1/promts', response_model=StatusResponse)
async def insert_promts_to_milvus(data: Dict):
    milvus_db = None
    try:
        milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Promts', promt_schema, promt_index_params, promt_search_params)
        await crud.insert_promts_to_milvus([data], milvus_db)
        return StatusResponse(status='success')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if milvus_db:
            milvus_db.connection_close()