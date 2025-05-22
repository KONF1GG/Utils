from typing import List
from fastapi import APIRouter, HTTPException
import config
import crud
from torch import cuda
from milvus_schemas import address_schema, address_index_params, address_search_params
from database import Milvus
from pyschemas import AddressModel, Count, StatusResponse

router = APIRouter()

@router.get('/v1/address', response_model=List[AddressModel])
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

            addresses_list.append(AddressModel(login=login, address=address, houseId=house_id))        
        if addresses_list:
            return addresses_list
        else:
            raise HTTPException(status_code=404, detail="Address not found")  
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
    finally:
        milvus_db.data_release()
        milvus_db.connection_close()


@router.post('/upload_address_data', response_model=StatusResponse)
async def upload_address_data():
    try:
        await crud.insert_addresses_from_redis_to_milvus()
        return StatusResponse(status='success')
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during data upload")


@router.get('/addresses_count', response_model=Count)
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


@router.post('/v1/addresses', response_model=StatusResponse)
async def insert_addresses_to_milvus(data: List[List]):
    milvus_db = None
    try:
        milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Address', address_schema, address_index_params, address_search_params)
        await crud.insert_addresses_to_milvus(data, milvus_db)
        return StatusResponse(status='success')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if milvus_db:
            milvus_db.connection_close()  