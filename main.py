from pydantic import BaseModel
import crud
from database import Milvus
import config
from fastapi import FastAPI, HTTPException
from typing import List, Literal
import uvicorn
from milvus_schemas import address_schema, address_index_params, address_search_params, category_schema, category_index_params, category_search_params
from torch import cuda

app = FastAPI()

class StatusResponse(BaseModel):
    status: Literal['success', 'error']

class ResponseAddress(BaseModel):
    login: str
    address: str
    houseId: str

class ResponseCategory(BaseModel):
    category: str

class Input(BaseModel):
    text_query: str

class Count(BaseModel):
    count: int

# @app.get('/v1/address', response_model=List[ResponseAddress])
# async def get_address_from_text(query: str):
#     try:
#         milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Address', address_schema, address_index_params, address_search_params)
#         milvus_db.collection.load()
#         result = milvus_db.search(query, ['text', 'house_id', 'flat'], limit=100)
#         addresses_list = []
#         matched_addresses = []
#         first =  result[0][0].fields
#         for hit in result[0]:
#             entity = hit.fields
#             login = entity.get('hash', '')
#             address = entity.get('text', '')[9:]
#             house_id = entity.get('house_id', '')
#             flat = entity.get('flat', '')
#             flat = None if flat == 'None' else flat
#             if first.get('house_id') == entity.get('house_id'):
#                 matched_addresses.append(ResponseAddress(login=login, address=address, houseId=house_id))

#             # for house_number in house_numbers:
#             #     if funcs.normalize_text(house_number.lower()) in funcs.normalize_text(address.lower()).split():
#             #         matched_address = address
#             #         break  
#             # addresses_list.append(ResponseAddress(login=login, address=address, houseId=house_id))        
#         if matched_addresses:
#             return matched_addresses
#         else:
#             raise HTTPException(status_code=404, detail="Address not found")  
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=e)
#     finally:
#         milvus_db.data_release()
#         milvus_db.connection_close()
    
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
            address = entity.get('text', '')
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
    
@app.get('/category', response_model=ResponseCategory)
async def get_category_from_text(query: str):
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Categories', category_schema, category_index_params, category_search_params)
    result = milvus_db.search(query_text=query, additional_fields=['text'], limit=1)
    for hit in result[0]:
        entity = hit.fields
        category = entity.get('text', '')
        break  
    
    milvus_db.connection_close()
    
    if category:
        return ResponseCategory(category=category)  
    else:
        raise HTTPException(status_code=404, detail="Category not found")  


@app.post('/upload_address_data', response_model=StatusResponse)
async def upload_data():
    try:
        await crud.main()
        return StatusResponse(status='success')
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during data upload")

# @app.post('/upload_category_data', response_model=StatusResponse)
# async def upload_data():
#     try:
#         crud.insert_categories_to_milvus()
#         return StatusResponse(status='success')
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Internal server error during data upload")

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

if __name__ == '__main__':
    uvicorn.run('main:app', reload=True, host='0.0.0.0')