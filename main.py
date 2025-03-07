import asyncio
from pydantic import BaseModel
import crud
from database import Milvus
import config
import funcs
from fastapi import FastAPI, HTTPException
from typing import List, Literal
import uvicorn
from milvus_schemas import address_schema, address_index_params, address_search_params, category_schema, category_index_params, category_search_params

app = FastAPI()

class StatusResponse(BaseModel):
    status: Literal['success', 'error']

class ResponseAddress(BaseModel):
    address: str

class ResponseCategory(BaseModel):
    category: str

class Input(BaseModel):
    text_query: str

@app.get('/address', response_model=ResponseAddress)
async def get_address_from_text(query: str):
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Address', address_schema, address_index_params, address_search_params)
    house_numbers = funcs.extract_all_numbers_and_combinations(query)
    result = milvus_db.search(query, ['text', 'house_id'])
    
    matched_address = None 
    
    for hit in result[0]:
        entity = hit.fields
        address = entity.get('text', '')
        house_id = entity.get('house_id', '')
        
        for house_number in house_numbers:
            if funcs.normalize_text(house_number.lower()) in funcs.normalize_text(address.lower()).split():
                matched_address = address
                break  
    
    milvus_db.connection_close()
    
    if matched_address:
        return ResponseAddress(address=matched_address)  
    else:
        raise HTTPException(status_code=404, detail="Address not found")  
    
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

@app.post('/upload_category_data', response_model=StatusResponse)
async def upload_data():
    try:
        crud.insert_categories_to_milvus()
        return StatusResponse(status='success')
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during data upload")
