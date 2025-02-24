from pydantic import BaseModel
import crud
from database import Milvus
import config
import funcs
from fastapi import FastAPI, HTTPException
from typing import List, Literal
import uvicorn

app = FastAPI()

class StatusResponse(BaseModel):
    status: Literal['success', 'error']

class Response(BaseModel):
    address: str

class Input(BaseModel):
    text_query: str

@app.get('/address', response_model=Response)
async def get_address_from_text(query: str):
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Address')
    house_numbers = funcs.extract_all_numbers_and_combinations(query)
    result = milvus_db.search(query)
    
    matched_address = None 
    
    for hit in result[0]:
        entity = hit.fields
        address = entity.get('address', '')
        
        for house_number in house_numbers:
            if funcs.normalize_text(house_number.lower()) in funcs.normalize_text(address.lower()).split():
                matched_address = address
                break  
    
    milvus_db.connection_close()
    
    if matched_address:
        return Response(address=matched_address)  
    else:
        raise HTTPException(status_code=404, detail="Address not found")  
    

@app.post('/upload_data', response_model=StatusResponse)
async def upload_data():
    try:
        crud.insert_addresses_to_milvus()
        return StatusResponse(status='success')
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during data upload")
