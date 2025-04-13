from typing import Literal
from pydantic import BaseModel


class StatusResponse(BaseModel):
    status: Literal['success', 'error']

class AddressModel(BaseModel):
    login: str
    address: str
    houseId: str

class PromtModel(BaseModel):
    id: str
    name: str
    template: str
    params: str

class ResponseCategory(BaseModel):
    category: str

class Input(BaseModel):
    text_query: str

class Count(BaseModel):
    count: int