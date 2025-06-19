from typing import List, Literal
from pydantic import BaseModel, Field


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

class Page(BaseModel):
    title: str
    text: str
    book_name: str
    book_slug: str
    page_slug: str

class UserData(BaseModel):
    user_id: int
    firstname: str
    lastname: str = ""
    username: str = ""

class SearchParams(BaseModel):
    user_id: int
    text: str


class SearchResponseData(BaseModel):
    combined_context: str = Field(..., description="Контекст Вики")
    chat_history: str = Field(..., description="Отформатированная история диалога")
    hashs: List[str] = Field(..., description="ID контекстов которые используются")

class Search2ResponseData(BaseModel):
    combined_context: str = Field(..., description="Контекст Вики")
    hashs: List[str] = Field(..., description="ID контекстов которые используются")

class MistralRequest(BaseModel):
    """Схема для входных данных запроса к Mistral"""
    text: str = Field(..., description="Текст запроса пользователя")
    combined_context: str = Field(..., description="Контекст для обработки запроса")
    chat_history: str = Field(..., description="История предыдущих сообщений в чате")
    input_type: Literal['voice', 'csv', 'text'] = Field(
        default='text',
        description="Тип входных данных: голос, csv или текст"
    )

class MistralResponse(BaseModel):
    mistral_response: str

class LoggData(BaseModel):
    user_id: int
    query: str
    mistral_response: str
    status: Literal[1, 0]
    hashes: List[str]

