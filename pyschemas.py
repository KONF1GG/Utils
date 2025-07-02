"""
Модели данных для API, используемые в приложении.
Содержит схемы для запросов, ответов и логирования.
"""

from typing import List, Literal
from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    """Ответ с указанием статуса выполнения."""
    status: Literal['success', 'error']


class AddressModel(BaseModel):
    """Модель данных для адреса пользователя."""
    login: str
    address: str
    houseId: str


class PromtModel(BaseModel):
    """Модель данных для промта."""
    id: str
    name: str
    template: str
    params: str


class ResponseCategory(BaseModel):
    """Модель данных для категории ответа."""
    category: str


class Input(BaseModel):
    """Модель данных для входного текста запроса."""
    text_query: str


class Count(BaseModel):
    """Модель данных для количества элементов."""
    count: int


class Page(BaseModel):
    """Модель данных для страницы книги."""
    title: str
    text: str
    book_name: str | None = None
    book_slug: str
    page_slug: str


class UserData(BaseModel):
    """Модель данных пользователя."""
    user_id: int
    firstname: str
    lastname: str = ""
    username: str = ""


class SearchParams(BaseModel):
    """Модель данных для параметров поиска."""
    user_id: int
    text: str


class SearchResponseData(BaseModel):
    """Ответ поиска с контекстом и историей диалога."""
    combined_context: str = Field(..., description="Контекст Вики")
    chat_history: str = Field(..., description="Отформатированная история диалога")
    hashs: List[str] = Field(..., description="ID контекстов которые используются")


class Search2ResponseData(BaseModel):
    """Ответ поиска с контекстом без истории диалога."""
    combined_context: str = Field(..., description="Контекст Вики")
    hashs: List[str] = Field(..., description="ID контекстов которые используются")


class AIRequest(BaseModel):
    """Схема для входных данных запроса к AI."""
    text: str = Field(..., description="Текст запроса пользователя")
    combined_context: str = Field(..., description="Контекст для обработки запроса")
    chat_history: str = Field(..., description="История предыдущих сообщений в чате")
    input_type: Literal['voice', 'csv', 'text'] = Field(
        default='text',
        description="Тип входных данных: голос, csv или текст"
    )
    model: str = Field(
        default='mistral-large-latest',
        description="Название модели AI, которая будет использоваться для обработки запроса"
    )


class AIResponse(BaseModel):
    """Ответ от модели AI."""
    ai_response: str


class LoggData(BaseModel):
    """Модель данных для логирования запросов."""
    user_id: int
    query: str
    ai_response: str
    status: Literal[1, 0]
    hashes: List[str]


class AddTopicRequest(BaseModel):
    """Запрос для добавления новой темы."""
    title: str = Field(..., description="Заголовок темы")
    text: str = Field(..., description="Текст темы")
    user_id: int = Field(..., description="ID пользователя")

