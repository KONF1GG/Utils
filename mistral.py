"""Взаимодействие с моделью Mistral."""

import asyncio
import logging
import json
from typing import Literal
from fastapi import HTTPException
from mistralai import Mistral
from config import API_KEY

logger = logging.getLogger(__name__)

async def get_mistral(
    query: str,
    context: str = "",
    history: str = "",
    input_type: Literal['voice', 'csv', 'text'] = 'text'
) -> str:
    """
    Асинхронная функция для взаимодействия с моделью Mistral.
    Формирует запрос на основе типа ввода и возвращает ответ модели.

    :param query: Запрос пользователя.
    :param context: Контекст для анализа (например,
      текст таблицы или расшифровка голосового сообщения).
    :param history: История диалога.
    :param input_type: Тип ввода ('voice', 'csv', 'text').
    :return: Ответ модели в виде строки.
    """
    model = "mistral-large-latest"

    # Формирование промптов для различных типов ввода
    prompts = {
        'csv': (
            "Ты - Фрида, бот помощник. Обработай файл таблицы по запросу. "
            "Если нет вопроса, то просто опиши таблицу. Используй HTML теги где нужно что-то выделить. "
            "Делай текст хорошо структурированным и понятным. НЕ ИСПОЛЬЗУЙ MARKDOWN. "
            "Только эти теги HTML (<b>, <i>, <a>, <code>, <pre>) НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ <ul> и <br>! "
            "Отвечай четко и кратко на вопрос и только на русском.\n"
            f"Вопрос: {query}\nТаблица: {context}"
        ),
        'voice': (
            "Ты - Фрида, бот помощник. Твоя задача проанализировать вопрос и контекст звукового файла. "
            "Учитывай, что текст может содержать ошибки, поскольку был обработан из голосового сообщения. "
            "Если вопроса нет, отвечай согласно тексту голосового сообщения. Используй HTML теги где нужно что-то выделить. "
            "Делай текст хорошо структурированным и понятным. НЕ ИСПОЛЬЗУЙ MARKDOWN. "
            "Только эти теги HTML (<b>, <i>, <a>, <code>, <pre>) НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ <ul> и <br>! "
            "Отвечай четко и кратко на вопрос и только на русском.\n"
            f"Вопрос: {query}\nРасшифрованное голосовое сообщение: {context}\nИстория диалога: {history}"
        ),
        'text': (
            "Ты — Фрида, бот-помощник компании Фридом. Твоя задача — отвечать на вопросы сотрудников компании, "
            "основываясь на предоставленных контекстах из корпоративной WIKI, содержащих важную информацию из статей.\n\n"
            "Инструкции:\n"
            "1. Если ответ есть в контексте — дай краткий и точный ответ.\n"
            "2. Если нет — используй знания, но укажи, что это не точная информация.\n"
            "3. Не выдумывай факты.\n"
            "4. Используй HTML теги (<b>, <i>, <a>, <code>, <pre>), но не <ul> и <br>.\n\n"
            f"ТЕКУЩИЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {query}\n\nТекст для анализа: {context}\n\nИстория диалога: {history}"
        )
    }

    message_content = prompts.get(input_type, prompts['text'])

    retries = 3
    delay = 2
    last_error = None

    for attempt in range(retries):
        try:
            async with Mistral(api_key=API_KEY) as client:
                response = await client.chat.complete_async(
                    model=model,
                    messages=[{"role": "user", "content": message_content}]
                )
                response_content = response.choices[0].message.content

                if isinstance(response_content, dict):
                    response_content = json.dumps(response_content, ensure_ascii=False)

                return str(response_content)

        except (asyncio.TimeoutError, ConnectionError, ValueError) as e:
            last_error = e
            logger.error("[Попытка %d] Ошибка при запросе к Mistral: %s", attempt + 1, str(e))
            await asyncio.sleep(delay)

    raise HTTPException(
        status_code=500,
        detail={
            "status": "error",
            "message": "Не удалось получить ответ от Mistral после нескольких попыток",
            "error": str(last_error)
        }
    )
