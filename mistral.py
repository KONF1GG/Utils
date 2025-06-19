import asyncio
import time
from typing import Literal
from mistralai import Mistral
from fastapi import HTTPException
import logging
from config import API_KEY

logger = logging.getLogger(__name__)

async def get_mistral(
    query: str,
    context: str = "",
    history: str = "",
    input_type: Literal['voice', 'csv', 'text'] = 'text'
) -> str:
    model = "mistral-large-latest"
    # Формирование промптов
    prompt_for_file = (
        "Ты - Фрида, бот помощник. Обработай файл таблицы по запросу. "
        "Если нет вопроса то просто опиши таблицу. Используй HTML теги где нужно что-то выделить. "
        "Делай текст хорошо структурированным и понятным. НЕ ИСПОЛЬЗУЙ MARKDOWN. "
        "Только эти теги HTML (<b>, <i>, <a>, <code>, <pre>) НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ <ul> и <br>!"
        " Отвечай четко и кратко на вопрос и только на русском.\n"
        "Вопрос: " + query + "\n"
        "Таблица: " + context
    )

    prompt_for_voice = (
        "Ты - Фрида, бот помощник. Твоя задача проанализировать вопрос и контекст звукового файла. "
        "Учитывай, что текст может содержать ошибки, поскольку был обработан из голосового сообщения. "
        "Если вопроса нет, отвечай согласно тексту голосового сообщения. Используй HTML теги где нужно что-то выделить. "
        "Делай текст хорошо структурированным и понятным. НЕ ИСПОЛЬЗУЙ MARKDOWN. "
        "Только эти теги HTML (<b>, <i>, <a>, <code>, <pre>) НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ <ul> и <br>!\n"
        "Отвечай четко и кратко на вопрос и только на русском.\n"
        "Вопрос: " + query + "\n"
        "Расшифрованное голосовое сообщение: " + context + "\n"
        "История диалога: " + history
    )

    prompt = (
        "Ты — Фрида, бот-помощник компании Фридом. Твоя задача — отвечать на вопросы сотрудников компании, "
        "основываясь на предоставленных контекстах из корпоративной WIKI, содержащих важную информацию из статей.\n\n"

        "Инструкции:\n"
        "1. Если ответ есть в контексте — дай краткий и точный ответ.\n"
        "2. Если нет — используй знания, но укажи, что это не точная информация.\n"
        "3. Не выдумывай факты.\n"
        "4. Используй HTML теги (<b>, <i>, <a>, <code>, <pre>), но не <ul> и <br>.\n\n"

        "ТЕКУЩИЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: " + query + "\n\n"
        "Текст для анализа: " + context + "\n\n"
        "История диалога: " + history
    )

    message_content = (
        prompt_for_file if input_type == 'csv'
        else prompt_for_voice if input_type == 'voice'
        else prompt
    )

    retries = 3
    delay = 2
    last_error = None

    for attempt in range(retries):
        try:
            async with Mistral(api_key=API_KEY) as client:
                response = await client.chat.complete_async(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": message_content
                    }]
                )
                # Обработка результата
                response_content = response.choices[0].message.content

                if isinstance(response_content, dict):
                    import json
                    response_content = json.dumps(response_content, ensure_ascii=False)

                return str(response_content)

        except Exception as e:
            last_error = e
            logger.error(f"[Попытка {attempt+1}] Ошибка при запросе к Mistral: {str(e)}")
            await asyncio.sleep(delay)

    raise HTTPException(
        status_code=500,
        detail={
            "status": "error",
            "message": "Failed to get response from Mistral after multiple attempts",
            "error": str(last_error)
        }
    )