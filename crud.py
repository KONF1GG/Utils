import requests
import json
import logging
from milvus_schemas import address_schema, address_index_params, address_search_params, category_schema, category_index_params, category_search_params
from sympy import ilcm
from database import Milvus
import config
import funcs
from bs4 import BeautifulSoup

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def get_addresses():
    url = 'https://ws.freedom1.ru/redis/raw?query=FT.SEARCH%20idx:adds%20%27@settlementId:[10193%2010193]%20@searchType:{house}%27%20Limit%200%205000&pretty=1'
    try:
        logger.info("Запрос к URL: %s", url)
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            pre_tag = soup.find('pre')

            if pre_tag:
                try:
                    # Загружаем JSON данные из текста в теге <pre>
                    data = json.loads(pre_tag.get_text())

                    # Создаем новый список словарей с 'title' и 'hash'
                    result = [{'address': value.get('title', ''), 'hash': key} for key, value in data.items()]
                    logger.info("Данные успешно получены и обработаны.")
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка при парсинге JSON: {e}")
            else:
                logger.warning("Тег <pre> не найден в ответе.")
        else:
            logger.error(f"Ошибка: Запрос не успешен. Статус код {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Ошибка при выполнении запроса: {e}")

    
    return []

def insert_addresses_to_milvus():
    logger.info("Инициализация соединения с Milvus.")
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Address', address_schema, address_index_params, address_search_params)
    milvus_db.init_collection()

    data = get_addresses()
    if not data:
        logger.warning("Не удалось получить адреса.")
        return

    formatted_data = []
    for entry in data:
        address = entry['address']
        uuid = entry['hash']
        
        formatted_data.append({
            'hash': uuid,
            'text': address
        })
    
    additional_fields = {'address': address}
    milvus_db.insert_data(formatted_data, additional_fields)
    logger.info("Адреса успешно загружены в Milvus.")
    milvus_db.connection_close()

def insert_categories_to_milvus():
    logger.info("Инициализация соединения с Milvus.")
    milvus_db = Milvus(config.MILVUS_HOST, config.MILVUS_PORT, 'Categories', category_schema, category_index_params, category_search_params)
    milvus_db.drop_collection()
    milvus_db.init_collection()

    # Данные категорий
    categories = [
        "Отсутствие интернета",
        "Оплата/Баланс",
        "Личный кабинет",
        "Оборудование",
        "Телевидение",
        "Тарифы",
        "Подключение",
        "Видеонаблюдение",
        "Домофония",
        "Сервисный выезд",
        "Неопределено/Категория неопределена",
        "Расторжение договора",
        "Переезд",
        "Приветствие",
        "Скорость/Проверка скорости",
        "Повышение стоимости"
    ]

    formatted_data = []
    for category in categories:
        formatted_data.append({
            'hash': funcs.generate_hash(category), 
            'text': category,
        })
        
    milvus_db.insert_data(formatted_data, additional_fields={}) 
    logger.info("Категории успешно загружены в Milvus.")
    milvus_db.connection_close()


