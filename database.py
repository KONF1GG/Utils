from typing import List
import funcs
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from sklearn.preprocessing import normalize
from pymilvus import utility
from pymilvus.exceptions import MilvusException

class Milvus:
    def __init__(self, host, port, collection_name, fields, index_params, search_params):
        # Подключаемся к Milvus
        connections.connect(host=host, port=port)

        self.collection_name = collection_name


        self.fields = fields

        self.index_params = index_params

        # Схема коллекции
        self.schema = CollectionSchema(fields=self.fields)

        # Проверяем, существует ли коллекция
        collections = utility.list_collections()

        if collection_name in collections:
            self.collection = Collection(self.collection_name)
            if self.collection.num_entities > 0:
                self.collection.load()
        else:
            self.collection = Collection(name=self.collection_name, schema=self.schema)

        # Параметры поиска
        self.search_params = search_params

    def create_index(self):
        """Создание индекса для поля embedding"""
        self.collection.create_index(field_name='embedding', index_params=self.index_params)
        self.collection.load()
        self.collection.flush()

    def init_collection(self):
        """Инициализация коллекции, если она уже существует, она будет удалена"""
        try:
            collections = utility.list_collections()
            if collections:
                collection = Collection(self.collection_name)
                collection.drop()
                print(f"Коллекция {self.collection_name} была удалена.")

            self.collection = Collection(name=self.collection_name, schema=self.schema)
            print(f"Коллекция {self.collection_name} была создана")

        except MilvusException as e:
            print(f"Ошибка при проверке или удалении коллекции: {e}")

    def insert_data(self, data: List, additional_fields: dict = None):
        """Вставка данных в коллекцию"""
        hashs, texts, embeddings, other_fields = [], [], [], {}

        for topic in data:
            hashs.append(topic.get('hash'))
            text = topic.get('text')
            texts.append(text)
            embeddings.append(funcs.generate_embedding(text))

            # Динамическое добавление других полей
            for field_name, field_value in (additional_fields or {}).items():
                if field_name not in other_fields:
                    other_fields[field_name] = []
                other_fields[field_name].append(field_value)

        embeddings = normalize(embeddings, axis=1)

        # Создаем список для вставки, где каждый элемент — отдельный список
        data_to_insert = [hashs, embeddings, texts] + list(other_fields.values())

        # Вставляем данные
        self.collection.insert(data_to_insert)
        self.create_index()



    def search(self, query_text: str, additional_fields: List = None, limit=5):
        """Поиск по запросу с возвратом всех полей"""
        query_embedding = funcs.generate_embedding(query_text)
        query_embedding = normalize([query_embedding], axis=1)

        # Формирование полей для поиска
        output_fields = ["hash"] + (additional_fields if additional_fields else [])

        # Поиск в коллекции
        results = self.collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=self.search_params,
            limit=limit, 
            output_fields=output_fields
        )

        return results



    def get_data_count(self):
        """Возвращает количество данных в коллекции"""
        return self.collection.num_entities

    def drop_collection(self):
        """Удаление коллекции"""
        self.collection.drop()

    def connection_close(self):
        """Закрытие соединения"""
        connections.disconnect("default")
