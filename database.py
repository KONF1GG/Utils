from typing import List
import funcs
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from sklearn.preprocessing import normalize
from pymilvus import utility
from pymilvus.exceptions import MilvusException

class Milvus:
    def __init__(self, host, port, collection_name):
        # Подключаемся к Milvus
        connections.connect("default", host, port)

        self.collection_name = collection_name

        self.fields = [
            FieldSchema(name='hash', dtype=DataType.VARCHAR, is_primary=True, max_length=255),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=512), 
            FieldSchema(name='address', dtype=DataType.VARCHAR, max_length=255),  
        ]

        # Параметры индекса
        self.index_params = {
            "index_type": "HNSW",  
            "metric_type": "L2",  
            "params": {"M": 16, "efConstruction": 300}
        }

        # Схема коллекции
        self.schema = CollectionSchema(fields=self.fields, description="Addresses")

        # Проверяем, существует ли коллекция
        collections = utility.list_collections()

        if collection_name in collections:
            self.collection = Collection(self.collection_name)
            if self.collection.num_entities > 0:
                self.collection.load()
        else:
            self.collection = Collection(name=self.collection_name, schema=self.schema)

        # Параметры поиска
        self.search_params = {"metric_type": "L2", "params": {"ef": 200, "nprobe": 10}}

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

    def insert_data(self, data: List):
        """Вставка данных в коллекцию"""
        hashs, embeddings, addresses = [], [], []

        for topic in data:
            hashs.append(topic.get('hash'))
            text = topic.get('text')
            embeddings.append(funcs.generate_embedding(text + ' ' + text.split()[-1] + ' ' + text.split()[-1])) 
            addresses.append(text)
        # Нормализуем embeddings
        embeddings = normalize(embeddings, axis=1)

        # Вставляем данные
        self.collection.insert([hashs, embeddings, addresses])
        self.create_index()

    def search(self, query_text: str):
        """Поиск по запросу с возвратом всех полей"""
        query_embedding = funcs.generate_embedding(query_text)
        query_embedding = normalize([query_embedding], axis=1)

        output_fields = ["hash", "address"] 

        # Поиск в коллекции
        results = self.collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=self.search_params,
            limit=5, 
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
