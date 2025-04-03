from typing import List
from GPU_control import gpu_lock
import funcs
from pymilvus import Collection, CollectionSchema, connections
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
            if self.collection.num_entities > 0 and self.collection.has_index():
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
            if self.collection_name in collections:
                collection = Collection(self.collection_name)
                collection.drop()
                print(f"Коллекция {self.collection_name} была удалена.")

            self.collection = Collection(name=self.collection_name, schema=self.schema)
            print(f"Коллекция {self.collection_name} была создана")

        except MilvusException as e:
            print(f"Ошибка при проверке или удалении коллекции: {e}")

    def insert_data(self, data: List[dict], additional_fields: List[str] = [], batch_size=2):
        """Вставка данных в коллекцию с динамическим количеством дополнительных полей."""
        hashs, texts, embeddings_all = [], [], []
        additional_data = {field: [] for field in additional_fields}
        
        for topic in data:
            hashs.append(topic.get('hash', ''))
            text = topic.get('text', '')
            texts.append(text)
            
            for field in additional_fields:
                additional_data[field].append(str(topic.get(field, '')))

        with gpu_lock(timeout=30):
            with funcs.use_device(funcs.model, funcs.device):
                for i in range(0, len(texts), batch_size):
                    embeddings_all.extend(funcs.generate_embedding(texts[i:i+batch_size]))


        funcs.clear_gpu_memory()

        embeddings_all = normalize(embeddings_all, axis=1)
        data_to_insert = [hashs, embeddings_all, texts] + [additional_data[field] for field in additional_fields]

        self.collection.insert(data_to_insert)
        

    def search(self, query_text: str, additional_fields: List = None, limit=5):
        """Поиск по запросу с возвратом нужных полей"""

        with funcs.use_device(funcs.model, funcs.device):
            query_embedding = funcs.generate_embedding([f'query: {query_text}'])

        funcs.clear_gpu_memory()
        
        query_embedding = normalize(query_embedding, axis=1)

        output_fields = ["hash"] + (additional_fields if additional_fields else [])

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
    
    def data_release(self):
        "Вытаскивает из опретивки данные"
        self.collection.release()

    def drop_collection(self):
        """Удаление коллекции"""
        self.collection.drop()

    def connection_close(self):
        """Закрытие соединения"""
        connections.disconnect("default")
