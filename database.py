"""
Модуль для работы с базами данных MySQL, PostgreSQL и Milvus.
Содержит классы и методы для взаимодействия с коллекциями, выполнения запросов и обработки данных.
"""
from typing import List

from pymilvus import Collection, CollectionSchema, connections
from pymilvus import utility
from pymilvus.exceptions import MilvusException

from sklearn.preprocessing import normalize

import psycopg2
import mysql.connector

from GPU_control import gpu_lock
import funcs

# from config import mysql_config, postgres_config


class Milvus:
    """Класс для работы с коллекциями Milvus."""
    def __init__(self, host, port, collection_name, fields, index_params, search_params):
        """Инициализация подключения к Milvus и создание коллекции."""
        connections.connect(host=host, port=port)
        self.collection_name = collection_name
        self.fields = fields
        self.index_params = index_params
        self.schema = CollectionSchema(fields=self.fields)
        collections = utility.list_collections()

        if collection_name in collections:
            self.collection = Collection(self.collection_name)
            self.create_index()
        else:
            self.collection = Collection(name=self.collection_name, schema=self.schema)

        self.search_params = search_params

    def create_index(self):
        """Создание индекса для поля embedding."""
        self.collection.create_index(field_name='embedding', index_params=self.index_params)
        self.collection.load()
        self.collection.flush()

    def init_collection(self):
        """Инициализация коллекции, удаление существующей и создание новой."""
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

    def insert_data(self, data: List[dict], additional_fields = None, batch_size=2):
        """Вставка данных в коллекцию с динамическим количеством дополнительных полей."""
        if additional_fields is None:
            additional_fields = []
        hashs, texts, embeddings_all = [], [], []
        additional_data = {field: [] for field in additional_fields}

        for topic in data:
            hashs.append(topic.get('hash', ''))
            text = topic.get('text', '')
            texts.append(text[:20000])
            
            for field in additional_fields:
                additional_data[field].append(str(topic.get(field, '')))

        with gpu_lock():
            with funcs.use_device(funcs.model, funcs.device):
                for i in range(0, len(texts), batch_size):
                    embeddings_all.extend(funcs.generate_embedding(texts[i:i+batch_size]))

        funcs.clear_gpu_memory()
        embeddings_all = normalize(embeddings_all, axis=1)
        data_to_insert = [
            hashs,
            embeddings_all,
            texts,
            *[additional_data[field] for field in additional_fields]
        ]
        self.collection.insert(data_to_insert)

    def search(self, query_text: str, additional_fields = None, limit=5):
        """Поиск по запросу с возвратом нужных полей."""
        if additional_fields is None:
            additional_fields = []
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

    def clean_similar_vectors(self, similarity_threshold: float = 1):
        """Удаление похожих векторов из коллекции."""
        all_vectors = self.collection.query(expr='hash != "0"', output_fields=["embedding"])
        vectors = [item['embedding'] for item in all_vectors]
        ids = [item['hash'] for item in all_vectors]
        deleted_ids = set()

        for i, query_vector in enumerate(vectors):
            if ids[i] in deleted_ids:
                continue
            search_results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 200, "nprobe": 10}},
                limit=3
            )
            for result in search_results[0]:
                similarity = result.distance
                if similarity >= similarity_threshold and result.id != ids[i]:
                    deleted_ids.add(result.id)
                    self.collection.delete(expr=f"hash == '{result.id}'")
                    print(f"Удален вектор с ID: {result.id} с похожестью на {ids[i]} {similarity * 100:.2f}%")
        self.collection.flush()
        self.collection.load()
        return deleted_ids

    def get_indexes(self, collection_name):
        """Возвращает все индексы коллекции."""
        return utility.list_indexes(collection_name)

    def get_data_count(self):
        """Возвращает количество данных в коллекции."""
        return self.collection.num_entities

    def data_release(self):
        """Освобождение данных из оперативной памяти."""
        self.collection.release()

    def drop_collection(self):
        """Удаление коллекции."""
        self.collection.drop()

    def connection_close(self):
        """Закрытие соединения с Milvus."""
        connections.disconnect("default")


class MySQL:
    """Класс для работы с базой данных MySQL."""
    def __init__(self, host, port, user, password, database) -> None:
        """Инициализация подключения к MySQL."""
        self.conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.conn.cursor()

    def connection_close(self):
        """Закрытие соединения с MySQL."""
        self.cursor.close()
        self.conn.close()

    def get_pages_data(self):
        """Получение данных страниц из базы."""
        self.cursor.execute("""
        SELECT DISTINCT p.name, p.text, b.slug, p.slug, c.name
        FROM pages p
        JOIN books b ON p.book_id = b.id
        LEFT JOIN chapters c ON p.chapter_id = c.id
        JOIN bookshelves_books bb ON bb.book_id = b.id
        WHERE bb.bookshelf_id NOT IN (1, 10) AND p.`text` <> ''
        """)
        rows = self.cursor.fetchall()

        result = [
            {
                "page_name": row[0],
                "page_text": row[1],
                "book_slug": row[2],
                "page_slug": row[3],
                "chapter_name": row[4]
            }
            for row in rows
        ]
        return result


class PostgreSQL:
    """Класс для работы с базой данных PostgreSQL."""
    def __init__(self, host, port, user, password, database) -> None:
        """Инициализация подключения к PostgreSQL."""
        self.connection = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.connection.cursor()

    def insert_new_topic(self, topic_hash, title, text, user_id):
        """Вставка новой темы в базу данных."""
        query = '''
            INSERT INTO frida_storage (hash, title, text, isexstra)
            VALUES (%s, %s, %s, %s)
        '''
        self.cursor.execute(query, (topic_hash, title, text, True))

        exstra_query = '''
            INSERT INTO exstraTopics (hash, user_id)
            VALUES (%s, %s)
        '''
        self.cursor.execute(exstra_query, (topic_hash, user_id))
        self.connection.commit()

    def add_user_to_db(self, user_id: int, username: str, first_name: str, last_name: str):
        """Добавление нового пользователя в базу данных."""
        query = '''
            INSERT INTO users (user_id, username, first_name, last_name)
            VALUES (%s, %s, %s, %s)
        '''
        self.cursor.execute(query, (user_id, username, first_name, last_name))
        self.connection.commit()

    def log_message(self, user_id, user_query, response, response_status, topic_hashs: List[str]):
        """Логирование сообщения пользователя."""
        query = '''
            INSERT INTO bot_logs (user_id, query, response, response_status)
            VALUES (%s, %s, %s, %s)
            RETURNING log_id;
        '''
        self.cursor.execute(query, (user_id, user_query, response, response_status))
        result = self.cursor.fetchone()
        if result is not None:
            log_id = result[0]
        else:
            log_id = None

        for topic_hash in topic_hashs:
            hash_query = '''
                INSERT INTO bot_log_topic_hashes (log_id, topic_hash)
                VALUES (%s, %s)
            '''
            self.cursor.execute(hash_query, (log_id, topic_hash))
        self.connection.commit()

    def user_exists(self, user_id: int):
        """Проверка существования пользователя в базе данных."""
        query = '''
            SELECT 1 FROM users WHERE user_id = %s
        '''
        self.cursor.execute(query, (user_id,))
        result = self.cursor.fetchone()
        return result is not None

    def check_user_is_admin(self, user_id):
        """Проверка, является ли пользователь администратором."""
        query = '''
            SELECT is_admin FROM users WHERE user_id = %s
        '''
        self.cursor.execute(query, (user_id,))
        result = self.cursor.fetchone()
        return result[0] if result is not None else None

    def get_admins(self):
        """Получение списка администраторов."""
        query = '''
            SELECT user_id, username FROM users WHERE is_admin = TRUE;
        '''
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        return result

    def get_data_for_vector_db(self):
        """Получение данных для векторной базы."""
        query = "SELECT hash, book_name, title, text FROM frida_storage"
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        return result

    def get_history(self, user_id):
        """Получение истории сообщений пользователя."""
        query = '''
        WITH LastThreeLogs AS (
            SELECT *
            FROM bot_logs bl
            WHERE bl.user_id = %s
            ORDER BY bl.created_at DESC
            LIMIT 3
        )
        SELECT *
        FROM LastThreeLogs
        ORDER BY created_at ASC;
        '''
        self.cursor.execute(query, (user_id,))
        result = self.cursor.fetchall()
        return result

    def get_topics_texts_by_hashs(self, hashs: tuple[str]):
        """Получение текстов тем по их хэшам."""
        if not hashs:
            return []

        placeholders = ', '.join(['%s'] * len(hashs))
        query = f'''
            SELECT book_name, text, url
            FROM frida_storage fs
            WHERE fs.hash IN ({placeholders})
        '''
        try:
            self.cursor.execute(query, hashs)
            result = self.cursor.fetchall()
            return result
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            return []

    def delete_items_by_hashs(self, hashs):
        """Удаление элементов из базы данных по хэшам."""
        hashs = tuple(hashs)
        query = '''
            DELETE FROM frida_storage WHERE hash IN %s 
        '''
        self.cursor.execute(query, (hashs,))
        self.connection.commit()
        affected_rows = self.cursor.rowcount
        return affected_rows

    def get_count(self):
        """Получение количества записей в базе данных."""
        query = '''
            SELECT COUNT(*) FROM frida_storage fs2 
        '''
        self.cursor.execute(query)
        result = self.cursor.fetchone()
        return result[0] if result is not None else None

    def connection_close(self):
        """Закрытие соединения с PostgreSQL."""
        self.cursor.close()
        self.connection.close()
