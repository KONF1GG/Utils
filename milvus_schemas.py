from pymilvus import DataType, FieldSchema


address_schema = [FieldSchema(name='hash', dtype=DataType.VARCHAR, is_primary=True, max_length=255),
 FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=512), 
 FieldSchema(name='address', dtype=DataType.VARCHAR, max_length=255),  ]

address_index_params = {
            "index_type": "HNSW",  
            "metric_type": "L2",  
            "params": {"M": 16, "efConstruction": 300}
        }

address_search_params = {"metric_type": "L2", "params": {"ef": 200, "nprobe": 10}}


category_schema = [FieldSchema(name='hash', dtype=DataType.VARCHAR, is_primary=True, max_length=255),
 FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=512), 
 FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=255),  ]

category_index_params = {
            "index_type": "HNSW",  
            "metric_type": "IP",  
            "params": {"M": 16, "efConstruction": 300}
        }

category_search_params = {"metric_type": "IP", "params": {"ef": 200, "nprobe": 10}}