# milvus_db.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION

EMBEDDING_DIM = 512

def get_or_create_collection():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(MILVUS_COLLECTION):
        collection = Collection(MILVUS_COLLECTION)
        collection.load()
        print(f"[Milvus] ✅ Loaded existing collection: {MILVUS_COLLECTION}")
        return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64,  is_primary=True, auto_id=True),
        FieldSchema(name="global_id", dtype=DataType.INT64),
        FieldSchema(name="track_id", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,  dim=EMBEDDING_DIM),
        FieldSchema(name="quality_score", dtype=DataType.FLOAT),

        FieldSchema(name="gender", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="age", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="race", dtype=DataType.VARCHAR, max_length=20),
    ]
    schema     = CollectionSchema(fields, description="Visitor face embeddings")
    collection = Collection(name=MILVUS_COLLECTION, schema=schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type":  "IVF_FLAT",
        "params":      {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()

    print(f"[Milvus] ✅ Created new collection: {MILVUS_COLLECTION}")
    return collection