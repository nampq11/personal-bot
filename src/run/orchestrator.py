import qdrant_client
from qdrant_client.models import VectorParams, Distance
from src.run.cfg import RunConfig


class RunOrchestrator:
    def __init__(self):
        pass

    def setup_db(cfg: RunConfig, qdrantdb: qdrant_client.QdrantClient):
        if not qdrantdb.collection_exists(cfg.db_collection):
            qdrantdb.create_collection(
                collection_name=cfg.db_collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )