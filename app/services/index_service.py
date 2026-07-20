from llama_index.core import StorageContext, load_index_from_storage
from app.config import config
from app.services.llm import configure_models


_query_engine = None


def _build_query_engine():
    configure_models()
    storage_context = StorageContext.from_defaults(persist_dir=config.storage_dir)
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(similarity_top_k=config.top_k)


def load_query_engine():
    global _query_engine
    if _query_engine is None:
        _query_engine = _build_query_engine()
    return _query_engine
