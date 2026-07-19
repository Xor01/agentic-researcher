from llama_index.core import StorageContext, load_index_from_storage
from app.config import config
from app.services.llm import configure_models


def load_query_engine():
    configure_models()
    storage_context = StorageContext.from_defaults(persist_dir=config.storage_dir)
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(similarity_top_k=config.top_k)