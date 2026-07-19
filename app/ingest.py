from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from app.config import config
from app.services.llm import configure_models
from app.logger import logger
def build_index() -> None:
    try:
        logger.info('Building data ingestion started.')
        configure_models()
        data_path = Path(config.data_dir)
        storage_path = Path(config.storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)
        documents = SimpleDirectoryReader(str(data_path), recursive=True).load_data()
        if not documents:
            raise RuntimeError("No documents found in the data directory.")
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=str(storage_path))
        print(f"Indexed {len(documents)} document(s) into {storage_path}.")
    except Exception as e:
        print("Error while building data ingestion.")
        logger.error('Error while building data ingestion.')
    else:
        logger.info('Building data ingestion finished with no errors.')


if __name__ == "__main__":
    build_index()
# 5 json files were created.