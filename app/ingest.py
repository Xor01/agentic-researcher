from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

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

        for document in documents:
            file_name = document.metadata.get("file_name", "unknown")
            document.metadata["source_type"] = Path(file_name).suffix.lower()
            document.metadata["collection"] = "bootcamp_knowledge"

        splitter = SentenceSplitter(chunk_size=700, chunk_overlap=100)
        nodes = splitter.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, show_progress=True)
        index.storage_context.persist(persist_dir=str(storage_path))
        print(f"Indexed {len(documents)} document(s) as {len(nodes)} chunks into {storage_path}.")
    except Exception as e:
        print("Error while building data ingestion.")
        logger.error('Error while building data ingestion.')
    else:
        logger.info('Building data ingestion finished with no errors.')


if __name__ == "__main__":
    build_index()
