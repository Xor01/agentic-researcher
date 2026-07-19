from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from app.config import config

def configure_models() -> None:
    Settings.llm = OpenAI(model=config.llm_model, temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model=config.embedding_model)