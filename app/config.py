from os import environ
from dotenv import load_dotenv
from app.logger import logger

load_dotenv()

class Config():
    def __init__(self, llm_model: str, temperature: float, embedding_model: str, data_dir = None, storage_dir = None, top_k: int = 5, max_tool_calls: int = 10, model_provider=None):
        try:
            if 'OPENAI_API_KEY' not in environ:
                raise Exception('Missing OPENAI_API_KEY environment variable')
            self.llm_model = llm_model
            self.temperature = temperature
            self.embedding_model = embedding_model
            self.model_provider = model_provider
            self.data_dir = data_dir
            self.storage_dir = storage_dir
            self.top_k = top_k
            self.max_tool_calls = max_tool_calls
        except Exception as e:
            logger.error(e)

config = Config(
    llm_model='gpt-5.4',
    temperature=0,
    embedding_model='text-embedding-3-small',
    data_dir='./data',
    storage_dir='./storage',
    top_k=5,
    max_tool_calls=10
)
logger.info('config object has been created.')