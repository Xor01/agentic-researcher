from os import environ
from dotenv import load_dotenv
from app.logger import logger

load_dotenv()

class Config():
    def __init__(self, llm_model: str, temperature: float, embedding_model: str, model_provider=None):
        try:
            if 'OPENAI_API_KEY' not in environ:
                raise Exception('Missing OPENAI_API_KEY environment variable')
            self.llm_model = llm_model
            self.temperature = temperature
            self.embedding_model = embedding_model
            self.model_provider = model_provider
        except Exception as e:
            logger.error(e)

config = Config(
    llm_model='gpt-5.4',
    temperature=0,
    embedding_model='text-embedding-3-small'
)
logger.info('config object has been created.')