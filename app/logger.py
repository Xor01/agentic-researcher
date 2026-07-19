import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='agentic-researcher.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)