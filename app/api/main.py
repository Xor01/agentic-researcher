from llama_cloud import LlamaCloud
from dotenv import load_dotenv

load_dotenv()

client = LlamaCloud()  # reads LLAMA_CLOUD_API_KEY from the environment

# Upload and parse a document
file = client.files.create(file="./test-data/attention_is_all_you_need.pdf", purpose="parse")
result = client.parsing.parse(
    file_id=file.id,
    tier="agentic",
    version="latest",
    expand=["markdown"],
)

# Print the markdown for the first page
print(result.markdown.pages[0].markdown)