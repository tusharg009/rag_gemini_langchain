# Install required library first:
# pip install langchain-openai httpx

from langchain_openai import ChatOpenAI
import httpx

# Disable SSL verification for testing (as required in Hackathon environment)
client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="XXXXXXXXXXX",   # Use the key provided during event
    http_client=client
)

response = llm.invoke("Hi")
print(response)
