# Install required library first:
# pip install langchain-openai httpx

from langchain_openai import ChatOpenAI
import httpx

# Disable SSL verification for testing (as required in Hackathon environment)
client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta",
    model="gemini-2.5-flash-lite",
    api_key="",   # Use the key provided during event
    http_client=client
)

response = llm.invoke("Hi")
print(response)
