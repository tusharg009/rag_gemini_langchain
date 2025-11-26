import json
from google.genai import Client
from google.genai.types import GenerateContentResponse


# Initialize Gemini client
client = Client(api_key="INSERT_API_KEY_HERE")


def process_gemini_query(input_json: dict) -> str:
    """
    Takes the search result JSON + query,
    Extracts the document chunks,
    Sends them to Gemini for answering,
    Returns the final answer as a string.
    """

    try:
        # Validate main keys
        if "results" not in input_json:
            return "Invalid input: 'results' field missing."

        results = input_json["results"]

        if "documents" not in results:
            return "Invalid input: 'documents' field missing."

        if "query" not in input_json:
            return "Invalid input: 'query' field missing."

        user_query = input_json["query"]

        # Extract documents safely
        documents_list = results.get("documents", [])

        if not documents_list or not isinstance(documents_list, list):
            return "No documents found."

        # documents is a list of lists, so take [0]
        doc_chunks = documents_list[0] if len(documents_list) > 0 else []

        if not doc_chunks:
            return "No document chunks available to answer the query."

        # Build context text
        context = "\n\n".join(doc_chunks)

        prompt = f"""
Use ONLY the following context to answer the question.

Context:
{context}

Question:
{user_query}

Give the best possible answer strictly based on the context.
"""

        # Call Gemini
        try:
            response: GenerateContentResponse = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )
        except Exception as api_error:
            return f"Gemini API Error: {str(api_error)}"

        # Ensure response contains text
        answer = getattr(response, "text", None)
        if not answer:
            return "Gemini returned an empty response."

        return answer.strip()

    except Exception as e:
        return f"Unhandled error: {str(e)}"


# ----------------- TEST EXAMPLE --------------------
if __name__ == "__main__":
    request_payload = {
        "status": "success",
        "results": {
            "ids": [["0922824d-9d4b-4170-934b-ea58382f08d0_2"]],
            "embeddings": None,
            "documents": [[
                "The field of AI research was founded at a workshop held on the campus of Dartmouth College in 1956."
            ]],
            "uris": None,
            "included": ["metadatas", "documents", "distances"],
            "data": None,
            "metadatas": [[{
                "chunk_index": 2,
                "filename": "document-aihostory-short.txt",
                "doc_id": "0922824d-9d4b-4170-934b-ea58382f08d0"
            }]],
            "distances": [[0.3447432518005371]]
        },
        "query": "Where was The field of AI research founded?"
    }

    result = process_gemini_query(request_payload)
    print("\nGemini Answer:", result)
