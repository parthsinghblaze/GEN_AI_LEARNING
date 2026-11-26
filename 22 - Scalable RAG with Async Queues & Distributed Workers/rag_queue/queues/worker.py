import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# 1. Load environment variables
load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="stock_rag",
    embedding=embedding_model,
    url="https://99adacd7-1b47-4e20-963d-1915a1792863.ap-northeast-2-0.aws.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def process_query(query: str):
    """RQ background worker function to run RAG retrieval and LLM generation."""
    print("Searching Chunks", query)
    search_results = vector_store.similarity_search(query=query, k=4)

    context_blocks = []
    for doc in search_results:
        page_num = doc.metadata.get("page", "Unknown")
        source_file = doc.metadata.get("source", "stock.pdf")
        context_blocks.append(
            f"[Source: {source_file} | Page: {page_num}]\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = f"""You are a helpful and factual financial assistant.
    Answer the user's question based strictly on the available context from the book below.
    Always cite the exact page number(s) where the answer was found so the user can cross-verify.
    If the information is not contained in the context, politely state that the available pages do not mention it.
    
    AVAILABLE CONTEXT FROM THE BOOK:
    {context}
    """

    response = client.chat.completions.create(
                    model="gemini-3.5-flash-lite",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ]
                )

    return response.choices[0].message.content