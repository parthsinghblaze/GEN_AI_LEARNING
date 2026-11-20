import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# 1. Load environment variables
load_dotenv()

# 2. Initialize the exact same Gemini embedding model used during indexing
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# 3. Connect to your existing Qdrant Cloud collection
print("🔌 Connecting to Qdrant Cloud collection 'stock_rag'...")
vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="stock_rag",
    embedding=embedding_model,
    url="https://99adacd7-1b47-4e20-963d-1915a1792863.ap-northeast-2-0.aws.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)

# 4. Initialize Gemini chat client using the OpenAI compatibility format (as in hello_world/gemini.py)
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def chat():
    print("=" * 65)
    print("📚 Stock Market RAG Chat Assistant (Powered by Gemini & Qdrant)")
    print("Type your question below, or type 'exit' / 'quit' to leave.")
    print("=" * 65)

    while True:
        try:
            user_query = input("\n👤 You: ").strip()
            if not user_query:
                continue

            if user_query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! 👋")
                break

            print("🔍 Searching Qdrant database for relevant information...")
            
            # 5. Perform similarity search in Qdrant
            search_results = vector_store.similarity_search(query=user_query, k=4)

            print(f"search_results {search_results}")
            # 6. Build context with page numbers and source metadata
            context_blocks = []
            for doc in search_results:
                page_num = doc.metadata.get("page", "Unknown")
                source_file = doc.metadata.get("source", "stock.pdf")
                context_blocks.append(
                    f"[Source: {source_file} | Page: {page_num}]\n{doc.page_content}"
                )

            context = "\n\n---\n\n".join(context_blocks)

            # 7. Construct augmented system prompt
            system_prompt = f"""You are a helpful and factual financial assistant.
Answer the user's question based strictly on the available context from the book below.
Always cite the exact page number(s) where the answer was found so the user can cross-verify.
If the information is not contained in the context, politely state that the available pages do not mention it.

AVAILABLE CONTEXT FROM THE BOOK:
{context}
"""

            # 8. Query Gemini via OpenAI-compatible endpoint
            response = client.chat.completions.create(
                model="gemini-3.5-flash-lite",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )

            print("\n🤖 Gemini:")
            print(response.choices[0].message.content)

        except KeyboardInterrupt:
            print("\nSession ended. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    chat()
