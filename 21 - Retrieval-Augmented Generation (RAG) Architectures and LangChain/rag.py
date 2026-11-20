import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# 1. Define the path to your PDF file
pdf_path = Path(__file__).parent / "stock.pdf"

# 2. Initialize the PyPDFLoader and load only the first 50 pages
print("📄 Loading first 50 pages from stock.pdf...")
loader = PyPDFLoader(file_path=str(pdf_path))

docs = []
for i, page in enumerate(loader.lazy_load()):
    if i >= 50:
        break
    docs.append(page)

print(f"✅ Loaded {len(docs)} pages.")

# 3. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
chunks = text_splitter.split_documents(docs)
print(f"✂️ Created {len(chunks)} chunks from 50 pages.")

# 4. Initialize Google's Gemini embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# 5. Ingest into Qdrant Cloud in rate-limit-safe batches (Gemini Free Tier limit: 100 RPM)
batch_size = 50
total_batches = (len(chunks) + batch_size - 1) // batch_size

print(f"\n⏳ Starting upload to Qdrant Cloud ({total_batches} batches)...")

# Batch 1 creates / connects to the Qdrant collection
print(f"🚀 Uploading Batch 1/{total_batches} ({min(batch_size, len(chunks))} chunks)...")
vector_store = QdrantVectorStore.from_documents(
    documents=chunks[:batch_size],
    embedding=embedding_model,
    url="https://99adacd7-1b47-4e20-963d-1915a1792863.ap-northeast-2-0.aws.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="stock_rag"
)
print("✅ Batch 1 uploaded!")

# Remaining batches uploaded with a 30s delay to stay safely under 100 RPM
for batch_idx, i in enumerate(range(batch_size, len(chunks), batch_size), start=2):
    print(f"⏳ Waiting 30s to respect Gemini Free Tier 100 RPM quota...")
    time.sleep(30)
    batch = chunks[i : i + batch_size]
    print(f"🚀 Uploading Batch {batch_idx}/{total_batches} ({len(batch)} chunks)...")
    vector_store.add_documents(batch)
    print(f"✅ Batch {batch_idx} uploaded!")

print("\n🎉 Indexing done! All 50 pages are stored in Qdrant Cloud collection 'stock_rag'.")
