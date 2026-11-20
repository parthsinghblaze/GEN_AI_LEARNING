# 1. Intro to RAG & LangChain — Section Overview

## What is RAG (Retrieval-Augmented Generation)?

Large Language Models (LLMs) are exceptionally powerful at reasoning and language understanding, but they have critical limitations:
1. **Knowledge Cutoff:** They are trained on static historical data and know nothing about recent events.
2. **No Private/Internal Knowledge:** They have zero awareness of private company data, proprietary PDFs, internal wikis, or customer databases.
3. **Hallucination:** When asked about facts they don't know, they frequently make up plausible-sounding false answers.

**RAG (Retrieval-Augmented Generation)** is the industry-standard architecture that bridges this gap. It grounds the LLM in external, factual knowledge without requiring expensive model retraining or fine-tuning.

---

### Breaking Down the Term: R.A.G.

| Component | Meaning | What Happens Behind the Scenes |
| :--- | :--- | :--- |
| **Retrieval** | Fetching relevant data | When a user asks a question, the system searches an external knowledge source (vector database, document store, or search index) to retrieve only the most relevant text passages. |
| **Augmented** | Enriching the context | The retrieved data is injected directly into the LLM's prompt alongside the user's original query. |
| **Generation** | Synthesizing the final answer | The LLM reads the augmented context and generates a precise, factual, and hallucination-free response grounded in the provided source material. |

---

### The High-Level RAG Architecture

```mermaid
flowchart LR
    A["User Query"] --> B["1. Retrieval Engine"]
    D[("Knowledge Base / Vector DB<br/>(PDFs, Docs, DB)")] --> B
    B --> C["Retrieved Context Chunks"]
    A --> E["2. Augment Prompt<br/>(Query + Context)"]
    C --> E
    E --> F["3. LLM (Gemini / OpenAI)"]
    F --> G["Grounded, Factual Output"]
```

---

## Why is RAG the Dominant Pattern in Enterprise AI?

> *"RAG is something that is 90% done in big companies and industries."*

In commercial and enterprise environments, RAG is by far the most deployed GenAI pattern because:

1. **Cost & Speed vs. Fine-Tuning:**
   - Fine-tuning an LLM costs thousands of dollars and takes hours or days.
   - With RAG, updating your knowledge base is instantaneous: simply add, update, or delete documents in your database.
2. **Verifiable Sources & Citations:**
   - RAG allows systems to cite exact document names, page numbers, and paragraphs so humans can verify answers.
3. **Data Security & Access Control:**
   - Enterprises can restrict document access based on user permissions (e.g., HR documents only retrieved for HR staff).
4. **Foundation for Agentic Workflows:**
   - In modern Agentic AI, RAG is treated as a core **Tool**. An autonomous agent can evaluate a user's request, decide when it needs more knowledge, query a vector store, read the retrieved context, and take action.

---

## What We Will Cover in this Section

Throughout this section, we move from conceptual understanding to building **production-grade, scalable RAG pipelines**:

* **Core RAG Architecture:** How embeddings, chunking, and vector databases work together.
* **Working with Real-World Documents:** Processing large, unstructured documents and multi-page PDF files.
* **Scalable Asynchronous Pipelines:** Building async, non-blocking RAG systems capable of handling production traffic.
* **Production Best Practices:** Writing robust, reliable code using industry-standard tools like LangChain.

---

# 2. Defining the Core Problem in RAG Systems

## The Real-World Business Scenario
Imagine you are building an AI solution for an enterprise—such as a legal firm, hospital, financial institution, or insurance company.

- **Vast Knowledge Base:** The organization possesses thousands of PDF documents, legal contracts, case files, technical manuals, or customer records.
- **Human Inefficiency:** Employees spend hours manually opening files, skimming word-by-word, and searching for critical information (e.g., *"What were the settlement terms in Case #32?"*).
- **The Desired Experience:** Employees want a conversational AI assistant where they can simply ask questions in plain English and instantly receive factual, accurate answers.

---

## The Two Core Bottlenecks of Base LLMs

When attempting to build this using a standard LLM (OpenAI GPT, Google Gemini, Anthropic Claude), you immediately run into two massive hurdles:

### 1. The Knowledge Deficit (Private Data Blindness)
Base LLMs are trained strictly on publicly available internet data up to a fixed training cutoff date.
- They have **zero awareness** of internal corporate documents, confidential legal files, or proprietary database records.
- If an employee asks: *"Tell me about Case #32"*, the LLM will either admit it doesn't know, or worse, **hallucinate** a fictional case with made-up parties and fake rulings.

### 2. The Naive Solution Fails (Why We Cannot "Stuff Everything into the Prompt")
A common naive approach is: *"Why not load all 1,000 PDFs into Python and pass them directly into the LLM prompt as context?"*

This approach fails completely due to four critical engineering constraints:

| Constraint | Why Prompt Stuffing Fails |
| :--- | :--- |
| **Token Limits (Context Window)** | Even with models supporting large context windows (128k to 1M tokens), thousands of dense documents will easily exceed practical capacity. |
| **Astronomical Costs** | Cloud LLM providers charge per input token. Sending millions of tokens on *every single query* would cost thousands of dollars very quickly. |
| **Severe Latency** | Processing megabytes of text for every simple question causes immense delays (15–60+ seconds per query). |
| **"Lost in the Middle" Phenomenon** | Research shows LLM attention degrades when prompts are bloated. Models easily overlook critical details buried in hundreds of pages of text. |

---

## What RAG Does Differently: The Selective Retrieval Paradigm

Instead of dumping everything into the LLM, RAG introduces **intelligent, targeted retrieval**:

```mermaid
flowchart TD
    subgraph Enterprise["Enterprise Data (1,000+ Files)"]
        F1["Case_01.pdf"]
        F2["Case_32.pdf (Target)"]
        F3["Case_100.pdf"]
        FN["..."]
    end

    Q["User: 'Tell me about Case #32'"] --> R["Intelligent Retriever<br/>(Vector / Semantic Search)"]
    Enterprise --> R

    R -- "Extracts ONLY relevant 2-3 paragraphs" --> Context["Targeted Context Chunk<br/>(from Case_32.pdf, Page 14)"]

    Context --> Prompt["Augmented Prompt<br/>(Query + Small Context)"]
    Q --> Prompt

    Prompt --> LLM["LLM (Gemini / OpenAI)"]
    LLM --> Out["Factual Answer with Citation:<br/>'Case #32 was resolved on... (Source: Case_32.pdf, Page 14)'"]
```

---

## Key Requirements of a Production RAG System

To solve the business problem effectively, the system must satisfy three conditions:

1. **Selective Extraction:** Out of 1,000+ files, dynamically identify and retrieve only the relevant 1–3 pages or text chunks related to the user's specific query.
2. **Grounded Generation:** Instruct the LLM to formulate its answer *strictly* using the retrieved context to guarantee 0% hallucination.
3. **Transparent Citations (Source Attribution):** Return the exact file name and page number alongside the answer so human employees can independently verify the source.

> **Core Summary:** RAG is not about giving the LLM all your data; it is about finding the **exact needle in the haystack** and handing only that needle to the LLM.

---

# 3. Naive Retrieval-Based Solution Approach

## What is the "Naive" RAG Approach?

Now that we know the problem statement, let's explore the simplest and most intuitive way a developer might try to build a system where users can "chat with their documents."

This is known as the **Naive RAG** (or **Prompt Stuffing**) approach.

---

### How the Naive Flow Works

```mermaid
flowchart TD
    A["Raw Documents<br/>(PDFs, Word Docs, Text Files)"] --> B["1. Extract Raw Text from All Files"]
    B --> C["2. Prepare Giant System Prompt<br/>(Inject entire text into prompt)"]
    D["User Query: 'Tell me about XYZ'"] --> E["3. LLM Request<br/>(System Prompt with ALL Text + Query)"]
    C --> E
    E --> F["4. LLM Generation"]
    F --> G["Response to User"]
```

#### Step-by-Step Implementation Flow:
1. **Text Extraction:** Ingest all available business documents (PDFs, DOCX, TXT) and convert their full contents into raw text strings.
2. **System Prompt Preparation:** Inject all the extracted text directly into the system prompt variable:
   ```python
   SYSTEM_PROMPT = f"""
   You are an AI assistant helping employees talk to company data.
   
   AVAILABLE DATA:
   {all_extracted_text_from_1000_files}
   
   Answer the user's question using only the available data above.
   """
   ```
3. **Chat Interaction:** Whenever a user submits a question (Query 1, Query 2, follow-ups), pass the entire `SYSTEM_PROMPT` containing all documents alongside the user query to the LLM (e.g. OpenAI GPT-4o, Gemini, or Claude).
4. **LLM Response:** Because the data is sitting directly in its active context, the model can read it and return an accurate response.

---

## Will This Naive System Work?

**Yes, but only at a trivial scale.**

If your business only has **one file** containing **2–3 pages** (e.g., a short company leave policy or a restaurant menu), this naive solution works completely fine and is very fast to build.

However, in real enterprise scenarios, this approach breaks down completely.

---

## Why the Naive Approach Breaks Down (The Math & Engineering Bottlenecks)

### 1. Astronomical Token Costs
Cloud AI providers charge per million input tokens. 
- In the naive approach, you send the **entire corpus of text** on **every single user message**.
- If a user asks 10 questions in a session, you re-transmit all 1,000 files 10 separate times, multiplying your API costs tenfold.

### 2. Context Window Limits (The 50,000 Files Problem)
Modern flagship models offer large context windows (e.g., 128k to 1 Million tokens). While a 1 Million token window sounds massive, consider realistic enterprise numbers:

$$\begin{aligned}
\text{Total Documents} &= 50,000\text{ files} \\
\text{Average Length} &= 10\text{ pages per file} \\
\text{Total Pages} &= 500,000\text{ pages (5 Lakh pages)} \\
\text{Estimated Tokens} &\approx 125,000,000\text{ to }250,000,000\text{ tokens}
\end{aligned}$$

> **Conclusion:** $250\text{ Million tokens} \gg 1\text{ Million token limit}$. You physically **cannot** fit an enterprise document library into any LLM's context window.

### 3. Severe Latency & Degrading Accuracy
- **Latency:** Parsing and attending over millions of tokens causes response times to skyrocket from 1 second to 30–60+ seconds.
- **Attention Degradation:** Even within large context windows, models suffer from the *"Lost in the Middle"* problem, where critical facts located in the middle of massive prompts are frequently overlooked.

---

## Summary: Premature RAG vs. True Scalable RAG

| Feature | Naive (Premature) RAG | True Scalable RAG |
| :--- | :--- | :--- |
| **How data is fed** | Dumps all text directly into the system prompt | Chunks data, stores as vector embeddings, and retrieves only top matching snippets |
| **Best suited for** | 1–2 small files (under 10 pages) | Tens of thousands of large files (unlimited scale) |
| **Cost per query** | Extremely high (scales with total corpus size) | Very low (scales only with retrieved chunk size) |
| **Speed / Latency** | Very slow for large text | Fast & responsive (sub-second retrieval) |

> **Next Step:** To handle 50,000+ files efficiently, we need **Chunking, Vector Embeddings, and Vector Databases**—the foundation of true Scalable RAG.

---

# 4. RAG Pipeline — Indexing Workflow Explained

## The Two Independent Phases of Production RAG

To scale beyond a naive prompt-stuffing approach and handle tens of thousands of documents, production RAG divides the system into **two completely separate, decoupled phases**:

```mermaid
flowchart LR
    subgraph Phase1["Phase 1: Indexing Phase (Data Ingestion)"]
        direction TB
        D["Documents (PDFs, Docs)"] --> C["Chunking (Splitting)"]
        C --> E["Embedding Model"]
        E --> VDB[("Vector Database<br/>(Vectors + Text + Metadata)")]
    end

    subgraph Phase2["Phase 2: Retrieval Phase (User Query)"]
        direction TB
        Q["User Query"] --> S["Semantic Search"]
        VDB -.-> S
        S --> Gen["Prompt Augmentation & LLM Generation"]
    end
```

| Phase | Purpose | When Does It Run? |
| :--- | :--- | :--- |
| **1. Indexing Phase** | Ingesting, splitting, embedding, and storing documents. | **Offline / Background** (when files are created, updated, or uploaded). |
| **2. Retrieval Phase** | Searching relevant chunks and generating answers. | **Online / Real-time** (when a user asks a question in the chat interface). |

---

## The Indexing Workflow (Step-by-Step)

The **Indexing Phase** is the pipeline that prepares raw unstructured documents so they can be searched in milliseconds.

```mermaid
flowchart TD
    Docs["1. Raw Documents<br/>(PDFs, Word, TXT, Databases)"] --> Chunking["2. Chunking Engine<br/>(Split by Paragraph, Page, or Fixed Token Size)"]
    
    subgraph Chunks["Generated Chunks"]
        C1["Chunk A (Paragraph 1)"]
        C2["Chunk B (Paragraph 2)"]
        C3["Chunk C (Paragraph 3)"]
    end
    
    Chunking --> Chunks
    
    Chunks --> EmbModel["3. Embedding Model<br/>(OpenAI text-embedding-3 / Gemini / HuggingFace)"]
    
    subgraph Vectors["Mathematical Vector Representations"]
        V1["[0.024, -0.812, 0.194, ...]"]
        V2["[0.108, -0.441, -0.320, ...]"]
        V3["[-0.051, 0.732, 0.219, ...]"]
    end
    
    EmbModel --> Vectors
    
    Vectors --> VStore["4. Vector Database (Pinecone, Chroma, Weaviate, Qdrant)"]
```

---

### Step 1: Document Ingestion
Enterprises provide raw unstructured data—such as hundreds or thousands of PDF files, legal contracts, Word documents, Markdown guides, or database exports.

---

### Step 2: Chunking (Splitting the Data)
You cannot pass an entire 50-page PDF to an embedding model or an LLM. We must split the document into smaller, logically coherent pieces called **Chunks**.

#### Common Chunking Strategies:
1. **Page-Level Chunking:** Each page of a PDF becomes 1 chunk.
2. **Paragraph-Level Chunking:** Split wherever there is a double newline (`\n\n`), keeping paragraphs intact.
3. **Fixed-Size Token / Character Chunking:** Split into fixed windows (e.g., 250–500 characters or tokens), often with a slight **overlap** (e.g., 50 tokens) to ensure semantic continuity between boundaries.

---

### Step 3: Generating Vector Embeddings
Raw text chunks cannot be searched mathematically. We feed each chunk through a specialized **Embedding Model** (e.g., OpenAI `text-embedding-3-small`, Gemini Embeddings, or open-source Hugging Face models).

* The model maps the semantic meaning of the text chunk into high-dimensional numerical space (an array of numbers like `[0.012, -0.453, 0.891, ...]`).
* Texts with similar meanings will have vectors positioned close together in this mathematical space.

---

### Step 4: Storing in a Vector Database (Vector DB)
The embeddings and their corresponding contexts are saved into a dedicated **Vector Database** (e.g., Pinecone, Weaviate, ChromaDB, Qdrant, Milvus, pgvector).

#### What is Stored Inside the Vector DB Record?

For every single chunk, the Vector Database stores **three critical pieces of information**:

1. **The Vector (Embedding):** The high-dimensional numeric array used for lightning-fast mathematical similarity search.
2. **The Raw Text Content:** The actual words of the chunk (e.g., *"Case 32 was filed on October 12, 2021 by..."*). This is needed so that when the vector is matched, we can feed the actual human-readable text into the LLM prompt.
3. **The Metadata:** Structured metadata about the origin of the chunk:
   - `file_name`: `"Case_32.pdf"`
   - `page_number`: `14`
   - `section`: `"Settlement Terms"`
   - `created_at`: `"2024-01-15"`

> **Why Metadata Matters:** When the LLM answers the user in the retrieval phase, it uses this metadata to say: *"I found this in Case_32.pdf on Page 14"*, providing full transparency and zero hallucination.

---

## Key Takeaway
The **Indexing Phase** is the foundational half of RAG. It transforms a chaotic mountain of 50,000 enterprise files into an organized, high-speed **Vector Database** containing chunks, vector embeddings, and citation metadata—ready for real-time querying.

---

# 5. RAG Pipeline — Retrieval Mechanism in Depth

## When Does the Retrieval Phase Start?

While the **Indexing Phase** runs in the background whenever files are uploaded, the **Retrieval Phase** is triggered **in real time** the moment a user asks a question in the chat interface (e.g., *"Can you tell me about Case #32?"*).

Its objective is to find the **most semantically relevant chunks** from the Vector Database and pass only those specific snippets to the LLM.

---

## The Retrieval Workflow (Step-by-Step)

```mermaid
flowchart TD
    UserQuery["1. User Query<br/>'Can you tell me about Case #32?'"] --> QueryModel["2. Embedding Model<br/>(MUST match the indexing model)"]
    
    QueryModel --> QueryVec["3. Query Vector<br/>[0.021, -0.795, 0.188, ...]"]
    
    QueryVec --> SimSearch["4. Vector Similarity Search<br/>(Cosine Similarity against Vector DB)"]
    VDB[("Vector Database<br/>(50,000+ Stored Chunks)")] --> SimSearch
    
    SimSearch --> TopK["5. Extract Top-K Relevant Chunks<br/>(e.g., Chunk A & Chunk C + Metadata)"]
    
    TopK --> AugPrompt["6. Augmented System Prompt<br/>(Query + ONLY Relevant Chunks)"]
    UserQuery --> AugPrompt
    
    AugPrompt --> LLM["7. Chat LLM (GPT-4o / Gemini / Claude)"]
    LLM --> FinalAnswer["8. Grounded Answer with Citations<br/>'Case #32 was settled on... (Source: Case_32.pdf, Page 14)'"]
```

---

### Step 1: User Query Ingestion
The user sends a question in plain natural language (e.g., *"What were the settlement conditions in Case #32?"*). At this stage, the system has no prior knowledge of which file contains this information.

---

### Step 2: Converting the Query to Vector Embeddings
The user query is passed to an **Embedding Model** to convert the question into a high-dimensional mathematical vector.

> [!IMPORTANT]
> **Golden Rule of RAG:** You **MUST** use the exact same embedding model (and dimension size) for the query as you used during the Indexing Phase. If you indexed your documents using OpenAI `text-embedding-3-small`, you must embed the query using `text-embedding-3-small`. Otherwise, the mathematical vector coordinates will not align!

---

### Step 3: Vector Similarity Search
The query vector is sent to the **Vector Database** (Pinecone, ChromaDB, Weaviate, etc.).

* The database calculates the mathematical similarity (commonly via **Cosine Similarity** or **Dot Product**) between the query vector and all the stored document vectors.
* Instead of reading all 50,000 files word-by-word, the database identifies the vectors that point in nearly the same mathematical direction within milliseconds.

---

### Step 4: Extracting Top-K Relevant Chunks & Metadata
The Vector DB returns only the **Top-K most relevant chunks** (typically 2 to 5 chunks):
- The mathematical vectors have done their job and can now be dropped.
- We retrieve the **raw text content** of those matching chunks and their **metadata** (file name, page number).
- All remaining irrelevant chunks (the other 49,998+ chunks) are completely ignored.

---

### Step 5: Formulating the Augmented Prompt
The system dynamically builds a compact, focused prompt for the LLM. Instead of passing 50,000 files, we only provide the 2–3 retrieved snippets:

```python
SYSTEM_PROMPT = """
You are a helpful and factual enterprise assistant.
Answer the user's question STRICTLY using only the provided context below.
Always include the source document and page number in your answer.
If the answer is not in the context, say "I cannot find this information in the documents."

CONTEXT:
[Document: Case_32.pdf | Page: 14]
"Case #32 settlement was reached on November 4, 2022. The parties agreed to..."

[Document: Case_32_Addendum.pdf | Page: 2]
"Financial penalty terms for Case #32 were finalized at $150,000..."
"""
```

---

### Step 6: LLM Generation with Citations
The augmented prompt (compact context + user query) is dispatched to the LLM (e.g., GPT-4o, Gemini 1.5 Pro/Flash).

Because the prompt is tiny (a few hundred tokens instead of millions):
1. **Cost is minimal:** Tiny token usage per query.
2. **Speed is fast:** Millisecond generation with zero bloat.
3. **No Hallucination:** The model has the exact ground truth directly in front of it.
4. **Verifiable Citations:** The model outputs an accurate summary and cites the exact file and page number:
   > *"Case #32 was settled on November 4, 2022 with a financial penalty of $150,000. (Sources: Case_32.pdf, Page 14; Case_32_Addendum.pdf, Page 2)"*

---

## Complete RAG Lifecycle Summary

| Dimension | Phase 1: Indexing Phase | Phase 2: Retrieval Phase |
| :--- | :--- | :--- |
| **Trigger** | Document upload / database sync | User sends a chat message |
| **Timing** | Offline / Background batch process | Real-time / Synchronous online request |
| **Input** | Thousands of raw documents (PDFs, DOCX) | Single user query string |
| **Key Action** | Chunk text $\rightarrow$ Embed $\rightarrow$ Store in Vector DB | Embed query $\rightarrow$ Similarity search $\rightarrow$ Augment prompt $\rightarrow$ LLM answer |
| **Output** | Stored records (Vector + Text + Metadata) | Grounded, factual response with citations |


---

# 6. Local Vector DB Setup with Docker Compose

## Setting Up RAG Infrastructure

To start writing code for our RAG system, we must first set up our **infrastructure layer**: the **Vector Database**.

As established in the indexing phase, the Vector Database is the central repository where we store text chunks, vector embeddings, and metadata.

---

## Vector Database Landscape: Which One to Choose?

There are several industry-standard vector databases available today:

| Database | Type | Characteristics |
| :--- | :--- | :--- |
| **Pinecone** | Cloud-Hosted | Fully managed, proprietary SaaS. No local installation; requires API keys and internet access. |
| **Weaviate** | Open Source / Cloud | Feature-rich, highly scalable, supports hybrid search out of the box. |
| **ChromaDB** | Open Source | Embedded / lightweight, popular for quick Python prototypes. |
| **PGVector** | Open Source Extension | Extends PostgreSQL with vector search capabilities; great for existing Postgres stacks. |
| **Qdrant (`qdrant/qdrant`)** | Open Source / Cloud | **Instructor's Choice:** Written in Rust, blazing fast, extremely lightweight, production-ready, and trivial to spin up locally via Docker. |

> [!TIP]
> **Transferable Knowledge:** All vector databases perform the exact same fundamental operations: storing vectors, attaching metadata payloads, and running similarity searches (cosine/dot product). The architecture you learn with Qdrant applies directly to Pinecone, Weaviate, or Chroma.

---

## Setting Up Qdrant Locally with Docker Compose

Running a database inside Docker ensures a clean, isolated environment without cluttering your host operating system.

### Step 1: Ensure Docker is Running
Verify that **Docker Desktop** (or the Docker daemon) is installed and active on your machine.

---

### Step 2: Create the Project Structure
Create a dedicated folder for the project and a `docker-compose.yml` file:

```text
rag/
└── docker-compose.yml
```

---

### Step 3: Write the `docker-compose.yml` File
In `rag/docker-compose.yml`, define the Qdrant vector database service:

```yaml
services:
  vector_db:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```

- **`image: qdrant/qdrant`:** Pulls the official, lightweight Qdrant image from Docker Hub.
- **`ports: "6333:6333"`:** Exposes port `6333` (Qdrant's REST API and built-in Web Dashboard) to your host machine.

---

### Step 4: Launching the Container

Open your terminal, navigate into your `rag` directory, and run:

```bash
# Navigate to the folder
cd rag

# Run in detached mode (background)
docker compose up -d
```

> [!NOTE]
> **Foreground vs. Detached Mode (`-d`):**
> - Running `docker compose up` starts the container in the foreground; pressing `Ctrl + C` stops the database.
> - Adding the `-d` flag runs the container in **detached mode**, freeing up your terminal while keeping the vector database running continuously in the background.

---

### Step 5: Verification & Web Dashboard
1. Run `docker ps` to verify that the `vector_db` container is healthy and running on port `6333`.
2. Open your browser and navigate to:
   ```text
   http://localhost:6333/dashboard
   ```
   You will see Qdrant's built-in web dashboard, where you can visually inspect your collections, stored vector points, and metadata payloads.


API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MzNhZGIxN2EtNGNkMS00MzRkLTlkNzktNDMxZjRiNzQxNDI3In0.B4wkZArPYkav35RJlkPauWAMalY5sNl_7qA-LSVxdgc
Cluster Endpoint=https://99adacd7-1b47-4e20-963d-1915a1792863.ap-northeast-2-0.aws.cloud.qdrant.io


---

# 7. LangChain Installation & Setup

## What is LangChain?
When building GenAI and Agentic applications, developers repeatedly perform common utility tasks:
- Loading documents (PDFs, Word docs, web pages, Notion).
- Splitting text into chunks with smart boundary rules.
- Calling embedding models to convert text into vectors.
- Connecting to vector databases (Qdrant, Pinecone, ChromaDB).
- Orchestrating LLM chat calls and prompt templates.

Instead of writing custom boilerplate from scratch for every project, **LangChain** provides a standardized, modular ecosystem of tools, loaders, and database adapters out of the box.

---

## Required Dependencies Installation

Run the following commands in your terminal inside your active `.venv`:

```powershell
# 1. Document Loaders & PDF parsing
pip install langchain-community pypdf

# 2. Text Chunking & Splitting
pip install langchain-text-splitters

# 3. Embeddings & OpenAI models
pip install langchain-openai

# 4. Qdrant Vector Database Adapter
pip install langchain-qdrant

# 5. Environment variable management
pip install python-dotenv
```

> [!TIP]
> **All-in-one command:**
> ```powershell
> pip install langchain-community pypdf langchain-text-splitters langchain-openai langchain-qdrant python-dotenv
> ```

---

# 8. LangChain Document Loaders for PDF

## Why Use Document Loaders?
PDFs are not plain text files; they contain binary streams, embedded fonts, multi-column layouts, and metadata.
LangChain's **`PyPDFLoader`** automatically:
1. Opens the PDF file.
2. Extracts text page-by-page.
3. Encapsulates each page as a LangChain `Document` object with two attributes:
   - `page_content`: The raw extracted text of that page.
   - `metadata`: A dictionary holding `source` (file path) and `page` (0-indexed page number).

---

## Step-by-Step Code: Loading a PDF

```python
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# 1. Define the path to your PDF file
pdf_path = Path(__file__).parent / "nodejs.pdf"

# 2. Initialize the PyPDFLoader
loader = PyPDFLoader(file_path=str(pdf_path))

# 3. Load documents (each page becomes one Document object)
docs = loader.load()

# Inspecting the loaded output:
print(f"Total Pages Loaded: {len(docs)}")
print("Sample Page Content:\n", docs[0].page_content[:300])
print("Sample Metadata:\n", docs[0].metadata)
# Output metadata: {'source': '.../nodejs.pdf', 'page': 0}
```

---

# 9. LangChain Document Chunking & Splitting

## Why Page-Level Loading is Not Enough
Even though `PyPDFLoader` splits by page, pages can be uneven:
- A page might contain 3,000 words (exceeding ideal chunk sizes for retrieval).
- Topics often start in the middle of a page and continue onto the next.
- If you slice text rigidly, sentences and concepts get broken in half.

To solve this, we use LangChain's **`RecursiveCharacterTextSplitter`**.

---

## The Power of `chunk_size` and `chunk_overlap`

```mermaid
flowchart LR
    subgraph Chunk1["Chunk 1 (1000 characters)"]
        A["[Text Content...]"]
        B["[Overlap Zone (400 chars)]"]
    end
    
    subgraph Chunk2["Chunk 2 (1000 characters)"]
        B2["[Same 400 chars repeated]"]
        C["[Next Text Content...]"]
    end
    
    Chunk1 -. "Preserves context across boundary" .-> Chunk2
```

- **`chunk_size` (e.g., 1000):** The maximum number of characters/tokens each chunk should hold.
- **`chunk_overlap` (e.g., 400):** The number of characters repeated from the end of the previous chunk into the start of the next chunk.
  - **Why overlap matters:** Without overlap, a critical sentence or definition sliced between two chunks loses its surrounding context. Overlap acts as a bridge so the embedding model retains the full semantic picture.

---

## Step-by-Step Code: Chunking Documents

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize the recursive character splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

# 2. Split the loaded PDF pages into smaller chunks
chunks = text_splitter.split_documents(docs)

print(f"Original Pages: {len(docs)}")
print(f"Total Chunks Created: {len(chunks)}")
```

---

# 10. LangChain Vector Store as Retrievers

## Creating Vector Embeddings & Ingesting into Qdrant

Now that our documents are chunked, we perform the final step of the **Indexing Phase**:
1. Convert each chunk into a mathematical vector using an **Embedding Model** (`text-embedding-3-large`).
2. Store the vectors, raw text, and metadata in **Qdrant Vector Database**.

---

## Complete Indexing Script (`index.py`)

Here is the complete code for the indexing pipeline:

```python
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# 1. Load environment variables (.env with OPENAI_API_KEY)
load_dotenv()

# 2. Locate and load the PDF document
pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path=str(pdf_path))
docs = loader.load()
print(f"✅ Loaded {len(docs)} pages from PDF.")

# 3. Chunk the documents with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
chunks = text_splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} chunks.")

# 4. Initialize the Embedding Model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# 5. Connect to local Qdrant Vector DB & Index the chunks
print("⏳ Generating embeddings and storing into Qdrant...")
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("🚀 Indexing of documents done! Check dashboard at http://localhost:6333/dashboard")
```

### Verifying in Qdrant Dashboard
When you open `http://localhost:6333/dashboard`:
- You will see the new collection `learning_rag`.
- Inspect the stored points: each record contains the high-dimensional vector, the original `page_content`, and metadata (`page` number, `source` file path).

---

# 11. LangChain-Powered RAG Retrieval Execution

## Building the Interactive Chat & Retrieval Script (`chat.py`)

Now we build the real-time **Retrieval Phase**:
1. Connect to our existing Qdrant collection (`from_existing_collection`).
2. Accept a natural language user query.
3. Perform a **similarity search** in Qdrant to retrieve only the top relevant chunks.
4. Construct an augmented `SYSTEM_PROMPT` containing the retrieved text and page numbers.
5. Send the augmented prompt to the LLM (e.g., `gpt-4o` / `gpt-4.5` / `gemini`) and output a factual answer with exact page citations.

---

## Complete Retrieval & Chat Script (`chat.py`)

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# 1. Load environment variables
load_dotenv()

# 2. Initialize the exact same embedding model used during indexing
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# 3. Connect to the existing Qdrant collection
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model
)

# 4. Initialize the OpenAI chat client
openai_client = OpenAI()

def main():
    user_query = input("\nAsk something about the document: ")
    if not user_query.strip():
        return

    print("🔍 Searching vector database for relevant chunks...")
    
    # 5. Perform vector similarity search
    search_results = vector_db.similarity_search(query=user_query, k=4)
    
    # 6. Build the context string with metadata
    context_blocks = []
    for result in search_results:
        page_num = result.metadata.get("page", "Unknown")
        source_file = result.metadata.get("source", "Unknown")
        context_blocks.append(
            f"[Source: {source_file} | Page: {page_num}]\n{result.page_content}"
        )
    
    context = "\n\n---\n\n".join(context_blocks)

    # 7. Construct the augmented system prompt
    system_prompt = f"""
You are a helpful and factual AI assistant.
Answer the user's question based strictly on the available context below.
Always guide the user by citing the exact page number(s) where the answer was found.
If the answer cannot be found in the context, state that the document does not contain this information.

AVAILABLE CONTEXT:
{context}
"""

    # 8. Query the LLM
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    print("\n🤖 AI Assistant:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
```

---

## Real-World Output Examples

### Example 1: Query about Debugging
> **User Query:** *"Can you help me understand debugging in Node.js?"*
> 
> **AI Assistant:**
> *"To debug Node.js applications, you can use the built-in inspect flag: `node --inspect index.js`. This allows you to attach Chrome DevTools or VS Code to set breakpoints and step through code. You can find detailed steps and debugging flags on **Page 23 and 24** of the guide."*

### Example 2: Query about Arrow Functions
> **User Query:** *"Can you help me understand the concept of arrow functions?"*
> 
> **AI Assistant:**
> *"Arrow functions provide a concise syntax for writing function expressions and lexically bind the `this` value. For complete code syntax and examples, refer to **Page 20 and 21**."*


---

# 7. LangChain Installation & Setup

## What is LangChain?
When building GenAI and Agentic applications, developers repeatedly perform common utility tasks:
- Loading documents (PDFs, Word docs, web pages, Notion).
- Splitting text into chunks with smart boundary rules.
- Calling embedding models to convert text into vectors.
- Connecting to vector databases (Qdrant, Pinecone, ChromaDB).
- Orchestrating LLM chat calls and prompt templates.

Instead of writing custom boilerplate from scratch for every project, **LangChain** provides a standardized, modular ecosystem of tools, loaders, and database adapters out of the box.

---

## Required Dependencies Installation

Run the following commands in your terminal inside your active `.venv`:

```powershell
# 1. Document Loaders & PDF parsing
pip install langchain-community pypdf

# 2. Text Chunking & Splitting
pip install langchain-text-splitters

# 3. Embeddings & OpenAI models
pip install langchain-openai

# 4. Qdrant Vector Database Adapter
pip install langchain-qdrant

# 5. Environment variable management
pip install python-dotenv
```

> [!TIP]
> **All-in-one command:**
> ```powershell
> pip install langchain-community pypdf langchain-text-splitters langchain-openai langchain-qdrant python-dotenv
> ```

---

# 8. LangChain Document Loaders for PDF

## Why Use Document Loaders?
PDFs are not plain text files; they contain binary streams, embedded fonts, multi-column layouts, and metadata.
LangChain's **`PyPDFLoader`** automatically:
1. Opens the PDF file.
2. Extracts text page-by-page.
3. Encapsulates each page as a LangChain `Document` object with two attributes:
   - `page_content`: The raw extracted text of that page.
   - `metadata`: A dictionary holding `source` (file path) and `page` (0-indexed page number).

---

## Step-by-Step Code: Loading a PDF

```python
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# 1. Define the path to your PDF file
pdf_path = Path(__file__).parent / "nodejs.pdf"

# 2. Initialize the PyPDFLoader
loader = PyPDFLoader(file_path=str(pdf_path))

# 3. Load documents (each page becomes one Document object)
docs = loader.load()

# Inspecting the loaded output:
print(f"Total Pages Loaded: {len(docs)}")
print("Sample Page Content:\n", docs[0].page_content[:300])
print("Sample Metadata:\n", docs[0].metadata)
# Output metadata: {'source': '.../nodejs.pdf', 'page': 0}
```

---

# 9. LangChain Document Chunking & Splitting

## Why Page-Level Loading is Not Enough
Even though `PyPDFLoader` splits by page, pages can be uneven:
- A page might contain 3,000 words (exceeding ideal chunk sizes for retrieval).
- Topics often start in the middle of a page and continue onto the next.
- If you slice text rigidly, sentences and concepts get broken in half.

To solve this, we use LangChain's **`RecursiveCharacterTextSplitter`**.

---

## The Power of `chunk_size` and `chunk_overlap`

```mermaid
flowchart LR
    subgraph Chunk1["Chunk 1 (1000 characters)"]
        A["[Text Content...]"]
        B["[Overlap Zone (400 chars)]"]
    end
    
    subgraph Chunk2["Chunk 2 (1000 characters)"]
        B2["[Same 400 chars repeated]"]
        C["[Next Text Content...]"]
    end
    
    Chunk1 -. "Preserves context across boundary" .-> Chunk2
```

- **`chunk_size` (e.g., 1000):** The maximum number of characters/tokens each chunk should hold.
- **`chunk_overlap` (e.g., 400):** The number of characters repeated from the end of the previous chunk into the start of the next chunk.
  - **Why overlap matters:** Without overlap, a critical sentence or definition sliced between two chunks loses its surrounding context. Overlap acts as a bridge so the embedding model retains the full semantic picture.

---

## Step-by-Step Code: Chunking Documents

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize the recursive character splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

# 2. Split the loaded PDF pages into smaller chunks
chunks = text_splitter.split_documents(docs)

print(f"Original Pages: {len(docs)}")
print(f"Total Chunks Created: {len(chunks)}")
```

---

# 10. LangChain Vector Store as Retrievers

## Creating Vector Embeddings & Ingesting into Qdrant

Now that our documents are chunked, we perform the final step of the **Indexing Phase**:
1. Convert each chunk into a mathematical vector using an **Embedding Model** (`text-embedding-3-large`).
2. Store the vectors, raw text, and metadata in **Qdrant Vector Database**.

---

## Complete Indexing Script (`index.py`)

Here is the complete code for the indexing pipeline:

```python
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# 1. Load environment variables (.env with OPENAI_API_KEY)
load_dotenv()

# 2. Locate and load the PDF document
pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path=str(pdf_path))
docs = loader.load()
print(f"✅ Loaded {len(docs)} pages from PDF.")

# 3. Chunk the documents with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
chunks = text_splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} chunks.")

# 4. Initialize the Embedding Model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# 5. Connect to local Qdrant Vector DB & Index the chunks
print("⏳ Generating embeddings and storing into Qdrant...")
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("🚀 Indexing of documents done! Check dashboard at http://localhost:6333/dashboard")
```

### Verifying in Qdrant Dashboard
When you open `http://localhost:6333/dashboard`:
- You will see the new collection `learning_rag`.
- Inspect the stored points: each record contains the high-dimensional vector, the original `page_content`, and metadata (`page` number, `source` file path).

---

# 11. LangChain-Powered RAG Retrieval Execution

## Building the Interactive Chat & Retrieval Script (`chat.py`)

Now we build the real-time **Retrieval Phase**:
1. Connect to our existing Qdrant collection (`from_existing_collection`).
2. Accept a natural language user query.
3. Perform a **similarity search** in Qdrant to retrieve only the top relevant chunks.
4. Construct an augmented `SYSTEM_PROMPT` containing the retrieved text and page numbers.
5. Send the augmented prompt to the LLM (e.g., `gpt-4o` / `gpt-4.5` / `gemini`) and output a factual answer with exact page citations.

---

## Complete Retrieval & Chat Script (`chat.py`)

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# 1. Load environment variables
load_dotenv()

# 2. Initialize the exact same embedding model used during indexing
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# 3. Connect to the existing Qdrant collection
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model
)

# 4. Initialize the OpenAI chat client
openai_client = OpenAI()

def main():
    user_query = input("\nAsk something about the document: ")
    if not user_query.strip():
        return

    print("🔍 Searching vector database for relevant chunks...")
    
    # 5. Perform vector similarity search
    search_results = vector_db.similarity_search(query=user_query, k=4)
    
    # 6. Build the context string with metadata
    context_blocks = []
    for result in search_results:
        page_num = result.metadata.get("page", "Unknown")
        source_file = result.metadata.get("source", "Unknown")
        context_blocks.append(
            f"[Source: {source_file} | Page: {page_num}]\n{result.page_content}"
        )
    
    context = "\n\n---\n\n".join(context_blocks)

    # 7. Construct the augmented system prompt
    system_prompt = f"""
You are a helpful and factual AI assistant.
Answer the user's question based strictly on the available context below.
Always guide the user by citing the exact page number(s) where the answer was found.
If the answer cannot be found in the context, state that the document does not contain this information.

AVAILABLE CONTEXT:
{context}
"""

    # 8. Query the LLM
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    print("\n🤖 AI Assistant:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
```

---

## Real-World Output Examples

### Example 1: Query about Debugging
> **User Query:** *"Can you help me understand debugging in Node.js?"*
> 
> **AI Assistant:**
> *"To debug Node.js applications, you can use the built-in inspect flag: `node --inspect index.js`. This allows you to attach Chrome DevTools or VS Code to set breakpoints and step through code. You can find detailed steps and debugging flags on **Page 23 and 24** of the guide."*

### Example 2: Query about Arrow Functions
> **User Query:** *"Can you help me understand the concept of arrow functions?"*
> 
> **AI Assistant:**
> *"Arrow functions provide a concise syntax for writing function expressions and lexically bind the `this` value. For complete code syntax and examples, refer to **Page 20 and 21**."*
