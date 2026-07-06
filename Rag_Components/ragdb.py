import os

from dotenv import load_dotenv
from langchain_core import documents
from sqlalchemy import create_engine
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_postgres import PGVector



load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_pre_ping=True,
    pool_recycle=1800,
)


embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = PGVector(
    embeddings=embedding,
    collection_name="documents",
    connection=engine,
    use_jsonb=True,
)

from langchain_core.documents import Document



# ======================
# Store Documents
# ======================

docs = [
    Document(page_content="Python is a programming language."),
    Document(page_content="Machine Learning is a subset of AI."),
    Document(page_content="Neon is a cloud PostgreSQL database."),
]

vector_store.add_documents(docs)

print("Documents Stored!")

# ======================
# Similarity Search
# ======================

results = vector_store.similarity_search(
    "What is Python?",
    k=1,
)

print()

for doc in results:
    print(doc.page_content)


#############################################################################################################
# ============================================================
# ADD DATA
# ============================================================

# Add Document objects to the vector store.
# Each Document contains page_content and optional metadata.
# ids are optional. If not provided, LangChain generates them.
vector_store.add_documents(
    documents=documents,
    ids=[
        "doc1",
        "doc2",
        "doc3"
    ]
)


# Add plain text directly to the vector store.
# Metadata and ids are optional.
vector_store.add_texts(
    texts=[
        "Python is a programming language.",
        "Java is object-oriented."
    ],
    metadatas=[
        {"source": "book1"},
        {"source": "book2"}
    ],
    ids=[
        "python_doc",
        "java_doc"
    ]
)


# ============================================================
# SEARCH
# ============================================================

# Performs semantic similarity search.
# Returns the top-k most relevant Document objects.
docs = vector_store.similarity_search(
    query="What is Python?",
    k=5
)


# Performs semantic similarity search and also returns similarity scores.
# Returns a list of (Document, Score) tuples.
results = vector_store.similarity_search_with_score(
    query="What is Python?",
    k=5
)

for doc, score in results:
    print(score)
    print(doc.page_content)


# Search using an already computed embedding vector.
# Useful when you've generated the embedding yourself.
query_vector = embedding.embed_query("Python")

docs = vector_store.similarity_search_by_vector(
    embedding=query_vector,
    k=5
)


# Maximum Marginal Relevance (MMR) Search.
# Returns relevant documents while avoiding very similar/duplicate chunks.
# fetch_k = number of candidate documents considered before selecting k results.
docs = vector_store.max_marginal_relevance_search(
    query="Python",
    k=5,
    fetch_k=20
)


# ============================================================
# METADATA FILTERING
# ============================================================

# Semantic search restricted to documents whose metadata matches the filter.
docs = vector_store.similarity_search(
    query="language",
    k=5,
    filter={
        "category": "programming"
    }
)


# Filter using multiple metadata fields.
# All filter conditions must match.
docs = vector_store.similarity_search(
    query="language",
    k=5,
    filter={
        "category": "programming",
        "author": "Nitish"
    }
)


# ============================================================
# DELETE
# ============================================================

# Delete specific documents using their IDs.
vector_store.delete(
    ids=[
        "doc1",
        "python_doc"
    ]
)


# Delete the entire collection.
# Removes all documents and embeddings in this collection.
# Does NOT delete the PostgreSQL tables.
vector_store.delete_collection()



#######################################################################################################################




similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)

docs = similarity_retriever.invoke("What is PGVector?")


# =====================================================
# 2. MMR Retriever
# =====================================================
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

docs = mmr_retriever.invoke("What is PGVector?")


# =====================================================
# 3. Similarity Score Threshold Retriever
# =====================================================

threshold_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,
        "k": 5
    }
)

docs = threshold_retriever.invoke("What is PGVector?")


# =====================================================
# 4. Similarity Retriever + Metadata Filter
# =====================================================

filter_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {
            "category": "programming"
        }
    }
)

docs = filter_retriever.invoke("language")


## =====================================================
# 5. Multi-Query Retriever
# ===================================================== 

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.retrievers import MultiQueryRetriever

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    ),
    llm=llm,
)

docs = retriever.invoke(
    "How does PGVector work?"
)

for doc in docs:
    print(doc.page_content)


## =====================================================
# 6. Contextual Compression Retriever
# ===================================================== 

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=vector_store.as_retriever(),
    base_compressor=compressor
)

docs = compression_retriever.invoke(
    "How does PGVector indexing work?"
)









# SQLAlchemy
# SQLAlchemy → Python ka database toolkit aur ORM jo database connection, connection pooling aur SQL operations ko manage karta hai.
# pgvector
# pgvector → PostgreSQL extension jo AI embeddings (vectors) ko store aur similarity search karne ke liye use hota hai.
# create_engine()
# create_engine() → SQLAlchemy Engine banata hai jo database connections aur connection pool ko manage karta hai.
# DATABASE_URL
# DATABASE_URL → Database ka connection string (host, port, username, password aur database name).
# pool_size
# pool_size → Pool me maximum kitne reusable (persistent) connections rahenge.
# max_overflow
# max_overflow → Pool full hone par kitne extra temporary connections ban sakte hain.
# pool_timeout
# pool_timeout → Connection na mile to kitni der wait karna hai, uske baad timeout error.
# pool_pre_ping
# pool_pre_ping → Connection use karne se pehle check karta hai ki wo alive hai ya dead.
# pool_recycle
# pool_recycle → Itne seconds baad purane connection ko close karke naya connection bana deta hai.