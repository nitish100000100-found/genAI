# ============================================
# 1. CharacterTextSplitter -> Normal Text
# ============================================

from langchain_text_splitters import CharacterTextSplitter

text = "Python is easy. AI is powerful."

splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=10,
    chunk_overlap=0
)

print(splitter.split_text(text))

# Summary:
# Splits a normal string into a list of strings.


# ============================================
# 2. CharacterTextSplitter -> Documents
# ============================================

from langchain_text_splitters import CharacterTextSplitter

text = "Python is easy. AI is powerful."

splitter = CharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=0
)

docs = splitter.create_documents([text])

print(docs)

# Summary:
# Converts text into Document objects after splitting.


# ============================================
# 3. CharacterTextSplitter -> PDF
# ============================================

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader("sample.pdf").load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)

# Summary:
# Splits existing Document objects (like PDFs) into smaller Documents.


# ============================================
# 4. RecursiveCharacterTextSplitter
# ============================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Python is easy.

AI is powerful.

Football is popular.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=0
)

print(splitter.split_text(text))

# Summary:
# Tries paragraphs → lines → words → characters to preserve context.
# Best general-purpose text splitter.


# ============================================
# 5. RecursiveCharacterTextSplitter (Code)
# ============================================

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = """
def add(a,b):
    return a+b

def sub(a,b):
    return a-b
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=30,
    chunk_overlap=0
)

print(splitter.split_text(code))

# Summary:
# Splits source code using language-specific rules (functions, classes, etc.).
# Best for Python, Java, JS, C++, Go, etc.


# ============================================
# 6. SemanticChunker
# ============================================

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

splitter = SemanticChunker(embedding)

text = """
Python is used for AI.

Football is a popular sport.
"""

docs = splitter.create_documents([text])

print(docs)

# Summary:
# Uses embeddings to split text based on meaning instead of character count.
# Best when you want semantically related sentences to stay together.