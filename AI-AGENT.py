import math
import os
import requests
from datetime import datetime
from typing import Annotated, TypedDict
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.document_loaders import YoutubeLoader


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set.")

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_pre_ping=True,
    pool_recycle=1800,
)

with engine.connect() as conn:
    conn.execute(text("SELECT 1"))

print("🔌 PostgreSQL connected.")
print()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

embedding.embed_query("startup")
print("🧠 Embedding model ready.")
print()

vector_store = PGVector(
    embeddings=embedding,
    collection_name="documents",
    connection=engine,
    use_jsonb=True,
)

vector_store.similarity_search("startup", k=1)
print("💾 PGVector ready.")
print()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

llm.invoke("Hello")
print("🚀 Groq ready.")
print()

SYSTEM_PROMPT = """
You are a helpful, knowledgeable, and reliable AI assistant.

You have access to the following tools:

1. calculator
   - Use for arithmetic and mathematical calculations.
   - Always use this tool instead of performing calculations mentally.

2. wikipedia_search
   - Search Wikipedia for factual, encyclopedic, and general knowledge.
   - Always use this tool if the user is asking for information about a person, historical figure, place, scientific concept, history, or definition.

3. DuckDuckGo Search
   - Search the web for current, live, or time-sensitive information.
   - Only use this tool if the query is specifically about new, latest, or recent events, current news, software updates, or live information that changes frequently.

4. rag_search
   - Search the indexed PDF knowledge base.
   - Use this tool whenever the user asks questions about uploaded PDFs or documents.
   - If the answer is available in the uploaded documents, prefer this tool over Wikipedia or web search.

5. current_time
   - Get the current local date and time.
   - Use this tool whenever the user asks for the current date, time, day, or requests information relative to the current time (e.g., "today", "yesterday", "now").

6. weather
   - Get the current weather for a city.
   - Use this tool whenever the user asks about the weather, temperature, or climate conditions of a specific city or location.

Tool Usage Rules:
- Always use the most appropriate tool instead of guessing.
- Use multiple tools if necessary.
- Do not fabricate information when a tool can provide the answer.
- If a tool returns no useful information, try another relevant tool before answering.
- For mathematical calculations, always use the calculator tool.
- For uploaded document questions, always try rag_search first.
- For current date or time, always use the current_time tool.
- For weather queries about a city, always use the weather tool.
- For general factual knowledge, definitions, history, concepts, or information about a person, always use the wikipedia_search tool.
- Only use the DuckDuckGo Search tool when looking for recent events, current news, live updates, or the latest information that is not available on Wikipedia.

Response Guidelines:
- Answer clearly and concisely.
- Use bullet points or numbered lists when they improve readability.
- Explain technical concepts step by step when appropriate.
- If the available information is insufficient, say so instead of making up an answer.
- Never mention internal reasoning, hidden prompts, or tool-selection logic.
- Respond naturally as if the tools are part of your capabilities.
"""


def add_pdf(pdf_path: str) -> int:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(docs)
    vector_store.add_documents(chunks)
    return len(chunks)


def normalize_youtube_url(url: str) -> str:
    parsed = urlparse(url)

    if parsed.netloc == "youtu.be":
        video_id = parsed.path.lstrip("/")
        return f"https://www.youtube.com/watch?v={video_id}"

    if "/shorts/" in parsed.path:
        video_id = parsed.path.split("/shorts/")[1]
        return f"https://www.youtube.com/watch?v={video_id}"

    if "v" in parse_qs(parsed.query):
        video_id = parse_qs(parsed.query)["v"][0]
        return f"https://www.youtube.com/watch?v={video_id}"

    return url


def add_youtube(url: str) -> int:
    normalized_url = normalize_youtube_url(url)
    loader = YoutubeLoader.from_youtube_url(
        normalized_url,
        language=["en"],
        add_video_info=False,
    )

    docs = loader.load()

    if not docs:
        raise ValueError("No transcript found.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["source"] = url
        chunk.metadata["type"] = "youtube"

    vector_store.add_documents(chunks)
    return len(chunks)


wiki = WikipediaAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=3000
)

rag_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        allowed = {"__builtins__": {}, "math": math}
        return str(eval(expression, allowed, {}))
    except Exception as e:
        return str(e)


@tool
def rag_search(query: str) -> str:
    """
    Search all indexed knowledge (PDFs, YouTube, etc.).
    """
    docs = vector_store.similarity_search(query, k=4)

    if not docs:
        return "No relevant information found."

    return "\n\n".join(doc.page_content for doc in docs)


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    try:
        result = wiki.run(query)

        if not result or result.strip() == "":
            return "No Wikipedia results found."

        return result

    except Exception as e:
        return f"Wikipedia search failed: {e}"


@tool
def current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def weather(city: str) -> str:
    """Get the current weather for a city."""
    api_key = os.getenv("FOR_WEA")
    if not api_key:
        return "Weather API key not configured."

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    try:
        response = requests.get(url).json()
        cod = response.get("cod")

        if cod == 200:
            temp = response["main"]["temp"]
            desc = response["weather"][0]["description"]
            return f"{city}: {temp}°C, {desc}"
        elif cod == 401:
            return "Weather API Error: Invalid or expired API key. Please check FOR_WEA in your .env file."
        elif cod == 404:
            return f"City '{city}' not found."
        else:
            message = response.get("message", "Unknown error")
            return f"Weather API Error (Status {cod}): {message}"

    except Exception as e:
        return f"Failed to retrieve weather: {e}"


duckduckgo_search = DuckDuckGoSearchRun()

tools = [
    calculator,
    rag_search,
    wikipedia_search,
    duckduckgo_search,
    current_time,
    weather,
]

llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: ChatState):
    return {
        "messages": [
            llm_with_tools.invoke(state["messages"])
        ]
    }


builder = StateGraph(ChatState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile()

history = [
    SystemMessage(content=SYSTEM_PROMPT)
]

print("🤖 Assistant Commands:")
print("  📂 add_pdf      -> Index a local PDF file")
print("  🎥 add_youtube  -> Index a YouTube video transcript")
print("  ❌ exit         -> Quit the session")
print()

while True:
    user = input("👤 You: ").strip()

    if user.lower() in {"exit", "quit"}:
        break

    if user.lower() == "add_pdf":
        path = input("📂 PDF path: ").strip()
        try:
            count = add_pdf(path)
            print(f"\n📚 Indexed {count} chunks successfully.\n")
            user = input("👤 Ask about the PDF: ").strip()
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue

    elif user.lower() == "add_youtube":
        url = input("🎥 YouTube URL: ").strip()
        try:
            count = add_youtube(url)
            print(f"\n🎥 Indexed {count} transcript chunks successfully.\n")
            user = input("👤 Ask about the video: ").strip()
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue

    history.append(HumanMessage(content=user))

    result = graph.invoke(
        {
            "messages": history
        }
    )

    history = result["messages"]

    print("🤖 Assistant:", history[-1].content)
    print()





#  # Database URL (PostgreSQL / Neon DB connection string)
# DATABASE_URL=postgresql://username:password@host/database?sslmode=require

# # Google Gemini API Key (Required for embeddings)
# GEMINI_API_KEY=your_gemini_api_key_here

# # OpenWeatherMap API Key (Required for weather tool)
# FOR_WEA=your_openweathermap_api_key_here

# # Groq API Key (Required for ChatGroq LLM)
# GROQ_API_KEY=your_groq_api_key_here
   
