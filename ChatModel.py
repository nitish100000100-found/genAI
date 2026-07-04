from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Create the LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Invoke the model
response = llm.invoke("What is the capital of India?")

print(response.content)