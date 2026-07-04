from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

# Invoke the model
response = llm.invoke("What is the future of AI in healthcare? in just 5 points")

print(response.content)