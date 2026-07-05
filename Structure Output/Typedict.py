from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class Person(TypedDict):
    name: str
    age: int

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

structured_llm = llm.with_structured_output(Person)

result = structured_llm.invoke("My name is John and I am 30 years old. What is my name and age?")
print(result)