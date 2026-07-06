from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import  HumanMessage
load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

llm_with_tools = llm.bind_tools([add])
query=HumanMessage(content="What is 25 + 17?")

response = llm_with_tools.invoke([query])

tool_call = response.tool_calls[0]

result =add.invoke(tool_call)

final = llm_with_tools.invoke([
    query,
    response,   
    result     
])

print(final.content)



