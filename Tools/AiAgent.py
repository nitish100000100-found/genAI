from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import  HumanMessage

load_dotenv()

# ---------------- LLM ----------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ---------------- Tool ----------------

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Example:
    "25 + 17"
    "(10 * 5) / 2"
    """
    return str(eval(expression))

# ---------------- Agent ----------------

agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt="You are a helpful AI assistant. Use the calculator tool whenever math is required."
)

# ---------------- Run ----------------

response = agent.invoke({
    "messages": [
        HumanMessage(content="What is the result of 25 + 17?")
    ]
})

for message in response["messages"]:
    print(f"{type(message).__name__}: {message.content}")



print(response["messages"][-1].content)    