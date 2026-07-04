from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0
)

response = llm.invoke("Solve Leetcode problem 2235: Add Two Integers in Python")
print(response.content)