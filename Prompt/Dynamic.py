from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate([
  ("system","You are expert in {topic} and you are a helpful assistant"),
])


history=prompt.invoke({'topic':'Shayari'}).to_messages()

while True:
    user_input = input("User(Press e to exit): ")

    if user_input == "e":
        break

    history.append(HumanMessage(content=user_input))

    response = llm.invoke(history)

    history.append(AIMessage(content=response.content))

    print("ChatBot:", response.content)



print()
print("Conversation History:")
print()
for message in history:
    if isinstance(message, SystemMessage):
        print("System:", message.content)
    elif isinstance(message, HumanMessage):
        print("User:", message.content)
    elif isinstance(message, AIMessage):
        print("ChatBot:", message.content)