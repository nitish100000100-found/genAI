from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

messages=[SystemMessage(content="You are an ChatBot,Talk in engaging way"),
          HumanMessage(content="Tell me a shayari about love")]

messages.append(AIMessage(content=llm.invoke(messages).content))


for message in messages:
    if isinstance(message, SystemMessage):
        print("System:", message.content)
    elif isinstance(message, HumanMessage):
        print("User:", message.content)
    elif isinstance(message, AIMessage):
        print("ChatBot:", message.content)