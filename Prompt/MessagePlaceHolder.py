from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)


chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = [
    HumanMessage(content='I have a problem with my order'),
    AIMessage(content='I am sorry to hear that. Can you please provide me with your order number so I can look into it?')
]   
final=chat_template.invoke({'query':'My order number is 12345', 'chat_history': chat_history}).to_messages()

final.append(llm.invoke(final).content)
print("ChatBot:", final[-1])