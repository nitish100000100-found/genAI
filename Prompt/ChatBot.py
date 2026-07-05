from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

history = ['You are an ChatBot,Talk in engaging way'];

while True:

    user_input = input("User(Press e to exit): ")

    if user_input == "e":
        break

    history.append(user_input)

    prompt = PromptTemplate(
        template='You are an ChatBot,Talk in engaging way. The conversation so far is: {history}',
        input_variables=['history']
    )

    filled_prompt = prompt.invoke({'history': history})

    response = llm.invoke(filled_prompt)

    history.append(response.content)

    print("ChatBot:", response.content) 
