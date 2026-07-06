from langchain_community.document_loaders import TextLoader, PyPDFLoader,WebBaseLoader,CSVLoader,DirectoryLoader,SeleniumURLLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model  = ChatGroq(
    model="llama-3.1-8b-instant"
)

prompt = PromptTemplate(
    template='Write a summary for the following Blob - \n {blog}',
    input_variables=['blog']
)

parser = StrOutputParser()

loader = TextLoader('Jobs.txt')

docs = loader.load()

text = "\n\n".join(doc.page_content for doc in docs)

chain = prompt | model | parser

print(chain.invoke({'blog':text}))





#more loaders(you can implement lazy loading for large files)


# 1. TextLoader (Loads one .txt file)
text_loader = TextLoader("notes.txt")
text_docs = text_loader.load()


# 2. PyPDFLoader (Loads one PDF)
# you have to install PyMuPDF library for this loader to work
# you have different pdf loaders available in langchain_community.document_loaders, you can choose any of them based on your requirement
pdf_loader = PyPDFLoader("book.pdf")
pdf_docs = pdf_loader.load()


# 3. CSVLoader (Loads one CSV file)
csv_loader = CSVLoader("students.csv")
csv_docs = csv_loader.load()


# 4. WebBaseLoader (Loads a webpage)
web_loader = WebBaseLoader("https://example.com")
web_docs = web_loader.load()


# 5. DirectoryLoader (Loads all .txt files from a folder)
dir_loader = DirectoryLoader(
    path="documents",
    glob="*.txt",
    loader_cls=TextLoader
)
dir_docs = dir_loader.load()


# 6. SeleniumURLLoader (Loads a webpage using Selenium)
loader = SeleniumURLLoader(
    urls=[
        "https://example.com"
    ]
)

docs = loader.load()


# Print number of loaded documents
print("Text Docs:", len(text_docs))
print("PDF Docs :", len(pdf_docs))
print("CSV Docs :", len(csv_docs))
print("Web Docs :", len(web_docs))
print("Dir Docs :", len(dir_docs))
