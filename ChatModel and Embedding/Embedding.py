from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Create Gemini embedding model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Documents
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar is known as the God of Cricket.",
    "Rohit Sharma is famous for his double centuries in ODIs.",
    "Jasprit Bumrah is an Indian fast bowler known for his yorkers."
]

# Query
query = "Tell me about Bumrah"
print("Query:", query)
print()
# Generate embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)

best_score = -1
best_index = -1

for i in range(len(documents)):
    print("Document:", documents[i])
    print("Similarity:", scores[0][i])
    print()

    if scores[0][i] > best_score:
        best_score = scores[0][i]
        best_index = i

print("Best Match:")
print(documents[best_index])
print("Similarity Score:", best_score)