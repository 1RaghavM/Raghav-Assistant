import chromadb
import os

MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")

chroma_client = chromadb.PersistentClient()

collection = chroma_client.create_collection(name="My_Collection")

collection.add(
    ids=["id1","id2"],
    documents=[
        "Docuement about me",
        "Document about you"
    ]
)

results = collection.query(
    query_texts=["This is a query about me"],
    n_results=2
)

print(results)