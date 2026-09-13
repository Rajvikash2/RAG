import fitz

from src.chunker import create_chunks
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, search_vector_store

pdf = fitz.open("data/notes.pdf")

print(f"Number of pages: {len(pdf)}")

full_text = ""

for page in pdf:
    text = page.get_text()
    full_text += text


created_chunks = create_chunks(
    full_text,
    chunk_size=500,
    overlap=100
)

print("Total chunks:", len(created_chunks))


embeddings = create_embeddings(created_chunks)

print("Embedding shape:", embeddings.shape)

index =  create_vector_store(embeddings)

print("Vector store created with", index.ntotal, "vectors.")

query = "What is interrupt?"

query_embedding = create_embeddings([query])

distances, indices = search_vector_store(index, query_embedding, k=3)

print("\nSearch results:")

for i in range(3):
    chunk_index = indices[0][i]
    print(f"\nResult {i + 1}")
    print("Chunk index:", chunk_index)
    print("Distance:", distances[0][i])
    print("Text:")
    print(created_chunks[chunk_index])
    