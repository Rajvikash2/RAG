
import chromadb

# Store the database inside the project directory
client = chromadb.PersistentClient(
    path="./chroma_db"
)


def get_collection(collection_name):
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def store_chunks(collection, chunks, embeddings):
    collection.upsert(
        ids=[
            str(chunk["chunk_id"])
            for chunk in chunks
        ],
        documents=[
            chunk["text"]
            for chunk in chunks
        ],
        metadatas=[
            {
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in chunks
        ],
        embeddings=embeddings.tolist()
    )


def search_chunks(collection, query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return results
