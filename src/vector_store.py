import faiss


def create_vector_store(embeddings):
    dimension = embeddings.shape[1]

    # Normalize the document embeddings
    faiss.normalize_L2(embeddings)

    # Inner Product on normalized vectors = Cosine Similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_vector_store(index, query_embedding, top_k=3):
    # Normalize the query embedding
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    return scores, indices