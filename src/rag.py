from src.pdf_loader import extract_pages
from src.chunker import create_chunks
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, search_vector_store
from src.llm import generate_answer


def build_rag(pdf_source):
    """
    Process a PDF and build the vector store.
    """

    pages = extract_pages(pdf_source)

    chunks = create_chunks(
        pages,
        chunk_size=500,
        overlap=100
    )

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(chunk_texts)

    index = create_vector_store(embeddings)

    return chunks, index


def ask_question(question, chunks, index, top_k=3):
    """
    Search the vector store and generate an answer.
    """

    # 1. Create embedding for the question
    query_embedding = create_embeddings([question])

    # 2. Search FAISS
    scores, indices = search_vector_store(
        index,
        query_embedding,
        top_k=top_k
    )

    # 3. Get the actual chunks
    retrieved_chunks = []

    for i in range(top_k):

        chunk_index = indices[0][i]

        chunk = chunks[chunk_index]

        retrieved_chunks.append(chunk)

    # 4. Create context for Gemini
    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    # 5. Generate answer
    answer = generate_answer(
        context,
        question
    )

    return answer, retrieved_chunks, scores