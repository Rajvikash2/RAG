
import hashlib
from pathlib import Path

from src.pdf_loader import extract_pages
from src.chunker import create_chunks
from src.embeddings import create_embeddings
from src.vector_store import get_collection, store_chunks, search_chunks
from src.llm import generate_answer


def build_rag(pdf_source):
    # Get PDF bytes to identify the document
    if isinstance(pdf_source, str):
        pdf_bytes = Path(pdf_source).read_bytes()
        loader_source = pdf_source
    else:
        pdf_bytes = pdf_source
        loader_source = pdf_source

    document_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    collection_name = f"pdf_{document_hash}"

    # Reuse the collection if this PDF was already indexed
    collection = get_collection(collection_name)

    if collection.count() == 0:
        pages = extract_pages(loader_source)

        chunks = create_chunks(
            pages,
            chunk_size=500,
            overlap=100
        )

        if not chunks:
            raise ValueError("No text could be extracted from this PDF.")

        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = create_embeddings(chunk_texts)

        store_chunks(
            collection,
            chunks,
            embeddings
        )

    return collection


def ask_question(question, collection, top_k=3):
    query_embedding = create_embeddings([question])

    results = search_chunks(
        collection,
        query_embedding,
        top_k=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieved_chunks = [
        {
            "text": document,
            "page": metadata["page"],
            "chunk_id": metadata["chunk_id"]
        }
        for document, metadata in zip(documents, metadatas)
    ]

    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    answer = generate_answer(context, question)

    return answer, retrieved_chunks, results["distances"]
