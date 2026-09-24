from src.rag import build_rag, ask_question


# Build RAG index once
chunks, index = build_rag("data/notes.pdf")

print("RAG system ready!")
print("Total chunks:", len(chunks))
print("Total vectors:", index.ntotal)


# Keep asking questions
while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    answer, retrieved_chunks, scores = ask_question(
        question,
        chunks,
        index
    )

    print("\nFinal Answer:")
    print(answer)

    print("\nSources:")

    for chunk in retrieved_chunks:
        print(
            f"- Page {chunk['page']} "
            f"(Chunk {chunk['chunk_id']})"
        )