def create_chunks(pages, chunk_size=500, overlap=100):
    chunks = []

    chunk_id = 0

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_number,
                "text": chunk_text
            })

            chunk_id += 1

            start += chunk_size - overlap

    return chunks