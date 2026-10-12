import streamlit as st

from src.rag import build_rag, ask_question


st.title("📄 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its contents."
)


# -------------------------
# PDF Upload
# -------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    current_file = uploaded_file.name

    needs_processing = (
        "index" not in st.session_state
        or
        st.session_state.get("file_name") != current_file
    )

    if needs_processing:

        pdf_bytes = uploaded_file.getvalue()

        with st.spinner("Processing PDF..."):

            collection = build_rag(pdf_bytes)

        st.session_state.collection = collection
        st.session_state.file_name = current_file

        st.success(
            f"PDF processed successfully! "
            f"Created {collection.count()} chunks in ChromaDB."
        )

    else:

        st.success(
            f"{current_file} is already processed."
        )


# -------------------------
# Question
# -------------------------

if "collection" in st.session_state:

    st.divider()

    question = st.text_input(
        "Ask a question about the document"
    )

    if question:

        with st.spinner("Thinking..."):

            answer, retrieved_chunks, scores = ask_question(
                question,
                st.session_state.collection
                
            )

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Sources")

        shown_pages = set()
        
        for chunk in retrieved_chunks:
        
            page = chunk["page"]
        
            if page not in shown_pages:
                st.write(f"📄 Page {page}")
                shown_pages.add(page)