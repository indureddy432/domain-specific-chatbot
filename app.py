
import streamlit as st

from rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📚",
    layout="wide",
)


st.title("📚 Domain-Specific RAG Chatbot")
st.write(
    "Upload PDF documents and ask questions. "
    "Answers are generated using retrieved content from your documents."
)


# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# Sidebar: upload and process PDFs
with st.sidebar:
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based PDF documents.",
    )

    if uploaded_files:
        st.write("Selected files:")

        for uploaded_file in uploaded_files:
            st.write(f"📄 {uploaded_file.name}")

    process_clicked = st.button(
        "Process Documents",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if process_clicked:
        try:
            with st.spinner(
                "Extracting text, creating embeddings, and indexing PDFs..."
            ):
                pipeline = RAGPipeline()
                result = pipeline.process_pdfs(uploaded_files)

                st.session_state.pipeline = pipeline
                st.session_state.messages = []
                st.session_state.processed_files = result["documents"]

            st.success("Documents processed successfully!")
            st.write(f"Pages processed: {result['pages']}")
            st.write(f"Text chunks created: {result['chunks']}")

        except Exception as error:
            st.error(f"Could not process the documents: {error}")

    st.divider()

    if st.session_state.processed_files:
        st.subheader("Processed Documents")

        for filename in st.session_state.processed_files:
            st.write(f"📄 {filename}")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(
                        f"📄 {source['source']} — Page {source['page']}"
                    )


# Ask questions
if st.session_state.pipeline is None:
    st.info("Upload your PDFs in the sidebar and click Process Documents.")

else:
    question = st.chat_input("Ask a question about your uploaded PDFs...")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your documents..."):
                    answer, sources = (
                        st.session_state.pipeline.answer_question(question)
                    )

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(
                                f"📄 {source['source']} — "
                                f"Page {source['page']}"
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                st.error(f"Could not generate an answer: {error}")
