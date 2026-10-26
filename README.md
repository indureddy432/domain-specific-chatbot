## 1. Project Overview

The Domain-Specific RAG Chatbot is an AI-based application that allows users to upload PDF documents and ask questions about their content. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate context-based answers.

The system helps users find information quickly without manually searching through lengthy PDF documents.

## 2. Objectives

- To extract text from uploaded PDF documents.
- To divide extracted text into smaller chunks for efficient processing.
- To generate text embeddings using Sentence Transformers.
- To retrieve relevant information using FAISS similarity search.
- To generate answers using a Large Language Model (LLM).
- To display source document names and page numbers with answers.
- To reduce unsupported answers by instructing the model to use the retrieved document context.

## 3. Technologies Used

- **Programming Language:** Python
- **User Interface:** Streamlit
- **PDF Processing:** pypdf
- **Text Chunking:** LangChain Text Splitters
- **Embedding Model:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Search:** FAISS
- **Language Model API:** Groq
- **Environment Configuration:** python-dotenv

## 4. Features

- Upload one or multiple PDF documents.
- Extract text page by page.
- Split text into overlapping chunks.
- Generate embeddings for document chunks.
- Retrieve relevant passages based on user questions.
- Generate answers using retrieved context.
- Display document names and page numbers as source references.
- Maintain chat history during the session.
- Clear the chat when required.
- Display a fallback message when the information is unavailable in the retrieved context.

## 5. System Workflow

1. The user uploads PDF documents.
2. The application extracts text from the PDF pages.
3. The extracted text is divided into chunks.
4. Sentence Transformers converts the chunks into embeddings.
5. FAISS indexes the embeddings for similarity search.
6. The user enters a question.
7. FAISS retrieves the most relevant chunks.
8. The retrieved context is passed to the Groq language model.
9. The model generates an answer based on the supplied context.
10. The application displays the answer and its source references.

## 6. Project Structure

```text
domain_rag_chatbot/
│
├── app.py
├── document_loader.py
├── vector_store.py
├── rag_pipeline.py
├── prompt.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── documents/
├── vector_store/
└── tests/
    └── test_questions.csv
```

## 7. Installation and Setup

### Prerequisites

- Python installed on your system
- Visual Studio Code
- A Groq API key

### Step 1: Open the project folder

Open the project folder in Visual Studio Code.

### Step 2: Create a virtual environment

Open the VS Code terminal and run:

```powershell
python -m venv .venv
```

### Step 3: Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 5: Configure the API key

Create or open the `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your actual API key. Do not upload the `.env` file to GitHub.

### Step 6: Run the application

```powershell
python -m streamlit run app.py --server.fileWatcherType none
```

The application will open in your web browser.

## 8. How to Use the Application

1. Open the chatbot application.
2. Upload one or more PDF documents using the sidebar.
3. Click **Process Documents**.
4. Wait for text extraction, embedding generation, and indexing to finish.
5. Enter a question related to the uploaded documents.
6. Read the generated answer and check the source document and page number.
7. Use **Clear Chat** to clear the conversation.

## 9. Example Use Case

Suppose a student uploads lecture notes in PDF format and asks:

**Question:** What is hypothesis testing?

The chatbot searches the uploaded document for relevant passages and uses the retrieved context to generate an answer. Source information helps the student locate the relevant page in the PDF.

## 10. Limitations

- Answer quality depends on the quality of the extracted text and the relevance of retrieved chunks.
- Incorrect or incomplete document extraction may affect the generated answers.
- The language model may still produce inaccurate answers despite context-based prompting.
- Scanned handwritten PDFs require OCR integration, and handwriting recognition may not be fully accurate.
- Access to the Groq API requires a valid API key and an available model.

## 11. Future Enhancements

- Improve OCR support for scanned handwritten PDFs.
- Add support for more document formats.
- Improve retrieval accuracy and answer evaluation.
- Add conversation export and document management.
- Deploy the application online for wider access.

## 12. Conclusion

The Domain-Specific RAG Chatbot demonstrates how Artificial Intelligence, semantic search, and Large Language Models can be combined to make document-based information retrieval faster, easier, and more efficient.

## 13. Author

Developed as an AI/ML project to explore Retrieval-Augmented Generation and intelligent document question answering.