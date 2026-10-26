
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class DocumentVectorStore:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.documents = []

    def build(self, documents):
        """
        Create embeddings for document chunks and store them in FAISS.
        Each document must contain text and metadata.
        """

        if not documents:
            raise ValueError("No document chunks provided.")

        self.documents = documents
        texts = [document["text"] for document in documents]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Inner product on normalized vectors gives cosine similarity.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, question, top_k=5):
        """
        Return the most relevant document chunks for a question.
        """

        if self.index is None:
            raise ValueError(
                "The vector store has not been built yet."
            )

        question_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        question_embedding = np.asarray(
            question_embedding,
            dtype=np.float32,
        )

        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(question_embedding, k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            result = dict(self.documents[index])
            result["score"] = float(score)
            results.append(result)

        return results

    def save(self, directory="vector_store/saved_index"):
        """Save the FAISS index and document metadata locally."""

        import json
        import os

        if self.index is None:
            raise ValueError("Build the vector store before saving.")

        os.makedirs(directory, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(directory, "index.faiss"),
        )

        with open(
            os.path.join(directory, "documents.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(self.documents, file, ensure_ascii=False, indent=2)

    def load(self, directory="vector_store/saved_index"):
        """Load a previously saved FAISS index and its metadata."""

        import json
        import os

        index_path = os.path.join(directory, "index.faiss")
        documents_path = os.path.join(directory, "documents.json")

        if not os.path.exists(index_path) or not os.path.exists(
            documents_path
        ):
            raise FileNotFoundError(
                "Saved vector store files were not found."
            )

        self.index = faiss.read_index(index_path)

        with open(documents_path, "r", encoding="utf-8") as file:
            self.documents = json.load(file)
