
from io import BytesIO
from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from every page of an uploaded PDF.
    Preserve the document name and page number.
    Skip pages that contain no extractable text.
    """

    documents = []
    filename = uploaded_file.name

    try:
        pdf_bytes = uploaded_file.getvalue()
        reader = PdfReader(BytesIO(pdf_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()

            # Skip empty pages safely
            if not text:
                continue

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_number,
                    },
                }
            )

    except Exception as error:
        raise ValueError(
            f"Could not read {filename}: {error}"
        ) from error

    return documents


def extract_text_from_pdfs(uploaded_files):
    """
    Extract text from multiple uploaded PDF files.
    """

    all_documents = []

    for uploaded_file in uploaded_files:
        extracted_pages = extract_text_from_pdf(uploaded_file)
        all_documents.extend(extracted_pages)

    if not all_documents:
        raise ValueError(
            "No extractable text was found in the uploaded PDFs. "
            "Scanned PDFs may require optional OCR support."
        )

    return all_documents
