import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf_document(file_path: str) -> list[Document]:
    """
    Reads a PDF file from disk, extracts all text content page-by-page,
    and splits it into overlapping contextual chunks ready for vector embedding.

    Args:
        file_path: Absolute path to the PDF file on disk.

    Returns:
        A list of LangChain Document objects, each representing a text chunk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at path: {file_path}")

    print(f"[PDF] Parsing: {os.path.basename(file_path)}")

    # Extract raw text from each page of the PDF
    reader = PdfReader(file_path)
    raw_pages = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            raw_pages.append(
                Document(
                    page_content=text,
                    metadata={"source": os.path.basename(file_path), "page": page_num + 1},
                )
            )

    if not raw_pages:
        raise ValueError(f"No extractable text found in: {file_path}")

    print(f"[OK] Extracted text from {len(raw_pages)} pages.")

    # Split into overlapping chunks for better semantic retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = splitter.split_documents(raw_pages)

    print(f"[SPLIT] Split into {len(chunks)} contextual chunks.")
    return chunks
