"""Load 49 CFR PDFs from data/raw/, chunk them, and build a local FAISS index."""

import re
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FAISS_INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"

SEPARATORS = ["\n\n", "\n", " ", ""]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "text-embedding-3-small"

CFR_PART_TITLES = {
    "213": "Track Safety Standards",
    "214": "Roadway Worker Safety",
    "225": "Railroad Accident/Incident Reporting",
    "229": "Locomotive Safety Standards",
    "232": "Brake System Safety Standards",
    "234": "Grade Crossing Signal Systems",
}

SECTION_FOOTER_PATTERN = re.compile(
    r"49 CFR (\d+\.\d+(?:\([a-z0-9]+\))*) \(enhanced display\)"
)
PART_FROM_FILENAME_PATTERN = re.compile(r"_CFR_(\d+)_")


def table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to markdown table format."""
    if not table:
        return ""
    rows = []
    for i, row in enumerate(table):
        # Replace None cells with empty string
        clean_row = [
            (cell or "").replace("\n", " ").strip()
            for cell in row
        ]
        rows.append("| " + " | ".join(clean_row) + " |")
        # Add header separator after first row
        if i == 0:
            rows.append("|" + "|".join(["---" for _ in clean_row]) + "|")
    return "\n".join(rows)


def load_pdfs(raw_dir: Path) -> list[Document]:
    """Load all PDFs from raw_dir using pdfplumber for clean text extraction including tables."""
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    continue

                try:
                    tables = page.extract_tables(
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
                    )
                    contains_table = len(tables) > 0
                except Exception:
                    tables = []
                    contains_table = False

                for table in tables:
                    markdown_table = table_to_markdown(table)
                    if markdown_table:
                        text = text + "\n\n" + markdown_table

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                            "contains_table": contains_table,
                        },
                    )
                )
    return documents


def enrich_metadata(documents: list[Document]) -> list[Document]:
    """Attach cfr_part, cfr_part_title, and section to each page Document in place."""
    for doc in documents:
        part_match = PART_FROM_FILENAME_PATTERN.search(doc.metadata["source"])
        part = part_match.group(1) if part_match else "unknown"

        section_matches = SECTION_FOOTER_PATTERN.findall(doc.page_content)
        section = section_matches[-1] if section_matches else None

        doc.metadata["cfr_part"] = part
        doc.metadata["cfr_part_title"] = CFR_PART_TITLES.get(part, "Unknown")
        doc.metadata["section"] = section
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks per the locked chunking strategy.

    Also corrects contains_table per chunk: the page-level flag set in
    load_pdfs() is inherited by every chunk split from that page, even
    chunks that are pure prose with no actual table content.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        # Override page-level contains_table with
        # accurate chunk-level detection.
        # table_to_markdown() always produces |---|
        # separator rows — use that as the signal.
        chunk.metadata["contains_table"] = (
            "|---|" in chunk.page_content
            or "|---" in chunk.page_content
        )

    return chunks


def build_faiss_index(chunks: list[Document], index_dir: Path) -> None:
    """Embed chunks with OpenAIEmbeddings and persist a FAISS index to index_dir."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))


def main() -> None:
    """Run the full ingestion pipeline: load, enrich, chunk, embed, save."""
    print(f"Loading PDFs from {DATA_RAW_DIR} ...")
    documents = load_pdfs(DATA_RAW_DIR)
    print(f"Loaded {len(documents)} pages.")

    print("Enriching metadata...")
    documents = enrich_metadata(documents)

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Building FAISS index at {FAISS_INDEX_DIR} ...")
    build_faiss_index(chunks, FAISS_INDEX_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
