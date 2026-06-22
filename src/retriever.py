"""Load the saved FAISS index and expose retrieval functions, including
two-stage retrieval with CFR internal cross-reference following."""

import hashlib
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from ingest import EMBEDDING_MODEL

load_dotenv()

FAISS_INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"

MAX_REFERENCE_HOPS = 2
MAX_INJECTED_CHUNKS = 4

REFERENCE_PATTERN = re.compile(r"§\s*(\d+\.\d+)(?:\([a-z0-9]+\))*")
IMPLICIT_REF_PATTERN = re.compile(
    r"paragraph\s+(\([a-z0-9]+\)(?:\([a-z0-9]+\))*)"
    r"(?:\s+of\s+this\s+section)?",
    re.IGNORECASE,
)
BASE_SECTION_PATTERN = re.compile(r"(\d+\.\d+)")


def load_vectorstore() -> FAISS:
    """Load the local FAISS index using the locked embedding model."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever(k: int = 4) -> VectorStoreRetriever:
    """Load the local FAISS index and return a retriever for the top-k chunks."""
    return load_vectorstore().as_retriever(search_kwargs={"k": k})


def _content_hash(doc: Document) -> str:
    """Stable content hash used to deduplicate chunks across reference hops."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def _extract_base_sections(text: str) -> set[str]:
    """Find every internal CFR cross-reference (e.g. "§ 232.205(c)") in text
    and normalize each to its base section number (e.g. "232.205"),
    discarding the parenthetical subsection."""
    bases: set[str] = set()
    for match in REFERENCE_PATTERN.finditer(text):
        full_ref = match.group(0).lstrip("§ \t")
        base_match = BASE_SECTION_PATTERN.match(full_ref)
        if base_match:
            bases.add(base_match.group(1))
    return bases


def _extract_implicit_referenced_bases(doc: Document) -> set[str]:
    """Find implicit same-section paragraph references (e.g. "paragraph (c)
    of this section") in a chunk's body text, and resolve each to the
    chunk's own base section number. "This section" can only be resolved
    relative to the chunk's own section metadata, so chunks with unknown
    or missing section metadata are skipped entirely."""
    own_section = doc.metadata.get("section")
    if not own_section or own_section == "unknown":
        return set()

    base_match = BASE_SECTION_PATTERN.match(own_section)
    if not base_match:
        return set()
    base = base_match.group(1)
    if base == "unknown":
        return set()

    bases: set[str] = set()
    for match in IMPLICIT_REF_PATTERN.finditer(doc.page_content):
        target = base + match.group(1)  # e.g. "232.205" + "(c)" = "232.205(c)"
        bases.add(base)
    return bases


def resolve_references(
    initial_chunks: list[Document],
    vectorstore: FAISS,
    max_hops: int = MAX_REFERENCE_HOPS,
    max_injected: int = MAX_INJECTED_CHUNKS,
) -> list[Document]:
    """
    Follow CFR internal cross-references found in retrieved chunks.

    For each hop:
    1. Scan current chunks for section reference patterns
    2. Extract base section numbers (strip parentheticals)
    3. Find chunks in vectorstore whose section metadata
       starts with that base section number
    4. Add new chunks to context, track visited sections
    5. Repeat up to max_hops times
    6. Deduplicate and return merged chunk list

    Two reference forms are scanned per chunk:
    - Explicit "§ XXX.XXX(...)" references (_extract_base_sections), and
    - Implicit same-section references like "paragraph (c) of this
      section" (_extract_implicit_referenced_bases), which restate no
      section number at all and must be resolved relative to the
      chunk's own section metadata.

    Uses prefix matching on the base section number rather than exact
    string match: section metadata records only the trailing endpoint of
    a page (the last "(enhanced display)" footer match), so a reference
    to an earlier subsection on a multi-subsection page (e.g. a body-text
    reference to "232.205(a)") would not exact-match that page's recorded
    section (e.g. "232.205(c)(1)(ii)") even though both belong to the same
    base section. Guards against circular references via a
    visited_sections set, so a section already fetched in an earlier hop
    is never re-fetched.
    """
    merged_chunks: list[Document] = list(initial_chunks)
    seen_hashes = {_content_hash(doc) for doc in merged_chunks}
    visited_sections: set[str] = set()

    current_chunks = initial_chunks
    for _ in range(max_hops):
        referenced_bases: set[str] = set()
        for doc in current_chunks:
            if doc.metadata.get("contains_table"):
                continue  # tables are terminal nodes
            referenced_bases |= _extract_base_sections(doc.page_content)

        # Second pass: implicit same-section paragraph references (e.g.
        # "paragraph (c) of this section") that never restate "§".
        for doc in current_chunks:
            if doc.metadata.get("contains_table"):
                continue  # tables are terminal nodes
            referenced_bases |= _extract_implicit_referenced_bases(doc)

        new_bases = sorted(referenced_bases - visited_sections)
        if not new_bases:
            break

        injected: list[Document] = []
        for base_section in new_bases:
            visited_sections.add(base_section)
            candidates = vectorstore.similarity_search(base_section, k=10)
            for doc in candidates:
                if not (doc.metadata.get("section") or "").startswith(base_section):
                    continue
                content_hash = _content_hash(doc)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)
                injected.append(doc)
                if len(injected) >= max_injected:
                    break
            if len(injected) >= max_injected:
                break

        if not injected:
            break

        merged_chunks.extend(injected)
        current_chunks = injected

    return merged_chunks


def split_prose_and_tables(
    chunks: list[Document],
) -> tuple[list[Document], list[Document]]:
    """
    Separate retrieved chunks into two lists:
    - prose_chunks: safe for LLM answer generation
    - table_chunks: directed to user as source
      references only, never summarized by LLM

    Table chunks are terminal nodes in the reference
    graph — CFR tables never cross-reference other
    tables, so we do not follow references from them.
    """
    prose = []
    tables = []
    for chunk in chunks:
        if chunk.metadata.get("contains_table"):
            tables.append(chunk)
        else:
            prose.append(chunk)
    return prose, tables


def get_enriched_context(
    question: str,
    vectorstore: FAISS,
    k: int = 6,
) -> tuple[list[Document], list[Document]]:
    """
    Full two-stage retrieval pipeline:
    1. Semantic search for top-k chunks
    2. resolve_references() to follow cross-reference
       pointers (skipping table chunks)
    3. split_prose_and_tables() to separate chunks

    Returns:
        (prose_chunks, table_chunks)
        prose_chunks: for LLM answer generation
        table_chunks: for direct user reference only
    """
    initial_chunks = vectorstore.similarity_search(question, k=k)
    merged_chunks = resolve_references(initial_chunks, vectorstore)
    return split_prose_and_tables(merged_chunks)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    query = "What is the maximum allowable gauge for Class 4 track?"
    retriever = get_retriever()
    results = retriever.invoke(query)

    print(f"Query: {query}")
    print(f"Retrieved {len(results)} chunks:\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(
            f"source={doc.metadata.get('source')} page={doc.metadata.get('page')} "
            f"section={doc.metadata.get('section')}"
        )
        print(doc.page_content[:300])
        print()
