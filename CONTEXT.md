# CONTEXT.md — cfr-rail-query

This file is the single source of truth for this project.
Read this before writing or modifying any file.
Do not deviate from decisions made here without explicit instruction.

---

## What This Project Is

A Retrieval-Augmented Generation (RAG) application for railroad industry
professionals (inspectors, locomotive engineers, roadway workers, etc.).

Users ask natural language questions about federal railroad safety regulations.
The system retrieves the most relevant chunks from a FAISS vector index built
on 49 CFR regulatory documents, and an LLM generates an answer grounded
strictly in those chunks.

No hallucination. No interpretation. Every answer is traceable to a source
regulation.

---

## Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| LangChain | latest | RAG chain orchestration |
| FAISS | latest | Vector index, local, no vendor lock-in |
| OpenAI GPT-4o-mini | latest | LLM — upgraded from GPT-3.5-turbo due to cross-chunk fusion hallucination that could not be resolved via prompt engineering alone |
| OpenAIEmbeddings (text-embedding-3-small) | latest | Embedding model |
| FastAPI | latest | Backend API |
| Streamlit | latest | Frontend chat UI |
| Docker + docker-compose | latest | Containerisation |
| Hugging Face Spaces | — | Deployment target |

---

## Corpus

Source: 49 CFR (Code of Federal Regulations) — Title 49: Transportation
Format: PDFs downloaded from the official eCFR or FRA website
Location in project: `data/raw/`

### Selected CFR Parts:

| Part | Topic |
|---|---|
| 213 | Track Safety Standards |
| 214 | Roadway Worker Safety |
| 229 | Locomotive Safety Standards |
| 232 | Brake System Safety Standards |
| 225 | Railroad Accident/Incident Reporting |
| 234 | Grade Crossing Signal Systems |

Total estimated corpus: ~320 pages across 6 parts.
Part 236 (Signal & Train Control) was excluded — too technically dense,
poor chunking characteristics.

---

## Folder Structure

```
cfr-rail-query/
  data/
    raw/              ← downloaded 49 CFR PDFs go here
    processed/        ← chunked text after ingestion (if saved)
  src/
    ingest.py         ← PDF loading, chunking, FAISS index build
    retriever.py      ← FAISS search, returns top-k chunks
    chain.py          ← LangChain chain: retriever + prompt + LLM
    api.py            ← FastAPI app, POST /query endpoint
    app.py            ← Streamlit chat UI
  faiss_index/        ← saved FAISS index (gitignored if large)
  Dockerfile
  docker-compose.yml
  requirements.txt
  README.md
  .env.example        ← OPENAI_API_KEY placeholder, never commit .env
  .gitignore
  CONTEXT.md          ← this file
```

---

## Chunking Strategy

- Loader: `pdfplumber` (replaced PyPDFLoader — cleaner text extraction,
  especially for tables)
- Splitter: `RecursiveCharacterTextSplitter`
- `separators=["\n\n", "\n", " ", ""]`
- `chunk_size=1000`
- `chunk_overlap=100`
- Each chunk metadata must carry: `source` (filename), `page` (page number),
  `cfr_part`, `cfr_part_title`, `section`, `contains_table`
- Target chunk count: 500–2000 chunks for this corpus

Rationale: regulatory sentences are long and cross-reference conditions and
consequences within the same provision. A 500-character chunk often cuts a
regulation before both halves are captured; 1000/100 keeps related clauses
together while still sliding context across boundaries.

| Field | Description | Values |
|---|---|---|
| `contains_table` | Detected by pdfplumber `lines`/`lines` strategy | `True`/`False` |

**Table Detection and Markdown Conversion:** During page loading, each page
is checked for bordered tables using pdfplumber's
`extract_tables(vertical_strategy="lines", horizontal_strategy="lines")`.
This strategy detects tables with ruled borders and zero false positives on
prose pages (confirmed by corpus-wide audit of 416 pages).

If tables are detected:
- `contains_table` metadata is set to `True`
- Tables are converted to markdown via `table_to_markdown()` and appended
  to page text
- Cell newlines are stripped to preserve markdown row structure

Pages without ruled borders (e.g. §213.233(c) inspection frequency table,
§213.113 defect remedial action table) return `contains_table=False` — see
Known Limitations.

### Metadata Enrichment

`enrich_metadata()` runs once per page in `load_pdfs()`'s output, before
chunking — not per-chunk after splitting. It attaches:

- `cfr_part` — parsed from the filename (e.g. `49_CFR_213_260617.pdf` → `"213"`)
- `cfr_part_title` — looked up from the static `CFR_PART_TITLES` dict below
- `section` — the CFR section number (e.g. `"232.714(b)"`), extracted via
  regex from eCFR's auto-generated `(enhanced display)` footer line present
  on most pages; `None` if the page has no such footer (e.g. table-of-contents
  pages)

Because enrichment runs pre-split, every chunk derived from a page inherits
that page's `cfr_part`/`cfr_part_title`/`section` automatically via
LangChain's metadata propagation — no per-chunk extraction needed.

Rationale: deterministic regex + static dict lookup, zero extra API calls
(no LLM classification, no graph traversal) — keeps cost down while giving
the retriever/prompt richer per-chunk identity than just `source`+`page`.

```python
CFR_PART_TITLES = {
    "213": "Track Safety Standards",
    "214": "Roadway Worker Safety",
    "225": "Railroad Accident/Incident Reporting",
    "229": "Locomotive Safety Standards",
    "232": "Brake System Safety Standards",
    "234": "Grade Crossing Signal Systems",
}
```

---

## Retrieval Strategy

- Vector store: FAISS, saved locally to `faiss_index/`
- Embeddings: `OpenAIEmbeddings(model="text-embedding-3-small")` — same model
  constant (`EMBEDDING_MODEL` in `ingest.py`, imported by `retriever.py`) must
  be used for both index build and query time, or retrieval silently breaks
- Simple top-k retriever exposed via `get_retriever(k=4)` in `retriever.py`
  (kept for standalone testing; the chain uses two-stage retrieval below)

### Two-Stage Retrieval with Cross-Reference Following

Motivation: CFR sections routinely cross-reference other sections (e.g. a
brake-test section says "as described in paragraph (c)" or "in accordance
with § 213.305"), and the referenced section is often not among the
initial top-k semantic-search results. Left unresolved, the LLM either
omits that content or — worse — fills the gap from its own pretrained
knowledge (a confirmed hallucination pattern; see chain.py revision
history for the Class 1 brake test and slow-order test failures).

Pipeline, in `retriever.py`:

1. `get_enriched_context(question, vectorstore, k=6)` runs an initial
   semantic search for the top-`k` chunks.
2. `resolve_references(initial_chunks, vectorstore)` scans those chunks'
   body text for two reference forms — explicit `§ XXX.XXX(...)` patterns,
   and implicit same-section references like "paragraph (c) of this
   section" (resolved against the chunk's own `section` metadata, since
   they restate no section number at all) — and for each new referenced
   base section, fetches matching chunks from the vectorstore and merges
   them into the context. **Table chunks are skipped entirely during
   reference scanning** — they are terminal nodes (see Table Handling
   Design below).
3. Repeats up to `MAX_REFERENCE_HOPS = 2` times (a reference found in an
   injected chunk can trigger one further hop), capping newly injected
   chunks at `MAX_INJECTED_CHUNKS = 4` per hop, to bound both API cost and
   context size.
4. A `visited_sections` set prevents re-fetching the same base section and
   guards against circular references (e.g. § A referencing § B which
   references § A back).
5. Final chunk list is deduplicated by a SHA-256 hash of `page_content`.
6. `split_prose_and_tables()` separates the deduplicated list into
   `prose_chunks` (passed to the LLM for answer generation) and
   `table_chunks` (returned as `table_references`, never summarized).

**Prefix matching, not exact match:** `enrich_metadata()`'s `section`
field records only the trailing endpoint of a page — the last
`(enhanced display)` footer match found in that page's text. A page can
span multiple subsections (e.g. a page might run from `232.205(a)(5)(v)(A)`
at the top to `232.205(c)(1)(ii)` at the footer), so a body-text reference
to an earlier subsection on that page (`232.205(a)`) would not exact-match
the page's recorded section. Matching is therefore done on the **base**
section number only (e.g. `232.205`, parentheticals stripped) via
`str.startswith()`, against candidate chunks fetched by a semantic search
on the base section string itself (the vectorstore has no direct
metadata-filter API, so candidates are over-fetched at `k=10` and then
filtered by prefix match).

---

## Table Handling Design

CFR regulatory tables contain the core values inspectors need most — speed
limits, gauge limits, inspection frequencies, brake pressures. However,
tables with inconsistent border structures cannot be reliably extracted
and comprehended by the LLM.

Design decision: do not attempt to summarize table content. Instead direct
users explicitly to the source section and page.

Why:
- Real regulatory tools (LexisNexis, Westlaw) direct users to source
  documents for safety-critical table values
- LLM misreading a speed limit or gauge tolerance in a safety domain is
  worse than no answer
- The system should be honest about what it can and cannot do

How it works:
1. Retrieved chunks are split into `prose_chunks` and `table_chunks` by
   `split_prose_and_tables()`
2. Only `prose_chunks` go to the LLM for answer generation
3. `table_chunks` are returned as `table_references` in the API response
   with `section`, `file`, `page`, `cfr_part`, `cfr_part_title` metadata
4. The UI shows a "See Table" card with PDF reference for each
   `table_reference`
5. Table chunks are terminal nodes — `resolve_references()` skips them
   since CFR tables never cross-reference other tables

Rule 7 in the prompt instructs the LLM to explicitly direct users to the
source CFR section when table values are needed but not in prose context.

---

## Known Limitations

1. Tables without full ruled borders cannot be detected by the
   `lines`/`lines` extraction strategy. Known affected sections:
   - §213.233(c) — inspection frequency table (header/column separators
     only, no row borders)
   - §213.113(c) — defect remedial action table (near-empty extraction,
     likely image-rendered)

   Queries about values in these specific tables will trigger Rule 7,
   directing users to the source document.

2. Section metadata records the trailing endpoint of each page (last
   footer match). Pages spanning multiple subsections only record the
   final one. Cross-reference resolution uses prefix matching to
   compensate (see Retrieval Strategy above).

3. Part 214's multi-page protection distance table spans pages 33–52.
   The `lines`/`lines` strategy detects it correctly across all pages.

4. pdfplumber cannot extract image-rendered tables or tables embedded as
   graphics in the PDF. Future fix: OCR layer for image pages.

---

## Prompt Template (LOCKED — do not modify without explicit instruction)

This prompt lives in `src/chain.py`. It is the primary enforcement
mechanism for all behavioral rules.

```
You are a regulatory reference assistant for the railroad industry,
specializing in 49 CFR federal railroad safety standards.

RULES YOU MUST FOLLOW:

1. Answer ONLY using the context provided below. Do not use any
outside knowledge, assumptions, or information not present in
the retrieved text.

2. Every answer must cite the CFR Part and Section number.
For example: "According to 49 CFR §213.233..."
Never give an answer without a citation.

3. If the retrieved context only partially answers the question,
clearly state: "I have partial information from [section].
The full regulation may contain additional requirements.
Please verify the complete section."

4. Do not interpret, infer, or give opinions beyond what is
explicitly stated in the regulatory text. Report only what
the regulation says.

4b. Never combine facts from different CFR sections or
chunks into a single synthesized claim. If two retrieved
chunks contain related but separate facts, present them
separately with their individual citations. Do not imply
a causal or conditional link between facts from different
sections unless that link is explicitly stated in the
retrieved text.

4c. Never combine facts from different paragraphs of
the same CFR section into a single synthesized claim.
If paragraph (b) states one rule and paragraph (c)
states a different rule, present them as separate facts
with separate paragraph-level citations. Do not merge
numbers or conditions from different paragraphs into
one claim.

5. If the question is unrelated to railroad safety regulations
or outside the scope of the loaded CFR documents, respond only
with: "This question is outside the scope of the loaded CFR
railroad safety documents."

6. End every answer with this disclaimer on a new line:
"⚠️ This answer is generated from CFR text for reference
purposes only. Always verify against the official current
CFR before making any operational decisions."

7. If the answer requires specific numerical values,
limits, frequencies, or thresholds that are typically
presented in regulatory tables (such as speed limits
by track class, gauge limits by class, inspection
frequencies, or brake pressure values), and the
retrieved context does not contain those values as
clear prose statements, state what the relevant
regulation covers and explicitly direct the user to
consult the specific CFR section directly in the
source document. Never guess or approximate a table
value.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
```

---

## Ground Rules (Behavioral Requirements)

These are non-negotiable. Every file must respect these:

| Rule | Description | Enforced By |
|---|---|---|
| No hallucination | Answer only from retrieved chunks | Prompt |
| Mandatory citation | Every answer cites CFR Part + Section | Prompt + metadata |
| Partial answer flagging | Flag if context is incomplete | Prompt |
| No interpretation | No opinions or inference beyond text | Prompt |
| Out of scope detection | Reject non-railroad questions | Prompt |
| Disclaimer | Every answer ends with warning | Prompt |
| Version banner | UI shows which CFR year is loaded | Streamlit UI |

---

## API Contract

### POST /query
Request:
```json
{"question": "string"}
```
Response:
```json
{
  "answer": "string",
  "sources": [
    {
      "file": "string",
      "page": int
    }
  ],
  "table_references": [
    {
      "section": "string",
      "file": "string",
      "page": int,
      "cfr_part": "string",
      "cfr_part_title": "string"
    }
  ]
}
```

### GET /health
Response:
```json
{"status": "ok"}
```

---

## UI Requirements (Streamlit)

- Simple chat interface: text input at bottom, conversation history above
- Each answer displayed with source citations in a collapsed expander
- Banner at top: "Knowledge base: 49 CFR Parts 213, 214, 229, 232, 225, 234 — [year]"
- Session state used to maintain conversation history across reruns
- Calls FastAPI via `requests.post("http://localhost:8000/query", ...)`

---

## Coding Standards

- Python 3.11
- Type hints on all functions
- Docstrings on all functions
- One file does one thing — no mixing responsibilities
- Each src/ file is independently runnable for testing
- All secrets via `.env` file — never hardcoded
- `.env` is gitignored, `.env.example` is committed

---

## Build Order (follow this sequence strictly)

1. Folder structure + requirements.txt + .env.example + .gitignore
2. `src/ingest.py` — load PDFs, chunk, build FAISS index
3. `src/retriever.py` — load index, expose get_retriever()
4. `src/chain.py` — prompt + chain, expose get_chain()
5. `src/api.py` — FastAPI endpoints
6. `src/app.py` — Streamlit UI
7. Dockerfile + docker-compose.yml
8. README.md

Do not skip steps. Do not combine steps.
Do not make large end-to-end changes.
One file at a time. Review before moving forward.

---

## Sample Questions (for testing)

1. "My track is Class 3, how often must I do a walking inspection?"
2. "What is the maximum allowable gauge for Class 4 track?"
3. "What are the speed limits for Class 2 freight track?"
4. "When does a defect require immediate action vs a slow order?"
5. "What PPE is required when working within 25 feet of live track?"
6. "What is the minimum distance I need from a moving train as a roadway worker?"
7. "When is an on-track safety briefing required?"
8. "What qualifications does a flagman need?"
9. "What are the rules for lone worker protection?"
10. "What daily inspection items are required before operating a locomotive?"
11. "When must a locomotive be removed from service for brake defects?"
12. "What are the headlight visibility requirements for a locomotive?"
13. "How often must a locomotive undergo a periodic inspection?"
14. "What percentage of brakes must be operative on a freight train?"
15. "When is a single car air brake test required?"
16. "What are the rules for a Class 1 brake test?"
17. "How long can a train sit before requiring a new brake test?"
18. "What are the response time requirements when a crossing signal malfunctions?"
19. "Who must be notified when a crossing warning system fails?"
20. "What injuries must be reported to FRA and within what timeframe?"
21. "What is the monetary threshold for reporting a rail equipment accident?"
22. "When is a post-accident drug and alcohol test mandatory?"
23. "What are the blue flag protection rules?"
24. "What are the rules for shoving movements?"
25. "When is a job briefing mandatory before a railroad operation?"
