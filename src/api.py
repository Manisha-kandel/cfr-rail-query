"""FastAPI app exposing POST /query and GET /health for the CFR rail RAG system."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chain import get_chain

app = FastAPI(title="CFR Rail Query API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
class QueryRequest(BaseModel):
    question: str
    chat_history: list[dict[str, str]] = []


class SourceItem(BaseModel):
    file: str
    page: int


class TableReference(BaseModel):
    section: str
    file: str
    page: int
    cfr_part: str
    cfr_part_title: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    table_references: list[TableReference]


class HealthResponse(BaseModel):
    status: str


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer a railroad safety question using the RAG chain, with source citations."""
    chain = get_chain(chat_history=request.chat_history)
    result = chain.invoke({"question": request.question})
    sources = [
        SourceItem(
            file=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page", -1),
        )
        for doc in result["context"]
    ]

    table_refs = []
    for doc in result.get("table_references", []):
        meta = doc.metadata
        table_refs.append(TableReference(
            section=meta.get("section") or "unknown",
            file=meta.get("source") or "unknown",
            page=int(meta.get("page", 0)),
            cfr_part=meta.get("cfr_part") or "unknown",
            cfr_part_title=meta.get("cfr_part_title")
                           or "unknown",
        ))

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        table_references=table_refs,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service liveness."""
    return HealthResponse(status="ok")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
