"""Compose the retriever, locked prompt, and LLM into a RAG chain via get_chain()."""

import sys

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from retriever import get_enriched_context, load_vectorstore

load_dotenv()

PROMPT_TEMPLATE = """You are a regulatory reference assistant for the railroad industry,
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

ANSWER:"""

REWRITE_PROMPT = """You are helping rewrite follow-up
questions for a railroad safety regulation assistant.

Given the conversation history below and a follow-up
question, rewrite the follow-up as a complete,
self-contained question that:
1. Preserves the specific regulatory topic from
   the conversation (e.g. PPE, brake tests,
   inspection frequencies)
2. Incorporates any pronouns or references
   ("that", "it", "those", "what about X")
   by replacing them with the actual subject
   from the history
3. Does NOT introduce new topics not present
   in either the history or the follow-up

If the follow-up is already self-contained and
references no prior context, return it exactly as-is.

Conversation History:
{chat_history}

Follow-up Question: {question}

Rewritten Standalone Question (one sentence only,
no preamble):"""


def format_docs(docs: list[Document]) -> str:
    """Join retrieved chunk texts into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def rewrite_question(
    question: str,
    chat_history: list[dict[str, str]],
) -> str:
    """
    Rewrite a follow-up question as a standalone
    question using conversation history.

    If no history exists or question is already
    standalone, returns question unchanged.

    Uses GPT-4o-mini with temperature=0 for
    deterministic rewrites.

    Args:
        question: the raw user question
        chat_history: list of dicts with
                     "role" and "content" keys,
                     last 5 exchanges maximum

    Returns:
        Rewritten standalone question string
    """
    if not chat_history:
        return question

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Format history as readable string
    history_str = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long assistant answers to first 200 chars
        # to keep rewrite prompt short
        content = msg["content"]
        if msg["role"] == "assistant" and len(content) > 200:
            content = content[:200] + "..."
        history_str += f"{role}: {content}\n"

    rewrite_prompt = PromptTemplate.from_template(REWRITE_PROMPT)
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    rewritten = rewrite_chain.invoke({
        "chat_history": history_str.strip(),
        "question": question,
    })
    return rewritten.strip()


def get_chain(
    k: int = 6,
    chat_history: list[dict[str, str]] | None = None,
) -> Runnable:
    """Build the RAG chain: retriever + locked prompt + LLM.

    Invoking with {"question": "..."} returns a dict with:
        "question": str
        "rewritten_question": str            # standalone form of question
        "answer": str
        "context": list[Document]            # prose chunks
        "table_references": list[Document]   # table chunks
        "_retrieved": tuple                  # internal, ignore
    """
    vectorstore = load_vectorstore()
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    answer_chain = prompt | llm | StrOutputParser()

    chat_history = chat_history or []

    return RunnablePassthrough.assign(
        rewritten_question=lambda x: rewrite_question(
            x["question"], chat_history
        )
    ).assign(
        _retrieved=lambda x: get_enriched_context(
            x["rewritten_question"], vectorstore, k=k
        )
    ).assign(
        context=lambda x: x["_retrieved"][0],
        table_references=lambda x: x["_retrieved"][1],
    ).assign(
        answer=lambda x: answer_chain.invoke(
            {
                "context": format_docs(x["context"]),
                "question": x["rewritten_question"]
            }
        )
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    chain = get_chain()
    question = "What is the maximum allowable gauge for Class 4 track?"
    result = chain.invoke({"question": question})

    print(f"Question: {question}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources:")
    for doc in result["context"]:
        print(
            f"  - {doc.metadata.get('source')} page={doc.metadata.get('page')} "
            f"section={doc.metadata.get('section')}"
        )
