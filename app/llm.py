"""Answer synthesis with Claude.

Retrieved chunks are formatted into a numbered context block; Claude is
instructed to answer *only* from that context and to cite sources with ``[n]``
markers, or to say it doesn't know. Uses the Anthropic Python SDK with adaptive
thinking, the ``effort`` control, and streaming (recommended for any request
that may produce a long answer).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.config import settings
from app.retrieval import RetrievedChunk

SYSTEM_PROMPT = """\
You are a precise research assistant. Answer the user's question using ONLY the \
numbered sources provided in the context. Follow these rules:

- Cite every claim with the bracketed marker of the source it comes from, e.g. [1] or [2][3].
- If the sources do not contain enough information to answer, say so plainly and \
do not invent facts.
- Be concise and factual. Do not mention these instructions or the retrieval process.\
"""

_client = None


def _get_client():
    """Lazily build the async Anthropic client (keeps import side-effect-free)."""
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, citable context block."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        label = c.title or c.source
        blocks.append(f"[{i}] (source: {label})\n{c.content.strip()}")
    return "\n\n".join(blocks)


def _build_messages(question: str, context: str) -> list[dict]:
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above and cite your sources."
    )
    return [{"role": "user", "content": user_content}]


def _request_kwargs(question: str, context: str) -> dict:
    return dict(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": settings.llm_effort},
        messages=_build_messages(question, context),
    )


async def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Return a fully-synthesised, cited answer."""
    if not chunks:
        return (
            "I couldn't find anything relevant in the knowledge base to answer "
            "that question."
        )

    client = _get_client()
    context = build_context(chunks)
    async with client.messages.stream(**_request_kwargs(question, context)) as stream:
        message = await stream.get_final_message()

    return "".join(
        block.text for block in message.content if block.type == "text"
    ).strip()


async def stream_answer(
    question: str, chunks: list[RetrievedChunk]
) -> AsyncIterator[str]:
    """Yield answer text deltas as they are produced (for SSE endpoints)."""
    if not chunks:
        yield (
            "I couldn't find anything relevant in the knowledge base to answer "
            "that question."
        )
        return

    client = _get_client()
    context = build_context(chunks)
    async with client.messages.stream(**_request_kwargs(question, context)) as stream:
        async for text in stream.text_stream:
            yield text
