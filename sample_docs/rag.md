# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation is a technique that grounds a large language
model's answers in an external corpus of documents rather than relying solely
on the model's parametric memory.

## Pipeline

1. **Ingestion**, documents are split into overlapping chunks.
2. **Embedding**, each chunk is converted into a dense vector by an embedding
   model.
3. **Indexing**, vectors are stored in a vector database such as pgvector.
4. **Retrieval**, at query time, the question is embedded and the most similar
   chunks are fetched by nearest-neighbour search.
5. **Generation**, the retrieved chunks are passed to the LLM as context, and
   the model produces an answer that cites its sources.

## Why it helps

RAG reduces hallucination because the model answers from supplied evidence, it
keeps knowledge current without retraining, and it makes answers auditable: each
claim can be traced back to a specific source chunk.
