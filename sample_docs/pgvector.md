# pgvector

pgvector is an open-source extension for PostgreSQL that adds support for
storing and querying vector embeddings directly in the database. It introduces
a `vector` column type and a set of distance operators.

## Distance operators

- `<->`, Euclidean (L2) distance
- `<#>`, negative inner product
- `<=>`, cosine distance

For semantic search with normalised embeddings, cosine distance (`<=>`) is the
most common choice. Cosine similarity equals one minus cosine distance.

## Indexes

pgvector supports two approximate-nearest-neighbour indexes: IVFFlat and HNSW.
HNSW (Hierarchical Navigable Small World) builds a multi-layer graph and
generally offers better query performance and recall than IVFFlat, at the cost
of slower index build time and higher memory use. Create an HNSW index for
cosine distance with `USING hnsw (embedding vector_cosine_ops)`.
