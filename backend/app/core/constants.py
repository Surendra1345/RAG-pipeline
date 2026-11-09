"""Fixed values shared across the app."""

# Size of the vectors made by all-MiniLM-L6-v2. The database column must match.
EMBEDDING_DIM = 384

# Reciprocal Rank Fusion: a chunk's score is the sum of 1 / (RRF_C + rank) over each list.
# 60 is the common default; a larger value makes the fusion less top-heavy.
RRF_C = 60

# How many candidates each retriever returns, and how many the reranker rescores.
DEFAULT_CANDIDATES = 20

# How many chunks the final answer uses by default.
DEFAULT_TOP_K = 5
