from src.phase2_semantic_search import SemanticSearchEngine

se = SemanticSearchEngine()
print("Collection stats:", se.get_collection_stats())
print("Similarity threshold from config:", se.similarity_threshold)

queries = [
    "Is cataract surgery covered?",
    "What is the grace period for premium payment?",
    "What are the waiting periods for pre-existing diseases?"
]

for query in queries:
    emb = se.encode_query(query)
    raw = se.collection.query(
        query_embeddings=[emb],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    print(f"\nRaw results for '{query}':")
    for i, doc in enumerate(raw['documents'][0]):
        dist = raw['distances'][0][i]
        sim = 1 - dist
        print(i, "sim", sim, "text", doc[:100])
    # threshold filtering
    filtered = [
        {
            'text': raw['documents'][0][i],
            'similarity_score': 1 - raw['distances'][0][i]
        }
        for i in range(len(raw['documents'][0]))
        if (1 - raw['distances'][0][i]) >= se.similarity_threshold
    ]
    print("Filtered results count", len(filtered))
    for r in filtered:
        print("  ", r['similarity_score'], r['text'][:100])
