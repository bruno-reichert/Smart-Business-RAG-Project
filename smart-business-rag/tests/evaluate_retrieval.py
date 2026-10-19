from typing import List, Dict, Any
from app.services.vector_store import similarity_search_with_metadata

# --- THE GOLDEN DATASET (Ground Truth) ---
# 5 representative business questions with expected page/content targets
GOLDEN_DATASET: List[Dict[str, Any]] = [
    {
        "id": 1,
        "query": "What is Nubank's customer base size and activity rate in Brazil?",
        "expected_quarter": "3Q25",
        "expected_page": 1,
        "expected_keywords": ["110 million", "85%"]
    },
    {
        "id": 2,
        "query": "What are the customer growth milestones in Mexico and Colombia?",
        "expected_quarter": "3Q25",
        "expected_page": 1,
        "expected_keywords": ["13 million", "4 million"]
    },
    {
        "id": 3,
        "query": "What was Nubank's total credit portfolio size and growth rate?",
        "expected_quarter": "3Q25",
        "expected_page": 1,
        "expected_keywords": ["30.4 billion", "42%"]
    },
    {
        "id": 4,
        "query": "What were the business highlights and new product launches across markets?",
        "expected_quarter": "4Q25",
        "expected_page": 3,
        "expected_keywords": ["100 new products", "payroll loan"]
    },
    {
        "id": 5,
        "query": "How is Nubank positioned relative to private financial institutions in Brazil?",
        "expected_quarter": "4Q25",
        "expected_page": 1,
        "expected_keywords": ["largest private", "Central Bank"]
    }
]


def run_retrieval_benchmark(k: int = 4):
    print("=" * 65)
    print("🎯 RUNNING RETRIEVAL EVALUATION BENCHMARK (Golden Dataset)")
    print("=" * 65)
    
    hits = 0
    keyword_matches = 0
    total = len(GOLDEN_DATASET)

    for item in GOLDEN_DATASET:
        q_id = item["id"]
        query = item["query"]
        expected_page = item["expected_page"]
        expected_quarter = item["expected_quarter"]
        keywords = item["expected_keywords"]

        print(f"\n[Q{q_id}] Query: \"{query}\"")
        
        # Perform retrieval
        retrieved_docs = similarity_search_with_metadata(query=query, k=k)
        
        # Check if expected document page is in top-k
        hit_found = False
        keyword_found = False

        for doc in retrieved_docs:
            meta = doc.metadata
            content = doc.page_content.lower()

            # Check page provenance hit
            if meta.get("page") == expected_page and meta.get("fiscal_period") == expected_quarter:
                hit_found = True

            # Check keyword hit
            for kw in keywords:
                if kw.lower() in content:
                    keyword_found = True
                    break

        if hit_found:
            hits += 1
            print(f"  ✅ Page Provenance HIT: Found {expected_quarter} Page {expected_page}")
        else:
            print(f"  ❌ Page Provenance MISS: Expected {expected_quarter} Page {expected_page}")

        if keyword_found:
            keyword_matches += 1
            print(f"  ✅ Keyword HIT: Matched target business metrics")
        else:
            print(f"  ❌ Keyword MISS: Missing expected terms {keywords}")

    # Calculate metrics
    hit_rate = (hits / total) * 100
    keyword_recall = (keyword_matches / total) * 100

    print("\n" + "=" * 65)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 65)
    print(f"Total Test Cases:            {total}")
    print(f"Page-Level Hit Rate (Top-{k}): {hit_rate:.1f}% ({hits}/{total})")
    print(f"Keyword Recall (Top-{k}):     {keyword_recall:.1f}% ({keyword_matches}/{total})")
    print("=" * 65)

    assert hit_rate >= 80.0, f"Hit rate dropped below 80%! Got {hit_rate}%"
    print("🎉 BENCHMARK PASSED: Retrieval pipeline meets enterprise standards!")


if __name__ == "__main__":
    run_retrieval_benchmark()