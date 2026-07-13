"""Run every golden-set question through the retriever and score the results."""
from demo_chatbot.rag.ingest import build_retriever
from eval.golden_set import GOLDEN_SET


def run() -> None:
    retriever = build_retriever()
    passed = 0
    failures = []

    print("Running golden-set evaluation...\n")

    for i, case in enumerate(GOLDEN_SET, start=1):
        question = case["question"]
        want = case["expect_keyword"]

        # 1. search the FAQ, take the single best chunk
        hits = retriever.search(question, k=1)
        top_text = hits[0].chunk.text if hits else ""

        # 2. is the expected keyword in it? (case-insensitive)
        ok = want.lower() in top_text.lower()

        # 3. count it
        if ok:
            passed += 1
        else:
            failures.append((question, want, top_text))

        print(f"[{i}] Q: {question}")
        print(f"    want keyword: '{want}'")
        print(f"    got chunk:    \"{top_text[:65]}...\"")
        print(f"    result:       {'PASS' if ok else 'FAIL'}\n")

    total = len(GOLDEN_SET)
    print("=" * 45)
    print(f"SCORE: {passed}/{total} passed ({round(passed/total*100)}%)")
    print("=" * 45)

    if failures:
        print("\nFailed cases (these tell you what to improve):")
        for q, want, got in failures:
            print(f" - \"{q}\"  wanted '{want}', got: \"{got[:45]}...\"")


if __name__ == "__main__":
    run()