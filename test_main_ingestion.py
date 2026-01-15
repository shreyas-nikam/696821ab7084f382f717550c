"""Test the updated main_ingestion function with parameters."""

import asyncio
from source import main_ingestion


async def test_main_ingestion():
    """Test the main ingestion function with various parameters."""

    print("=" * 70)
    print("TEST 1: Fetch recent documents only (2024+), limit to 6")
    print("=" * 70)
    docs = await main_ingestion(limit=6, recent_only=True)
    print(f"\n✓ Fetched {len(docs)} documents")

    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. [{doc.document_type.value}] {doc.title}")
        print(
            f"   Meeting: {doc.meeting_date}, Words: {doc.metadata.word_count:,}")
        print(f"   URL: {doc.source_url[:70]}...")

    print("\n" + "=" * 70)
    print("TEST 2: Fetch limited documents from all years, limit to 4")
    print("=" * 70)
    docs2 = await main_ingestion(limit=4, recent_only=False)
    print(f"\n✓ Fetched {len(docs2)} documents")

    years = set([doc.meeting_date.year for doc in docs2])
    print(f"Years covered: {sorted(years)}")

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total unique documents fetched: {len(docs) + len(docs2)}")
    print(
        f"Statement documents: {sum(1 for d in docs + docs2 if d.document_type.value == 'statement')}")
    print(
        f"Minutes documents: {sum(1 for d in docs + docs2 if d.document_type.value == 'minutes')}")
    print(f"Total words: {sum(d.metadata.word_count for d in docs + docs2):,}")

if __name__ == "__main__":
    asyncio.run(test_main_ingestion())
