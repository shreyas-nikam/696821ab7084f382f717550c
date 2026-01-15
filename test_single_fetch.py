"""Quick test of URL fetching."""

import asyncio
from source import fetch_fomc_statement_from_url
from datetime import date


async def test_fetch():
    """Test fetching a single document."""
    url = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240131a.htm"
    meeting_date = date(2024, 1, 31)

    print(f"Testing fetch from: {url}")
    doc = await fetch_fomc_statement_from_url(url, meeting_date)

    if doc:
        print(f"\n✓ Successfully fetched document")
        print(f"ID: {doc.document_id}")
        print(f"Type: {doc.document_type.value}")
        print(f"Word Count: {doc.metadata.word_count}")
        print(f"\nFirst 500 characters:")
        print(doc.raw_text[:500])
    else:
        print("✗ Failed to fetch document")

if __name__ == "__main__":
    asyncio.run(test_fetch())
