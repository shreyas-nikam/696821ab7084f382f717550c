"""Test script to verify CSV-based FOMC document ingestion."""

import asyncio
import sys
from source import main_ingestion


async def test_ingestion():
    """Test the CSV-based ingestion."""
    print("Starting CSV-based ingestion test...")
    try:
        documents = await main_ingestion()
        print(f"\n✓ Successfully ingested {len(documents)} documents")

        # Print summary of each document
        for doc in documents[:5]:  # Show first 5
            print(f"\n--- Document ---")
            print(f"ID: {doc.document_id}")
            print(f"Type: {doc.document_type.value}")
            print(f"Meeting Date: {doc.meeting_date}")
            print(f"Title: {doc.title}")
            print(f"URL: {doc.source_url}")
            print(f"Word Count: {doc.metadata.word_count}")
            print(f"Sections: {len(doc.sections)}")
            if doc.raw_text:
                print(f"Preview: {doc.raw_text[:200]}...")

        if len(documents) > 5:
            print(f"\n... and {len(documents) - 5} more documents")

    except Exception as e:
        print(f"\n✗ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_ingestion())
