#!/usr/bin/env python
"""
Integration test for the CSV-based FOMC document ingestion system.
Tests the complete workflow from CSV reading to document fetching.
"""

import asyncio
import sys
from datetime import datetime


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


async def run_integration_test():
    """Run comprehensive integration test."""

    try:
        from source import main_ingestion
    except ImportError as e:
        print(f"❌ Failed to import main_ingestion: {e}")
        return False

    print_header("FOMC Document Ingestion - Integration Test")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test 1: Fetch recent documents with limit
    print_header("Test 1: Recent Documents (2024+) with Limit")
    try:
        docs_recent = await main_ingestion(limit=4, recent_only=True)
        print(f"✓ Successfully fetched {len(docs_recent)} recent documents")

        if len(docs_recent) > 0:
            print(f"\nSample document:")
            doc = docs_recent[0]
            print(f"  - Title: {doc.title}")
            print(f"  - Type: {doc.document_type.value}")
            print(f"  - Date: {doc.meeting_date}")
            print(f"  - Words: {doc.metadata.word_count:,}")
            print(f"  - Sections: {len(doc.sections)}")
            print(f"  - Preview: {doc.raw_text[:100]}...")
        else:
            print("⚠️  No documents were fetched")
            return False

    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Verify document structure
    print_header("Test 2: Document Structure Validation")
    try:
        errors = []
        for doc in docs_recent:
            # Check required fields
            if not doc.document_id:
                errors.append(f"{doc.title}: Missing document_id")
            if not doc.source_url:
                errors.append(f"{doc.title}: Missing source_url")
            if not doc.raw_text:
                errors.append(f"{doc.title}: Missing raw_text")
            if len(doc.sections) == 0:
                errors.append(f"{doc.title}: No sections")
            if doc.metadata.word_count == 0:
                errors.append(f"{doc.title}: Zero word count")

        if errors:
            print("❌ Structure validation errors found:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("✓ All documents have valid structure")

    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False

    # Test 3: Document statistics
    print_header("Test 3: Document Statistics")
    try:
        total_words = sum(d.metadata.word_count for d in docs_recent)
        total_sections = sum(len(d.sections) for d in docs_recent)
        doc_types = {}
        years = set()

        for doc in docs_recent:
            doc_type = doc.document_type.value
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            years.add(doc.meeting_date.year)

        print(f"Total Documents: {len(docs_recent)}")
        print(f"Total Words: {total_words:,}")
        print(f"Total Sections: {total_sections}")
        print(f"Document Types:")
        for doc_type, count in sorted(doc_types.items()):
            print(f"  - {doc_type}: {count}")
        print(f"Years: {sorted(years)}")
        print("✓ Statistics generated successfully")

    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False

    # Test 4: CSV file accessibility
    print_header("Test 4: CSV File Accessibility")
    try:
        import os
        import csv
        csv_path = "fomc_data.csv"

        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return False

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"✓ CSV file accessible")
        print(f"  - Total rows: {len(rows)}")
        print(f"  - Columns: {', '.join(rows[0].keys())}")

        # Check for valid URLs
        valid_statements = sum(
            1 for r in rows if r['Statement HTML Link'] != 'N/A')
        valid_minutes = sum(1 for r in rows if r['Minutes HTML Link'] != 'N/A')
        print(f"  - Valid statement URLs: {valid_statements}")
        print(f"  - Valid minutes URLs: {valid_minutes}")

    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return False

    # Final summary
    print_header("Integration Test Summary")
    print("✅ All tests passed successfully!")
    print(f"System is ready to ingest FOMC documents from CSV")
    print(f"Total documents available in CSV: {len(rows)}")
    print(f"Tested with {len(docs_recent)} recent documents")

    return True

if __name__ == "__main__":
    print("\nStarting FOMC Document Ingestion Integration Test...\n")

    try:
        result = asyncio.run(run_integration_test())

        if result:
            print("\n" + "=" * 70)
            print("  🎉 ALL TESTS PASSED - System Ready for Production")
            print("=" * 70 + "\n")
            sys.exit(0)
        else:
            print("\n" + "=" * 70)
            print("  ❌ SOME TESTS FAILED - Please review errors above")
            print("=" * 70 + "\n")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
