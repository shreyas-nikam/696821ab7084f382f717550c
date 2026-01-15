"""Test script to verify CSV-based FOMC document ingestion (limited test)."""

import asyncio
import csv
import os
from datetime import date
from dateutil import parser as date_parser
from source import fetch_fomc_statement_from_url, fetch_fomc_minutes_from_url


async def test_limited_ingestion():
    """Test the CSV-based ingestion with only a few documents."""
    print("Starting limited CSV-based ingestion test...")

    csv_path = "fomc_data.csv"
    documents = []
    max_docs = 4  # Test with just 2 meetings (statement + minutes each)

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Test with the most recent meeting (2024 Jan)
            test_rows = [r for r in rows if r['Year']
                         == '2024'][:1]  # Just Jan 2024

            for row in test_rows:
                year = row['Year']
                date_str = row['Date']
                statement_url = row['Statement HTML Link']
                minutes_url = row['Minutes HTML Link']

                print(f"\nProcessing: {year} {date_str}")

                # Parse the date
                try:
                    if '/' in date_str:
                        parts = date_str.split()
                        if len(parts) >= 2:
                            month_part = parts[0].split('/')[-1]
                            day_part = parts[1].split('-')[0]
                            meeting_date = date_parser.parse(
                                f"{month_part} {day_part}, {year}").date()
                        else:
                            meeting_date = date_parser.parse(
                                f"{date_str}, {year}").date()
                    else:
                        parts = date_str.split()
                        if len(parts) >= 2 and '-' in parts[1]:
                            day = parts[1].split('-')[0]
                            meeting_date = date_parser.parse(
                                f"{parts[0]} {day}, {year}").date()
                        else:
                            meeting_date = date_parser.parse(
                                f"{date_str}, {year}").date()
                    print(f"  Parsed date: {meeting_date}")
                except Exception as e:
                    print(f"  ✗ Failed to parse date: {e}")
                    continue

                # Fetch statement
                if statement_url and statement_url != 'N/A':
                    print(
                        f"  Fetching statement from: {statement_url[:60]}...")
                    try:
                        statement_doc = await fetch_fomc_statement_from_url(statement_url, meeting_date)
                        if statement_doc:
                            documents.append(statement_doc)
                            print(
                                f"  ✓ Statement: {statement_doc.metadata.word_count} words")
                    except Exception as e:
                        print(f"  ✗ Failed to fetch statement: {e}")

                # Fetch minutes
                if minutes_url and minutes_url != 'N/A' and len(documents) < max_docs:
                    print(f"  Fetching minutes from: {minutes_url[:60]}...")
                    try:
                        minutes_doc = await fetch_fomc_minutes_from_url(minutes_url, meeting_date)
                        if minutes_doc:
                            documents.append(minutes_doc)
                            print(
                                f"  ✓ Minutes: {minutes_doc.metadata.word_count} words")
                    except Exception as e:
                        print(f"  ✗ Failed to fetch minutes: {e}")

                if len(documents) >= max_docs:
                    break

        print(f"\n{'='*60}")
        print(f"✓ Successfully ingested {len(documents)} documents")
        print(f"{'='*60}")

        # Print summary
        for i, doc in enumerate(documents, 1):
            print(f"\n{i}. {doc.title}")
            print(f"   Type: {doc.document_type.value}")
            print(f"   Meeting: {doc.meeting_date}")
            print(f"   Words: {doc.metadata.word_count:,}")
            print(f"   Sections: {len(doc.sections)}")
            print(f"   Preview: {doc.raw_text[:150]}...")

    except Exception as e:
        print(f"\n✗ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_limited_ingestion())
