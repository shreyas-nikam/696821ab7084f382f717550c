# Quick Start Guide: CSV-Based FOMC Document Ingestion

## What's New?

The `main_ingestion()` function now automatically fetches real FOMC documents from the Federal Reserve website using URLs stored in `fomc_data.csv`.

## Quick Usage

### In Your Code

```python
from source import main_ingestion
import asyncio

# Fetch all recent documents (2024+)
documents = asyncio.run(main_ingestion(recent_only=True))

# Or fetch with a limit
documents = asyncio.run(main_ingestion(limit=10, recent_only=True))

# Or fetch all available documents
documents = asyncio.run(main_ingestion())
```

### Parameters

- **`limit`** (int, optional): Maximum number of documents to fetch. Default: None (fetch all)
- **`recent_only`** (bool, optional): If True, only fetch documents from 2024 onwards. Default: False

## What Gets Fetched?

From `fomc_data.csv` (51 FOMC meetings from 2020-2025):
- ✅ **Statements**: 51 available
- ✅ **Minutes**: 48 available (3 meetings have no minutes)
- ❌ **Press Conferences**: Not yet implemented

## Document Structure

Each fetched document is a `FOMCDocument` object with:

```python
{
    "document_id": "statement_2024-01-31",
    "document_type": "statement",  # or "minutes"
    "meeting_date": datetime.date(2024, 1, 31),
    "publication_date": datetime.date(2024, 1, 31),
    "title": "FOMC Statement - January 31, 2024",
    "source_url": "https://www.federalreserve.gov/...",
    "raw_text": "Recent indicators suggest...",
    "sections": [...],  # List of DocumentSection objects
    "metadata": {
        "ingestion_timestamp": datetime,
        "source_hash": "abc123...",
        "parser_version": "1.0.0",
        "word_count": 365
    }
}
```

## Running Tests

```bash
# Quick single document test
python test_single_fetch.py

# Limited CSV ingestion test
python test_limited_csv.py

# Full main_ingestion test with parameters
python test_main_ingestion.py

# Complete integration test
python test_integration.py
```

## CSV File Format

`fomc_data.csv` has 4 columns:

| Year | Date | Statement HTML Link | Minutes HTML Link |
|------|------|-------------------|------------------|
| 2024 | Jan 30-31 | https://... | https://... |
| 2024 | Mar 19-20 | https://... | https://... |
| ... | ... | ... | ... |

- Dates can be in various formats: "Jan 28-29", "Jan/Feb 31-1", "Mar 15"
- URLs marked as "N/A" are skipped

## Performance Tips

For production use:

```python
# Recommended: Fetch only recent documents for faster loading
documents = asyncio.run(main_ingestion(recent_only=True))

# For testing: Limit the number of documents
documents = asyncio.run(main_ingestion(limit=6, recent_only=True))

# For analysis: Fetch all available documents (may take 1-2 minutes)
documents = asyncio.run(main_ingestion())
```

## Troubleshooting

### "Failed to read CSV"
- Check that `fomc_data.csv` exists in the workspace
- Verify CSV file encoding is UTF-8

### "Failed to fetch statement/minutes"
- Check internet connection
- Verify Federal Reserve website is accessible
- Some documents may genuinely not be available (check URL)

### "Failed to parse date"
- Check CSV date format
- Function supports: "Jan 28-29", "Jan/Feb 31-1", "Mar 15", etc.

## Integration with Streamlit App

The app automatically loads documents on startup:

```python
# In app.py
st.session_state.current_fomc_documents = asyncio.run(
    main_ingestion(recent_only=True)
)
```

## Examples

### Example 1: Fetch 2024 Documents Only

```python
import asyncio
from source import main_ingestion

async def fetch_2024_documents():
    docs = await main_ingestion(recent_only=True)
    docs_2024 = [d for d in docs if d.meeting_date.year == 2024]
    
    print(f"Fetched {len(docs_2024)} documents from 2024")
    for doc in docs_2024:
        print(f"  - {doc.title} ({doc.metadata.word_count:,} words)")
    
    return docs_2024

# Run
documents = asyncio.run(fetch_2024_documents())
```

### Example 2: Get Word Count Statistics

```python
import asyncio
from source import main_ingestion

async def analyze_documents():
    docs = await main_ingestion(limit=20, recent_only=True)
    
    total_words = sum(d.metadata.word_count for d in docs)
    avg_words = total_words / len(docs)
    
    print(f"Total documents: {len(docs)}")
    print(f"Total words: {total_words:,}")
    print(f"Average words per document: {avg_words:,.0f}")
    
    # By type
    statements = [d for d in docs if d.document_type.value == 'statement']
    minutes = [d for d in docs if d.document_type.value == 'minutes']
    
    print(f"\nStatements: {len(statements)} documents")
    print(f"  Avg words: {sum(d.metadata.word_count for d in statements)/len(statements):,.0f}")
    print(f"Minutes: {len(minutes)} documents")
    print(f"  Avg words: {sum(d.metadata.word_count for d in minutes)/len(minutes):,.0f}")

# Run
asyncio.run(analyze_documents())
```

## Need Help?

- Check [CSV_INGESTION_SUMMARY.md](CSV_INGESTION_SUMMARY.md) for detailed implementation docs
- Run `python test_integration.py` to verify your setup
- Review error logs for specific issues

## Next Steps

After fetching documents, you can:
1. Build vector store with documents for semantic search
2. Analyze hawkish/dovish tone
3. Extract key themes
4. Compare across different meetings
5. Generate research memos

See the main notebook/app for complete workflows!
