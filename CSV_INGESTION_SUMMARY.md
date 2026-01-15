# CSV-Based FOMC Document Ingestion - Implementation Summary

## Overview
The `main_ingestion` function has been updated to use the `fomc_data.csv` file to dynamically fetch FOMC statements and minutes from the Federal Reserve website.

## Changes Made

### 1. New Functions Added to `source.py`

#### `fetch_fomc_statement_from_url(url, meeting_date)`
- Fetches and parses FOMC statements from any given URL
- Uses `httpx` for async HTTP requests
- Uses `BeautifulSoup4` for HTML parsing
- Extracts main content from Fed's HTML structure
- Returns a structured `FOMCDocument` object

#### `fetch_fomc_minutes_from_url(url, meeting_date)`
- Fetches and parses FOMC minutes from any given URL
- Similar architecture to statement fetcher
- Attempts to extract section headers for better structure
- Falls back to single section if no headers found
- Returns a structured `FOMCDocument` object

### 2. Updated `main_ingestion()` Function

#### New Parameters:
- `limit: Optional[int]` - Maximum number of documents to fetch (None for all)
- `recent_only: bool` - If True, only fetch documents from 2024 onwards

#### Features:
- Reads from `fomc_data.csv` in the workspace
- Parses various date formats (e.g., "Jan 28-29", "Jan/Feb 31-1", "Mar 15")
- Handles both statements and minutes URLs
- Skips entries with "N/A" URLs
- Provides detailed logging with progress indicators
- Falls back to mock data if CSV reading fails
- Proper error handling for individual document fetch failures

### 3. CSV File Structure
The `fomc_data.csv` file contains:
- **Year**: Meeting year
- **Date**: Meeting date (various formats supported)
- **Statement HTML Link**: URL to the FOMC statement
- **Minutes HTML Link**: URL to the FOMC minutes (or "N/A" if not available)

### 4. Dependencies Added
- `python-dateutil` - For flexible date parsing

## Usage Examples

### Fetch all recent documents (2024+)
```python
documents = await main_ingestion(recent_only=True)
```

### Fetch limited number of documents for testing
```python
documents = await main_ingestion(limit=6)
```

### Fetch specific recent documents with limit
```python
documents = await main_ingestion(limit=10, recent_only=True)
```

### Fetch all available documents (default)
```python
documents = await main_ingestion()
```

## Testing

Run the test scripts to verify functionality:

```bash
# Test single document fetch
python test_single_fetch.py

# Test limited CSV ingestion
python test_limited_csv.py

# Test main_ingestion with parameters
python test_main_ingestion.py
```

## Results

The implementation successfully:
- ✓ Fetches real FOMC documents from Federal Reserve URLs
- ✓ Parses HTML content into structured data
- ✓ Handles 50+ FOMC meetings from 2020-2025
- ✓ Extracts text content with proper formatting
- ✓ Creates document metadata (word counts, hashes, timestamps)
- ✓ Supports flexible date formats
- ✓ Provides progress logging
- ✓ Handles errors gracefully

## Sample Output

```
Processing 17 FOMC meetings from CSV
Processing meeting 1/17: 2024 Jan 30-31
✓ Fetched statement for 2024-01-30 (365 words)
✓ Fetched minutes for 2024-01-30 (7553 words)
...
Successfully ingested 34 documents from CSV
```

## Integration with Streamlit App

The `app.py` file calls `main_ingestion()` on startup:
```python
st.session_state.current_fomc_documents = asyncio.run(main_ingestion())
```

For better performance in production, consider:
- Adding `recent_only=True` to fetch only recent documents
- Using `limit=20` to cap the number of documents for faster loading
- Caching the results to avoid repeated fetches

## Future Enhancements

Potential improvements:
1. Add caching mechanism to store fetched documents locally
2. Implement incremental updates (only fetch new documents)
3. Add support for press conference transcripts
4. Parallel fetching with `asyncio.gather()` for faster ingestion
5. Add retry logic with exponential backoff for failed requests
6. Store documents in a database for persistence
