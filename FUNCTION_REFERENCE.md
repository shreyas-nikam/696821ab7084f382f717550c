# Quick Reference: Implemented Functions

## Session Management

### SessionStateManager
```python
manager = SessionStateManager()
request_id = manager.start_request(meeting_date)  # Start new analysis
state = manager.get_state(request_id)  # Get workflow state
manager.update_step(request_id, "step_name", 0.5, log_entry)  # Update progress
manager.record_human_decision(request_id, "approved", "comments", "reviewer_id")
```

## Document Functions

### Meeting Dates & Availability
```python
from datetime import date

# Get FOMC meeting dates
dates = get_fomc_meeting_dates(
    start=date(2023, 1, 1), 
    end=date(2024, 12, 31)
)

# Check document availability for a meeting
availability = check_document_availability(date(2024, 1, 31))
# Returns: {"statement": True, "minutes": True, "press_conference": True, "sep": False}
```

### Load Documents
```python
documents = load_source_documents(request_id)
```

## Memo Functions

### Load & Update Memos
```python
from source import load_memo, update_memo_status, ReviewAction
from datetime import datetime

# Load a memo
memo = load_memo(memo_id)

# Update memo status
update_memo_status(
    memo_id=memo.memo_id,
    status="approved",
    review_action=ReviewAction(
        timestamp=datetime.now(),
        reviewer_id="Alex Chen",
        action="approved",
        comments="Looks good!",
        version_reviewed=memo.version
    )
)
```

## Review Functions

### Pending Reviews
```python
# Get count of pending reviews
count = get_pending_review_count()

# Get list of pending reviews
pending = get_pending_reviews()
# Returns list of dicts with: request_id, meeting_date, status, memo_id
```

## Workflow

### Create & Run Workflow
```python
import asyncio

# Create workflow
workflow = create_fomc_workflow()

# Run workflow (async)
async def run():
    initial_state = AgentState(
        meeting_date=date(2024, 1, 31),
        request_id="req_123",
        requested_at=datetime.now(),
        config={"num_themes_to_extract": 5}
    )
    
    async for event in workflow.astream_events(
        initial_state, 
        {"configurable": {"thread_id": "req_123"}}, 
        version="v1"
    ):
        print(f"Event: {event['event']} - {event['name']}")

asyncio.run(run())
```

## Visualization

### Tone Trajectory Chart
```python
import streamlit as st

# Historical scores, current tone, and meeting dates required
render_tone_trajectory(
    historical_scores=[score1, score2, score3],  # List[ToneScore]
    current_tone=current_tone_score,  # ToneScore
    meeting_dates=[date1, date2, date3]  # List[date]
)
```

### Citation Network Graph
```python
# Visualize theme-citation relationships
render_citation_network(
    citations=memo.all_citations,  # List[Citation]
    themes=memo.themes  # List[ThemeExtraction]
)
```

## Mock Data

### Historical Tone Scores
```python
from source import mock_prior_tone_scores

# Access mock historical data
for score in mock_prior_tone_scores:
    print(f"Score: {score.score}, Confidence: {score.confidence}")
```

## AgentState Structure

```python
state = AgentState(
    meeting_date=date(2024, 1, 31),
    request_id="req_123",
    requested_at=datetime.now(),
    config={
        "include_historical_context": True,
        "num_themes_to_extract": 5,
        "auto_validate_citations": True,
        "minimum_confidence_threshold": 0.7
    },
    documents=None,  # Optional[List[FOMCDocument]]
    themes=None,  # Optional[List[ThemeExtraction]]
    tone_analysis=None,  # Optional[ToneScore]
    tone_delta=None,  # Optional[ToneDelta]
    surprises=None,  # Optional[List[Surprise]]
    draft_memo=None,  # Optional[FOMCMemo]
    validation_report=None,  # Optional[ValidationReport]
    current_step=None,  # Optional[str]
    error=None  # Optional[str]
)
```

## Complete Workflow Example

```python
import streamlit as st
from datetime import date, datetime
from source import *

# Initialize session manager
manager = SessionStateManager()

# Start new request
meeting_date = date(2024, 1, 31)
request_id = manager.start_request(meeting_date)

# Create and run workflow
workflow = create_fomc_workflow()
initial_state = AgentState(
    meeting_date=meeting_date,
    request_id=request_id,
    requested_at=datetime.now(),
    config={"num_themes_to_extract": 5}
)

# Execute workflow (in async context)
async def execute():
    async for event in workflow.astream_events(
        initial_state,
        {"configurable": {"thread_id": request_id}},
        version="v1"
    ):
        # Process events
        if event["event"] == "on_chain_end":
            print(f"Completed: {event['name']}")
    
    # Load and display results
    memo = load_memo(f"memo_{request_id}")
    
    # Visualize results
    render_tone_trajectory([mock_prior_tone_scores[0]], memo.tone_analysis, [meeting_date])
    render_citation_network(memo.all_citations, memo.themes)

# Run in streamlit (handles async automatically)
# or: asyncio.run(execute())
```
