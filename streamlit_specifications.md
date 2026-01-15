
# Streamlit Application Specification: FOMC Insights Dashboard

## 1. Application Overview

**Purpose**:
The "FOMC Insights Dashboard" is a Streamlit-based application designed for Quantitative Analysts (persona: Alex Chen at Global Macro Alpha Fund - GMAF). Its primary purpose is to transform raw Federal Open Market Committee (FOMC) communications (Statements, Minutes, Press Conference transcripts) into structured, auditable investment research. The application interactively quantifies the Fed's hawkish/dovish stance, identifies significant policy shifts and surprises, and extracts key themes, all while maintaining a clear audit trail and facilitating human-in-the-loop approval gates. This enables GMAF to gain a competitive edge in macro trading decisions by providing objective and verifiable insights.

**High-level Story Flow**:

1.  **Dashboard Entry**: Alex lands on the home page, greeted by key system status metrics and navigation cards.
2.  **Requesting a New Analysis**: Alex navigates to the "New Analysis" page. He selects a specific FOMC meeting date from a dropdown, observes the availability of documents (Statement, Minutes, Press Conference), and customizes analysis parameters such as whether to include historical context, the number of themes to extract, automatic citation validation, and a minimum confidence threshold for outputs.
3.  **Executing the Analysis Workflow**: Upon clicking "Start Analysis", an AI agent workflow (orchestrated by LangGraph) is initiated in the backend. The application displays a real-time progress bar and status updates as the agent performs sequential steps:
    *   **Ingestion**: Fetches and parses FOMC documents.
    *   **Memory/Context Building**: Indexes the documents into a vector store and retrieves relevant historical context.
    *   **Analysis**: Extracts key themes, computes the hawkish/dovish tone score (including components and delta from prior meetings), and detects policy surprises.
    *   **Memo Generation**: Synthesizes all findings into a draft `FOMCMemo`.
    *   **Validation**: Runs automated checks on the memo for citation validity, hallucination detection, tone score bounds, and prohibited recommendations.
4.  **Human Review Gate**: Once the automated analysis and validation are complete, the draft memo enters a "pending review" state. Alex navigates to the "Pending Reviews" page. He selects a memo for review, presented with a comprehensive view including an executive summary, detailed themes with citations, tone analysis (overall score, components, delta, trajectory chart), detected surprises, and the full validation report.
5.  **Decision and Audit**: Alex critically evaluates the AI's output. He can approve the memo, request revisions, or reject it, providing mandatory comments for each action. This decision, along with his comments, is meticulously recorded in the audit trail, completing the human-in-the-loop governance.
6.  **Analysis History**: Approved memos are archived and accessible on the "Analysis History" page, allowing Alex to browse past research, visualize historical tone trends, and review the evolution of themes over time, providing valuable longitudinal insights for macro strategy development.

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import asyncio
from datetime import date, datetime, timedelta
from typing import List, Optional, Literal, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json # for displaying JSON output for audit log

# Assume all Python code for Pydantic models, utility functions, mock ChromaDB,
# agentic core functions, ValidationPipeline, AuditTrail, SessionStateManager,
# LangGraph workflow (AgentState, create_fomc_workflow, node functions, routers),
# and Streamlit utility/visualization components (e.g., get_fomc_meeting_dates,
# check_document_availability, get_pending_review_count, get_pending_reviews,
# load_memo, load_source_documents, update_memo_status, render_tone_trajectory,
# render_citation_network) are provided and importable directly from source.py.
from source import * 
```

### `st.session_state` Usage

`st.session_state` is leveraged to manage application state across user interactions and page navigations, simulating a stateful workflow in a stateless environment.

*   **Initialization (on first run of `app.py`)**:

    ```python
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home" # Tracks active page for conditional rendering: "home", "new_analysis", "pending_reviews", "analysis_history"
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = {} # Dictionary to store LangGraph AgentState objects (or their dict representation) for each ongoing/completed request, keyed by request_id.
    if "request_history" not in st.session_state:
        st.session_state.request_history = [] # List of dicts, summarizing all analysis requests: {"request_id": str, "meeting_date": date, "status": Literal["initialized", "running", "pending_review", "approved", "rejected", "revision_requested"], "memo_id": Optional[str], "generated_at": datetime}
    if "current_request_id" not in st.session_state:
        st.session_state.current_request_id = None # Stores the request_id of the analysis currently being run or viewed.
    if "selected_review_memo_id" not in st.session_state:
        st.session_state.selected_review_memo_id = None # Stores the memo_id of the FOMCMemo currently selected for human review.
    if "review_action" not in st.session_state:
        st.session_state.review_action = None # Stores the user's selected action on the review page: "approve", "revise", "reject". Cleared after submission.
    if "reviewer_comments" not in st.session_state:
        st.session_state.reviewer_comments = "" # Stores comments for human review action.
    if "latest_memo_data" not in st.session_state:
        st.session_state.latest_memo_data = None # Stores the full FOMCMemo object generated by the most recent analysis run, for immediate display.
    if "active_workflow_task" not in st.session_state:
        st.session_state.active_workflow_task = None # Stores an asyncio.Task object if a workflow is running asynchronously, to prevent re-triggering.
    ```

*   **Updates**:
    *   **Navigation**: Setting `st.session_state.current_page` in `app.py` or using `st.switch_page()` from other pages.
    *   **New Analysis Page (`pages/1_📈_Analysis.py`)**:
        *   Upon "Start Analysis" button click:
            *   `request_id = session_manager.start_request(selected_meeting)`: The `SessionStateManager` (assumed from `source.py`) initializes a new entry in `st.session_state.workflow_state` and updates `st.session_state.current_request_id`.
            *   During `workflow.astream()` execution, `session_manager.update_step(request_id, current_node, event)` will update the corresponding entry in `st.session_state.workflow_state` with current progress and status.
            *   Upon workflow completion, the final `FOMCMemo` object is stored in `st.session_state.latest_memo_data`. A summary dict is also appended to `st.session_state.request_history` (or `SessionStateManager` handles this).
            *   `st.session_state.active_workflow_task` is set when the workflow starts and cleared when it completes.
    *   **Human Review Page (`pages/2_📝_Review.py`)**:
        *   Selecting an analysis from `st.selectbox` updates `st.session_state.selected_review_memo_id`.
        *   Clicking "Approve", "Request Revision", "Reject" buttons updates `st.session_state.review_action`.
        *   Typing in the comment `st.text_area` updates `st.session_state.reviewer_comments`.
        *   Upon "Submit Decision":
            *   `session_manager.record_human_decision(st.session_state.current_request_id, st.session_state.review_action, st.session_state.reviewer_comments)`: Records the decision and updates `st.session_state.workflow_state` and `st.session_state.request_history`.
            *   `update_memo_status(memo_id, new_status, review_action_object)`: (Assumed from `source.py`) updates the memo's status and review history.
            *   `del st.session_state.review_action`, `st.session_state.reviewer_comments = ""` to clear the form.

*   **Reads**:
    *   `st.session_state.current_page`: Determines which page content is rendered (`app.py`).
    *   `st.session_state.workflow_state[st.session_state.current_request_id]`: Accessed on "New Analysis" page to display live progress, status, and final memo details.
    *   `st.session_state.request_history`: Used on "Pending Reviews" to populate the list of pending memos and on "Analysis History" to list all past analyses.
    *   `st.session_state.selected_review_memo_id`: Used on "Pending Reviews" to fetch the specific `FOMCMemo` for detailed display.
    *   `st.session_state.review_action`: Conditionally displays the "Add Comments" section and sets button text on "Pending Reviews".
    *   `st.session_state.reviewer_comments`: Pre-populates the comments text area if navigating back.
    *   `st.session_state.latest_memo_data`: Used for immediate display of the generated memo after analysis completes on the "New Analysis" page.

### Application Structure and Flow

The application simulates a multi-page experience using conditional rendering based on `st.session_state.current_page` in `app.py`. For actual multi-page apps (Streamlit 1.20+), `st.switch_page()` could be used. Given the structure of `fomc_agent_app/pages/1_  _Analysis.py` etc. in the OCR, it implies a true multi-page app with `st.switch_page`. I will use `st.switch_page` for navigation.

---

#### `app.py` (Main Application)

**1. Page Configuration and Custom CSS**:

```python
st.set_page_config(
    page_title="FOMC Research Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #0a0f1a; /* Dark background */
}
.main-header {
    color: #00d4ff; /* Accent color */
    font-size: 2.5rem;
    font-weight: bold;
}
.status-pending {
    background-color: #ffa500; /* Orange */
    color: black;
    padding: 4px 12px;
    border-radius: 4px;
}
.status-approved {
    background-color: #4ade80; /* Green */
    color: black;
    padding: 4px 12px;
    border-radius: 4px;
}
.citation-card {
    background-color: #1a1f2e;
    border: 1px solid #2a3654;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.1em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
```

**2. Sidebar Navigation and System Status**:

```python
# Initialize SessionStateManager
session_manager = SessionStateManager()

with st.sidebar:
    # Assuming "logo.png" exists
    # st.image("logo.png", width=200) 
    st.markdown("### FOMC Research Agent")
    st.markdown("*Agentic AI for Fed Analysis*")
    st.divider()

    st.markdown("#### System Status")
    col1, col2 = st.columns(2)
    with col1:
        # Placeholder metrics. In a real app, these would come from persistent storage.
        st.metric("Documents", "247", "+3") 
    with col2:
        st.metric("Analyses", "52", "+1") # Count of all analyses in request_history
    
    st.divider()

    # Navigation for actual multi-page app
    st.markdown(f"**Navigation**")
    if st.button("📈 New Analysis", use_container_width=True, key="nav_new_analysis_sidebar"):
        st.switch_page("pages/1_Analysis.py")
    if st.button(f"📝 Pending Reviews ({get_pending_review_count()})", use_container_width=True, key="nav_pending_reviews_sidebar"):
        st.switch_page("pages/2_Review.py")
    if st.button("📚 Analysis History", use_container_width=True, key="nav_analysis_history_sidebar"):
        st.switch_page("pages/3_History.py")
    if st.button("⚙️ Settings", use_container_width=True, key="nav_settings_sidebar"):
        st.switch_page("pages/4_Settings.py")
```
*   **Function Calls**:
    *   `session_manager = SessionStateManager()`: Initializes the session state manager.
    *   `get_pending_review_count()`: (Assumed from `source.py`) Retrieves the number of analyses with "pending_review" status from `st.session_state.request_history` (or a persistent store).

**3. Main Content (Home Page)**:

```python
st.markdown('<h1 class="main-header">FOMC Research Agent</h1>', unsafe_allow_html=True)
st.markdown(f"Transform Federal Reserve communications into structured, auditable investment research for Quantitative Analysts.")

st.divider()

st.markdown(f"### Get Started")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"#### 📈 New Analysis")
    st.markdown(f"Request analysis of FOMC meeting materials.")
    st.markdown(f"- Statement + Minutes + Press Conference")
    st.markdown(f"- Theme extraction with citations")
    st.markdown(f"- Tone scoring and delta computation")
    if st.button("Start New Analysis", key="nav_analysis_card", use_container_width=True):
        st.switch_page("pages/1_Analysis.py")

with col2:
    st.markdown(f"#### 📝 Pending Reviews")
    pending_count = get_pending_review_count()
    st.markdown(f"Review and approve draft memos.")
    st.markdown(f"- Citation verification")
    st.markdown(f"- Confidence assessment")
    st.markdown(f"- Approve/Revise/Reject workflow")
    if st.button(f"Review ({pending_count} pending)", key="nav_review_card", use_container_width=True):
        st.switch_page("pages/2_Review.py")

with col3:
    st.markdown(f"#### 📚 Analysis History")
    st.markdown(f"Browse past analyses and trends.")
    st.markdown(f"- Historical tone trajectory")
    st.markdown(f"- Theme evolution")
    st.markdown(f"- Market reaction correlation")
    if st.button("View History", key="nav_history_card", use_container_width=True):
        st.switch_page("pages/3_History.py")
```
*   **Function Calls**:
    *   `get_pending_review_count()`: (Assumed from `source.py`) to display the count of pending reviews on the card.

---

#### `pages/1_Analysis.py` (New Analysis Page)

**1. Page Title and Meeting Selection**:

```python
st.markdown(f"## 📈 New FOMC Analysis")

session_manager = SessionStateManager() # Initialize SessionStateManager
current_request_id = st.session_state.current_request_id

st.markdown(f"### Select FOMC Meeting")
available_meetings = get_fomc_meeting_dates(start=date(2020, 1, 1), end=date.today()) # Assumed from source.py

col1, col2 = st.columns([2, 1])
with col1:
    selected_meeting = st.selectbox(
        "Meeting Date",
        options=available_meetings,
        format_func=lambda d: d.strftime("%B %d, %Y"),
        index=0,
        key="selected_meeting_date"
    )

with col2:
    st.markdown(f"#### Available Documents")
    docs_available = check_document_availability(selected_meeting) # Assumed from source.py
    for doc_type, available in docs_available.items():
        icon = "✅" if available else "❌"
        st.markdown(f"{icon} {doc_type.replace('_', ' ').title()}")
st.divider()
```
*   **Function Calls**:
    *   `session_manager = SessionStateManager()`: Initializes the session state manager.
    *   `get_fomc_meeting_dates()`: (Assumed from `source.py`) to fetch a list of available FOMC meeting dates.
    *   `check_document_availability()`: (Assumed from `source.py`) to check if statement, minutes, press conference are available for the selected date.

**2. Analysis Options**:

```python
st.markdown(f"### Analysis Options")
col1, col2 = st.columns(2)

with col1:
    include_historical_context = st.checkbox(
        "Include Historical Context",
        value=True,
        help="Compare current meeting to prior 3 meetings for tone shifts and surprises."
    )
    num_themes_to_extract = st.slider(
        "Number of Themes to Extract",
        min_value=3,
        max_value=10,
        value=5
    )

with col2:
    auto_validate_citations = st.checkbox(
        "Auto-Validate Citations",
        value=True,
        help="Automatically verify if all generated citations reference real source text."
    )
    minimum_confidence_threshold = st.slider(
        "Minimum Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.7,
        step=0.05,
        help="Only display themes, tone, and surprises with confidence above this threshold."
    )
st.divider()
```

**3. Execution and Progress Tracking**:

```python
st.markdown(f"### Execute Analysis")
if st.button("🚀 Start Analysis", type="primary", use_container_width=True, key="start_analysis_button") and st.session_state.active_workflow_task is None:
    # Initialize request
    request_id = session_manager.start_request(selected_meeting)
    st.session_state.current_request_id = request_id # Set current_request_id

    # Progress tracking
    progress_bar = st.progress(0, text="Initializing workflow...")
    status_text = st.empty()
    log_container = st.container()

    async def run_analysis_workflow():
        workflow = create_fomc_workflow() # Assumed from source.py
        initial_state = AgentState( # Assumed from source.py
            meeting_date=selected_meeting,
            request_id=request_id,
            requested_at=datetime.now(),
            config={ # Pass config to workflow state
                "include_historical_context": include_historical_context,
                "num_themes_to_extract": num_themes_to_extract,
                "auto_validate_citations": auto_validate_citations,
                "minimum_confidence_threshold": minimum_confidence_threshold,
            }
        )

        step_count = 0
        total_steps = 10 # Approximate number of LangGraph nodes/steps in the workflow (e.g., plan, ingest, theme, tone, surprise, memo, validate, human_review, etc.)
        final_state = None

        try:
            async for event in workflow.astream(initial_state, {"configurable": {"thread_id": request_id}}): # Thread_id for LangGraph checkpointing
                final_state = event
                current_node = event.get("__node__", "unknown")
                if current_node != "unknown": # Skip initial state if just state dict
                    step_count += 1
                    progress = min(step_count / total_steps, 0.95) # Cap at 95% before final completion
                    progress_bar.progress(progress, text=f"Running: **{current_node.replace('_', ' ').title()}**")
                    status_text.markdown(f"Current Step: `{current_node}`")
                    with log_container:
                        st.markdown(f"✓ Completed: `{current_node}`")
                    
                    # Update session state with latest full state (optional, but good for debugging/resilience)
                    session_manager.update_step(request_id, current_node, event) # Assumed from source.py

            progress_bar.progress(1.0, text="Analysis Complete!")
            status_text.success("Workflow finished successfully!")

            # Store final memo data in session state for display
            if final_state and final_state.get("draft_memo"):
                st.session_state.latest_memo_data = FOMCMemo(**final_state["draft_memo"]) # Convert dict to Pydantic model
                # Update request history for pending reviews/history
                # This should be handled by SessionStateManager.update_step or similar at relevant nodes
                # For demo, let's ensure it's in request_history with final status
                found = False
                for i, req in enumerate(st.session_state.request_history):
                    if req["request_id"] == request_id:
                        st.session_state.request_history[i]["status"] = final_state.get("draft_memo", {}).get("status", "completed")
                        st.session_state.request_history[i]["memo_id"] = st.session_state.latest_memo_data.memo_id
                        found = True
                        break
                if not found: # If it wasn't added earlier, add it now (e.g., if initial plan phase didn't add it)
                    st.session_state.request_history.append({
                        "request_id": request_id,
                        "meeting_date": selected_meeting,
                        "status": final_state.get("draft_memo", {}).get("status", "completed"),
                        "memo_id": st.session_state.latest_memo_data.memo_id,
                        "generated_at": datetime.now()
                    })
                
                if st.session_state.latest_memo_data.validation_report.all_checks_passed:
                    st.success("Analysis complete! Ready for review.")
                    st.markdown(f"[📝 Go to Review](pages/2_Review.py?id={request_id})")
                else:
                    st.warning("Analysis complete with warnings/errors. Review required.")
                    st.markdown(f"[📝 Go to Review](pages/2_Review.py?id={request_id})")

        except Exception as e:
            st.error(f"Analysis workflow failed: {e}")
            logger.error("Analysis workflow failed", request_id=request_id, error=str(e))
            status_text.error("Workflow failed!")
            progress_bar.progress(0.0, text="Failed!")
        finally:
            st.session_state.active_workflow_task = None # Clear the task

    # Run workflow asynchronously
    st.session_state.active_workflow_task = asyncio.create_task(run_analysis_workflow())
    st.rerun() # Rerun to show initial progress widgets

elif st.session_state.active_workflow_task is not None:
    st.info("Analysis is currently running. Please wait or navigate to another page.")
    # Show current status if already running
    if current_request_id and current_request_id in st.session_state.workflow_state:
        current_workflow_info = st.session_state.workflow_state[current_request_id]
        progress = current_workflow_info.get("progress", 0) # Assumed field in workflow_state
        current_step = current_workflow_info.get("current_step", "Initializing...")
        progress_bar = st.progress(progress, text=f"Running: **{current_step.replace('_', ' ').title()}**")
        status_text = st.empty()
        log_container = st.container()
        status_text.markdown(f"Current Step: `{current_step}`")
        with log_container:
            for log_entry in current_workflow_info.get("steps_completed", []): # Assumed field
                st.markdown(f"✓ Completed: `{log_entry['step']}`")
    st.stop() # Stop further execution until task completes or user navigates

# Display results if available after workflow completion
if st.session_state.latest_memo_data:
    st.markdown(f"---")
    st.markdown(f"### Last Generated Memo Preview")
    memo = st.session_state.latest_memo_data
    
    st.markdown(f"**Meeting Date:** {memo.meeting_date.strftime('%B %d, %Y')}")
    st.markdown(f"**Status:** <span class='status-{memo.status}'>{memo.status.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
    st.markdown(f"**Overall Confidence:** {memo.confidence_assessment.overall_confidence:.0%}")
    st.markdown(f"**Validation Status:** {'✅ Passed' if memo.validation_report.all_checks_passed else '❌ Failed'}")

    st.markdown(f"#### Executive Summary")
    st.markdown(f"{memo.executive_summary}")

    st.markdown(f"#### Key Themes")
    for theme in memo.themes:
        st.markdown(f"- **{theme.theme_name}** (Confidence: {theme.confidence:.0%}): {theme.description}")
    
    st.markdown(f"[📝 Go to Review for full details](pages/2_Review.py?id={current_request_id})")
```
*   **Function Calls**:
    *   `session_manager.start_request()`: (Assumed from `source.py`) to create a new entry in `st.session_state.workflow_state`.
    *   `create_fomc_workflow()`: (Assumed from `source.py`) to instantiate the LangGraph workflow.
    *   `AgentState()`: (Assumed from `source.py`) to create the initial state.
    *   `workflow.astream(initial_state, {"configurable": {"thread_id": request_id}})`: Executes the LangGraph workflow. This implicitly calls many functions from `source.py` via the LangGraph nodes:
        *   `planning_node` (uses `get_available_documents`, `get_historical_context` implicitly)
        *   `ingestion_node` (calls `fetch_fomc_statement`, `fetch_fomc_minutes`, `fetch_press_conference`, `index_document`)
        *   `theme_analysis_node` (calls `extract_themes`)
        *   `tone_analysis_node` (calls `compute_tone_score`)
        *   `surprise_detection_node` (calls `detect_surprises`)
        *   `memo_generation_node` (calls `generate_fomc_memo`)
        *   `validation_node` (calls `ValidationPipeline.validate` which in turn calls `validate_citation`, `check_hallucination`, `verify_tone_bounds`, `no_recommendation_check`)
        *   `human_review_node`
    *   `session_manager.update_step()`: (Assumed from `source.py`) to update progress and status.
    *   `FOMCMemo()`: Pydantic model from `source.py` used to reconstruct memo from workflow output.

---

#### `pages/2_Review.py` (Human Review Page)

**1. Page Title and Memo Selection**:

```python
st.markdown(f"## 📝 Review Pending Analysis")

session_manager = SessionStateManager() # Initialize SessionStateManager

# Get pending reviews
pending_reviews = get_pending_reviews() # Assumed from source.py

# If a specific request_id is passed via URL parameter, select it
query_params = st.query_params
preselect_request_id = query_params.get("id")

selected_review = None
selected_index = 0

if pending_reviews:
    # Find the index of the preselected request_id
    if preselect_request_id:
        for i, review_item in enumerate(pending_reviews):
            if review_item["request_id"] == preselect_request_id:
                selected_index = i
                break
    
    selected_review = st.selectbox(
        "Select Analysis to Review",
        options=pending_reviews,
        format_func=lambda r: f"{r['meeting_date'].strftime('%B %d, %Y')} - {r['status'].replace('_', ' ').title()}",
        index=selected_index,
        key="selected_review_selectbox"
    )
    
    # Store current memo ID in session state
    if selected_review:
        st.session_state.selected_review_memo_id = selected_review["memo_id"]
        st.session_state.current_request_id = selected_review["request_id"] # Also set current_request_id for consistency

else:
    st.info("No analyses pending review.")
    st.stop()

st.divider()

# Load the memo and source documents
if st.session_state.selected_review_memo_id:
    memo = load_memo(st.session_state.selected_review_memo_id) # Assumed from source.py
    documents = load_source_documents(st.session_state.current_request_id) # Assumed from source.py (to verify citations)
else:
    st.error("Please select an analysis to review.")
    st.stop()

# Display Memo Summary
st.markdown(f"### FOMC Memo: {memo.meeting_date.strftime('%B %d, %Y')}")
st.markdown(f"**Status:** <span class='status-{memo.status}'>{memo.status.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
st.markdown(f"**Generated At:** {memo.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")

st.divider()
```
*   **Function Calls**:
    *   `session_manager = SessionStateManager()`: Initializes the session state manager.
    *   `get_pending_reviews()`: (Assumed from `source.py`) Fetches a list of pending review items from `st.session_state.request_history` (filtered by status).
    *   `load_memo(memo_id)`: (Assumed from `source.py`) Retrieves the full `FOMCMemo` object by its ID.
    *   `load_source_documents(request_id)`: (Assumed from `source.py`) Retrieves the original `FOMCDocument` objects ingested for this analysis, needed for citation verification.

**2. Memo Content Display (Tabs for better organization)**:

```python
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Summary", "Themes", "Tone Analysis", "Surprises", "Validation", "Audit Trail"])

with tab1: # Summary
    st.markdown(f"#### Executive Summary")
    st.markdown(f"{memo.executive_summary}")
    st.markdown(f"#### Market Implications")
    st.markdown(f"{memo.market_implications}")
    if memo.historical_context_summary:
        st.markdown(f"#### Historical Context Summary")
        st.markdown(f"{memo.historical_context_summary}")
    
with tab2: # Themes
    st.markdown(f"#### Key Themes Extracted")
    for theme in memo.themes:
        with st.expander(f"**{theme.theme_name}** (Confidence: {theme.confidence:.0%})"):
            st.markdown(f"{theme.description}")
            st.markdown(f"**Keywords:** {', '.join(theme.keywords)}")
            if theme.citations:
                st.markdown(f"**Supporting Citations:**")
                for cite in theme.citations:
                    st.markdown(f"""
                        <div class="citation-card">
                            <small>Document ID: `{cite.document_id}` | Section: `{cite.section_id}` | Paragraph: `{cite.paragraph_number}`</small><br>
                            <em>"{cite.quote}"</em>
                        </div>
                    """, unsafe_allow_html=True)

with tab3: # Tone Analysis
    st.markdown(f"#### Overall Tone Assessment")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        tone_label = "Hawkish" if memo.tone_analysis.score > 0 else ("Dovish" if memo.tone_analysis.score < 0 else "Neutral")
        st.metric("Overall Tone", f"{memo.tone_analysis.score:+.2f}", tone_label)
    with col_b:
        st.metric("Confidence", f"{memo.tone_analysis.confidence:.0%}")
    with col_c:
        if memo.tone_delta:
            st.metric("Delta vs Prior", f"{memo.tone_delta.delta:+.2f}", memo.tone_delta.delta_significance.title())
    
    st.markdown(f"##### Tone Components Breakdown")
    # Horizontal bar chart for tone components
    components = memo.tone_analysis.components
    component_scores = [
        components.inflation_stance,
        components.employment_stance,
        components.growth_outlook,
        components.policy_bias,
        components.uncertainty_level # Added uncertainty_level as per Pydantic model
    ]
    component_names = ['Inflation Stance', 'Employment Outlook', 'Growth Outlook', 'Policy Bias', 'Uncertainty Level']
    
    fig = go.Figure(go.Bar(
        x=component_scores,
        y=component_names,
        orientation='h',
        marker_color=[
            '#ff6b6b' if v > 0.05 else ('#4ade80' if v < -0.05 else '#cccccc') for v in component_scores
        ], # Red for hawkish (positive), Green for dovish (negative), Gray for neutral
        name="Tone Component Scores"
    ))
    fig.update_layout(
        title="Tone Components: (-1=Dovish, +1=Hawkish)",
        xaxis_title="Score",
        height=350,
        margin=dict(l=150), # Adjust left margin to prevent y-axis labels from being cut off
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"##### Explanation")
    st.markdown(f"{memo.tone_analysis.explanation}")
    st.markdown(f"##### Supporting Citations")
    if memo.tone_analysis.citations:
        for cite in memo.tone_analysis.citations:
            st.markdown(f"""
                <div class="citation-card">
                    <small>Document ID: `{cite.document_id}` | Section: `{cite.section_id}` | Paragraph: `{cite.paragraph_number}`</small><br>
                    <em>"{cite.quote}"</em>
                </div>
            """, unsafe_allow_html=True)

with tab4: # Surprises
    st.markdown(f"#### Detected Policy Surprises")
    if memo.surprises:
        for s in memo.surprises:
            with st.expander(f"**{s.category.replace('_', ' ').title()}** (Relevance: {s.market_relevance.title()}, Confidence: {s.confidence:.0%})"):
                st.markdown(f"{s.description}")
                if s.citations:
                    st.markdown(f"**Supporting Citations:**")
                    for cite in s.citations:
                        st.markdown(f"""
                            <div class="citation-card">
                                <small>Document ID: `{cite.document_id}` | Section: `{cite.section_id}` | Paragraph: `{cite.paragraph_number}`</small><br>
                                <em>"{cite.quote}"</em>
                            </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("No notable surprises detected for this meeting.")

with tab5: # Validation Report
    st.markdown(f"#### Automated Validation Report")
    validation_report = memo.validation_report
    if validation_report.all_checks_passed:
        st.success("✅ All automated validation checks passed!")
    else:
        st.warning("⚠️ Some automated validation checks failed or raised warnings. Review required.")

    st.markdown(f"**Validation Timestamp:** {validation_report.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(f"---")

    for check in validation_report.checks:
        icon = "✅" if check.passed else ("⚠️" if check.severity == "warning" else "❌")
        with st.expander(f"{icon} **{check.check_name.replace('_', ' ').title()}** (Severity: `{check.severity.upper()}`)", expanded=not check.passed):
            st.markdown(f"{check.details}")
    
    st.markdown(f"---")
    st.markdown(f"#### Confidence Assessment")
    conf = memo.confidence_assessment
    st.progress(conf.overall_confidence, text=f"Overall Confidence: **{conf.overall_confidence:.0%}**")
    st.progress(conf.evidence_strength, text=f"Evidence Strength: **{conf.evidence_strength:.0%}**")
    st.progress(conf.citation_coverage, text=f"Citation Coverage: **{conf.citation_coverage:.0%}**")
    
    if conf.flags:
        st.markdown(f"**Flags/Concerns:**")
        for flag in conf.flags:
            st.markdown(f"- ⚠️ {flag}")

with tab6: # Audit Trail Preview
    st.markdown(f"#### Audit Trail (Last 5 Entries)")
    # Assuming AuditTrail is stored in workflow_state for this request_id
    workflow_details = session_manager.get_state(st.session_state.current_request_id) # Assumed from source.py
    if workflow_details and workflow_details.get("execution_log"): # Assuming execution_log is part of AgentState
        audit_log = workflow_details["execution_log"]
        if audit_log:
            # Display audit log entries in JSON format, truncated for brevity
            st.json([entry.model_dump() for entry in audit_log[-5:]]) # Show last 5
        else:
            st.info("No audit log entries found yet.")
    else:
        st.info("No audit log available for this analysis.")
```
*   **Formula Handling**:
    *   No specific formula is defined in `source.py` output for this section directly, but the tone components are numerical scores. The "Tone Component Breakdown" chart visualizes values between -1.0 and +1.0.

**3. Human Review Actions**:

```python
st.divider()
st.markdown(f"### 💬 Review Decision")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("✅ Approve", type="primary", use_container_width=True, key="approve_button"):
        st.session_state.review_action = "approved"
        st.rerun()
with col2:
    if st.button("✏️ Request Revision", use_container_width=True, key="revise_button"):
        st.session_state.review_action = "revision_requested"
        st.rerun()
with col3:
    if st.button("❌ Reject", use_container_width=True, key="reject_button"):
        st.session_state.review_action = "rejected"
        st.rerun()

if st.session_state.review_action:
    action_title = st.session_state.review_action.replace("_", " ").title()
    st.markdown(f"#### {action_title} - Add Comments")
    
    comments = st.text_area(
        "Comments (required)",
        value=st.session_state.reviewer_comments, # Retain comments if rerunning
        placeholder="Explain your decision and provide feedback...",
        height=150,
        key="review_comments_textarea"
    )
    st.session_state.reviewer_comments = comments # Update session state with current comments

    if st.button("Submit Decision", key="submit_decision_button"):
        if not comments.strip():
            st.error("Comments are required to submit your decision.")
        else:
            # Record decision using SessionStateManager
            request_id = st.session_state.current_request_id
            decision_action = st.session_state.review_action
            
            # This call should ideally update the workflow_state and request_history
            session_manager.record_human_decision(
                request_id=request_id,
                decision=decision_action,
                comments=comments,
                reviewer_id="Alex Chen" # Placeholder, in a real app this would be authenticated user
            )
            
            # Update the memo's status based on the decision
            update_memo_status( # Assumed from source.py
                memo_id=memo.memo_id,
                status=decision_action, # Set the new status
                review_action=ReviewAction( # Create ReviewAction Pydantic model
                    timestamp=datetime.now(),
                    reviewer_id="Alex Chen",
                    action=decision_action,
                    comments=comments,
                    version_reviewed=memo.version
                )
            )

            st.success(f"Decision '{action_title}' recorded for memo ID: {memo.memo_id}")
            st.balloons()
            
            # Clear state and navigate away
            del st.session_state.review_action
            st.session_state.reviewer_comments = ""
            st.session_state.selected_review_memo_id = None
            st.session_state.current_request_id = None
            st.switch_page("pages/2_Review.py") # Go back to pending reviews list (which will now be updated)
```
*   **Function Calls**:
    *   `session_manager.record_human_decision()`: (Assumed from `source.py`) Logs the human decision and updates the workflow state.
    *   `update_memo_status()`: (Assumed from `source.py`) Updates the status of the `FOMCMemo` in persistent storage and adds a `ReviewAction` to its `review_history`.
    *   `ReviewAction()`: Pydantic model from `source.py`.

---

#### `pages/3_History.py` (Analysis History Page)

**1. Page Title and History Display**:

```python
st.markdown(f"## 📚 Analysis History")

session_manager = SessionStateManager() # Initialize SessionStateManager

# Filter history for completed (approved/rejected) analyses
history_items = [
    item for item in st.session_state.request_history
    if item["status"] in ["approved", "rejected", "revision_requested"] # Include revision requested for full history
]

if history_items:
    selected_history_item = st.selectbox(
        "Select Past Analysis",
        options=history_items,
        format_func=lambda r: f"{r['meeting_date'].strftime('%B %d, %Y')} - {r['status'].replace('_', ' ').title()}",
        key="selected_history_selectbox"
    )
    
    st.session_state.selected_review_memo_id = selected_history_item["memo_id"]
    st.session_state.current_request_id = selected_history_item["request_id"] # For audit trail lookup
else:
    st.info("No past analyses found in history.")
    st.stop()

st.divider()

if st.session_state.selected_review_memo_id:
    memo = load_memo(st.session_state.selected_review_memo_id) # Assumed from source.py
    # documents = load_source_documents(st.session_state.current_request_id) # Not strictly needed for display here

    st.markdown(f"### FOMC Memo: {memo.meeting_date.strftime('%B %d, %Y')}")
    st.markdown(f"**Status:** <span class='status-{memo.status}'>{memo.status.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
    st.markdown(f"**Generated At:** {memo.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(f"**Last Reviewed:** {memo.review_history[-1].timestamp.strftime('%Y-%m-%d %H:%M:%S')} by {memo.review_history[-1].reviewer_id}" if memo.review_history else "N/A")
    st.markdown(f"**Overall Confidence:** {memo.confidence_assessment.overall_confidence:.0%}")
    st.markdown(f"**Validation Status:** {'✅ Passed' if memo.validation_report.all_checks_passed else '❌ Failed'}")
    
    st.divider()

    tabs_history = st.tabs(["Tone Trajectory", "Citation-Theme Network", "Full Memo View", "Audit Log"])

    with tabs_history[0]: # Tone Trajectory
        st.markdown(f"#### Historical Tone Trajectory")
        # Fetch mock historical tone scores for the chart
        historical_scores = mock_prior_tone_scores + [memo.tone_analysis] # Example, should fetch from persistent store
        meeting_dates = [score.citations[0].document_id.split('_')[2] for score in historical_scores] # Extract dates

        st.markdown(r"$$ S_{{tone}} = \frac{{1}}{{N}} \sum_{{i=1}}^{{N}} w_i \cdot s_i $$")
        st.markdown(r"where $S_{{tone}}$ is the overall tone score, $N$ is the number of components, $w_i$ is the weighting of component $i$, and $s_i$ is the sentiment score for component $i$.")

        render_tone_trajectory(historical_scores, memo.tone_analysis, [date.fromisoformat(d) for d in meeting_dates]) # Assumed from source.py
        
    with tabs_history[1]: # Citation-Theme Network
        st.markdown(f"#### Citation-Theme Relationship Network")
        render_citation_network(memo.all_citations, memo.themes) # Assumed from source.py
        st.markdown(f"Visualize the interconnectedness of extracted themes and their supporting evidence. ")

    with tabs_history[2]: # Full Memo View (similar to Review page, but read-only)
        st.markdown(f"##### Executive Summary")
        st.markdown(f"{memo.executive_summary}")
        st.markdown(f"##### Key Themes Extracted")
        for theme in memo.themes:
            with st.expander(f"**{theme.theme_name}** (Confidence: {theme.confidence:.0%})"):
                st.markdown(f"{theme.description}")
                st.markdown(f"**Keywords:** {', '.join(theme.keywords)}")
                if theme.citations:
                    st.markdown(f"**Supporting Citations:**")
                    for cite in theme.citations:
                        st.markdown(f"""
                            <div class="citation-card">
                                <small>Document ID: `{cite.document_id}` | Section: `{cite.section_id}` | Paragraph: `{cite.paragraph_number}`</small><br>
                                <em>"{cite.quote}"</em>
                            </div>
                        """, unsafe_allow_html=True)
        
        st.markdown(f"##### Overall Tone Assessment")
        st.markdown(f"Score: {memo.tone_analysis.score:+.2f} (Confidence: {memo.tone_analysis.confidence:.0%})")
        if memo.tone_delta:
            st.markdown(f"Delta vs Prior: {memo.tone_delta.delta:+.2f} ({memo.tone_delta.delta_significance.title()})")
        st.markdown(f"Explanation: {memo.tone_analysis.explanation}")

        st.markdown(f"##### Detected Policy Surprises")
        if memo.surprises:
            for s in memo.surprises:
                st.markdown(f"- **{s.category.replace('_', ' ').title()}** (Relevance: {s.market_relevance.title()}, Confidence: {s.confidence:.0%}): {s.description}")
        else:
            st.info("No notable surprises detected.")
        
        st.markdown(f"##### Market Implications")
        st.markdown(f"{memo.market_implications}")

        st.markdown(f"##### Automated Validation Report")
        if memo.validation_report.all_checks_passed:
            st.success("✅ All automated validation checks passed!")
        else:
            st.warning("⚠️ Some automated validation checks failed or raised warnings.")
        for check in memo.validation_report.checks:
            icon = "✅" if check.passed else ("⚠️" if check.severity == "warning" else "❌")
            with st.expander(f"{icon} **{check.check_name.replace('_', ' ').title()}** (Severity: `{check.severity.upper()}`)"):
                st.markdown(f"{check.details}")
        
    with tabs_history[3]: # Audit Log
        st.markdown(f"#### Full Audit Log for Request ID: `{st.session_state.current_request_id}`")
        workflow_details = session_manager.get_state(st.session_state.current_request_id)
        if workflow_details and workflow_details.get("execution_log"):
            st.json([entry.model_dump() for entry in workflow_details["execution_log"]])
        else:
            st.info("No detailed audit log entries found for this request.")

else:
    st.info("Please select an analysis from the history to view its details.")
```
*   **Function Calls**:
    *   `session_manager = SessionStateManager()`: Initializes the session state manager.
    *   `load_memo()`: (Assumed from `source.py`) Retrieves the full `FOMCMemo` object.
    *   `mock_prior_tone_scores`: A list of `ToneScore` objects from `source.py` (mocked historical data).
    *   `render_tone_trajectory()`: (Assumed from `source.py`) Visualizes the evolution of tone scores over time.
    *   `render_citation_network()`: (Assumed from `source.py`) Visualizes the relationship between themes and their supporting citations.
    *   `session_manager.get_state()`: (Assumed from `source.py`) to retrieve the full `AgentState` including the `execution_log` for the audit trail.
    *   `FOMCMemo`, `ToneScore`, `Citation`, `ThemeExtraction`, `Surprise`, `ValidationReport`, `ConfidenceAssessment`, `ReviewAction`: Pydantic models from `source.py` for structured data display.

---

#### `pages/4_Settings.py` (Settings Page)

```python
st.markdown(f"## ⚙️ Settings and Configuration")

st.markdown(f"This page would allow configuration of API keys, model parameters, and other system settings.")
st.markdown(f"**API Keys (e.g., OpenAI, Anthropic)**: Input fields to set/update API keys, likely stored securely.")
st.markdown(f"**Model Configuration**: Dropdowns/sliders for default LLM model (`gpt-4-turbo`), embedding model, and temperature settings for different agents (planner, analyst, validator).")
st.markdown(f"**Memory Persistence**: Options to configure ChromaDB persistence directory.")
st.markdown(f"**Validation Thresholds**: Sliders for minimum confidence thresholds, max retries.")
st.markdown(f"**Logging**: Options to adjust log level (`INFO`, `DEBUG`) and format (`JSON`, `console`).")

# Example placeholder for a setting:
# api_key = st.text_input("OpenAI API Key", type="password", help="Your OpenAI API key for LLM interactions.")
# if api_key:
#    # Logic to save API key securely, e.g., to .env or secrets
#    st.success("API Key updated (demonstration only).")

st.info("Settings functionality is conceptual and not fully implemented in this pedagogical demo.")
```

---
