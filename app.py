
import streamlit as st
import asyncio
from datetime import date, datetime, timedelta
from typing import List, Optional, Literal, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from source import *  # Assuming source.py has been refactored to remove top-level 'await'

st.set_page_config(
    page_title="QuLab: FOMC Research Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: FOMC Research Agent")
st.divider()

# CSS Styles
st.markdown("""
<style>
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

# Session State Initialization
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = {}
if "request_history" not in st.session_state:
    st.session_state.request_history = []
if "current_request_id" not in st.session_state:
    st.session_state.current_request_id = None
if "selected_review_memo_id" not in st.session_state:
    st.session_state.selected_review_memo_id = None
if "review_action" not in st.session_state:
    st.session_state.review_action = None
if "reviewer_comments" not in st.session_state:
    st.session_state.reviewer_comments = ""
if "latest_memo_data" not in st.session_state:
    st.session_state.latest_memo_data = None
if "active_workflow_task" not in st.session_state:
    st.session_state.active_workflow_task = None

# Fix for "await outside function" error from source.py:
# The original error indicates that `current_fomc_documents = await main_ingestion()`
# was likely at the top level of `source.py`. This causes a SyntaxError upon import.
# The fix requires `source.py` itself to be refactored such that `main_ingestion()`
# is defined as an `async` function, but *not* called with `await` at the module's top level.
# The `app.py` then correctly manages its asynchronous execution and state.

if "fomc_documents_initialized" not in st.session_state:
    st.session_state.fomc_documents_initialized = False
if "current_fomc_documents" not in st.session_state:
    st.session_state.current_fomc_documents = []  # Initialize as empty list

# Initialize Manager
session_manager = SessionStateManager()

# Schedule the async document loading if not already done
if not st.session_state.fomc_documents_initialized:
    st.info("Loading initial FOMC documents (this may take a moment)...")
    try:
        # Run the async function synchronously using asyncio.run()
        st.session_state.current_fomc_documents = asyncio.run(main_ingestion())
        st.session_state.fomc_documents_initialized = True
        st.rerun()  # Rerun to display the loaded documents
    except Exception as e:
        st.error(f"Failed to load FOMC documents: {e}")
        # Mark as initialized to prevent infinite retries
        st.session_state.fomc_documents_initialized = True
        # Continue with empty documents for demo purposes

# Sidebar Navigation
with st.sidebar:
    st.markdown(f"**Navigation**")
    if st.button("🏠 Home", width='stretch', key="nav_home_sidebar"):
        st.session_state.current_page = "home"
        st.rerun()
    if st.button("📈 New Analysis", width='stretch', key="nav_new_analysis_sidebar"):
        st.session_state.current_page = "analysis"
        st.rerun()

    # get_pending_review_count() should be defined in source.py
    pending_count = get_pending_review_count()
    if st.button(f"📝 Pending Reviews ({pending_count})", width='stretch', key="nav_pending_reviews_sidebar"):
        st.session_state.current_page = "review"
        st.rerun()

    if st.button("📚 Analysis History", width='stretch', key="nav_analysis_history_sidebar"):
        st.session_state.current_page = "history"
        st.rerun()

    st.divider()
    st.markdown(f"#### System Status")
    col1, col2 = st.columns(2)
    with col1:
        # Display count of loaded documents
        st.metric("Documents", str(
            len(st.session_state.current_fomc_documents)), "+3")
    with col2:
        st.metric("Analyses", str(len(st.session_state.request_history)), "+1")

# Page Functions


def render_home():
    st.markdown(f"### Welcome to the FOMC Insights Dashboard")
    st.markdown(
        f"Transform Federal Reserve communications into structured, auditable investment research for Quantitative Analysts.")

    st.divider()

    st.markdown(f"### Get Started")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"#### 📈 New Analysis")
        st.markdown(f"Request analysis of FOMC meeting materials.")
        st.markdown(f"- Statement + Minutes + Press Conference")
        st.markdown(f"- Theme extraction with citations")
        st.markdown(f"- Tone scoring and delta computation")
        if st.button("Start New Analysis", key="nav_analysis_card", width='stretch'):
            st.session_state.current_page = "analysis"
            st.rerun()

    with col2:
        st.markdown(f"#### 📝 Pending Reviews")
        pending_count = get_pending_review_count()
        st.markdown(f"Review and approve draft memos.")
        st.markdown(f"- Citation verification")
        st.markdown(f"- Confidence assessment")
        st.markdown(f"- Approve/Revise/Reject workflow")
        if st.button(f"Review ({pending_count} pending)", key="nav_review_card", width='stretch'):
            st.session_state.current_page = "review"
            st.rerun()

    with col3:
        st.markdown(f"#### 📚 Analysis History")
        st.markdown(f"Browse past analyses and trends.")
        st.markdown(f"- Historical tone trajectory")
        st.markdown(f"- Theme evolution")
        st.markdown(f"- Market reaction correlation")
        if st.button("View History", key="nav_history_card", width='stretch'):
            st.session_state.current_page = "history"
            st.rerun()


def render_new_analysis():
    st.markdown(f"## 📈 New FOMC Analysis")

    current_request_id = st.session_state.current_request_id

    st.markdown(f"### Select FOMC Meeting")
    available_meetings = get_fomc_meeting_dates(
        start=date(2020, 1, 1), end=date.today())

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_meeting = st.selectbox(
            "Meeting Date",
            options=available_meetings,
            format_func=lambda d: d.strftime("%B %d, %Y"),
            # Handle case where available_meetings might be empty
            index=0 if available_meetings else None,
            key="selected_meeting_date"
        )

    with col2:
        st.markdown(f"#### Available Documents")
        if selected_meeting:  # Only check if a meeting is selected
            docs_available = check_document_availability(selected_meeting)
            for doc_type, available in docs_available.items():
                icon = "✅" if available else "❌"
                st.markdown(f"{icon} {doc_type.replace('_', ' ').title()}")
        else:
            st.info("No meeting dates available or selected.")
    st.divider()

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

    st.markdown(f"### Execute Analysis")
    # Only allow starting analysis if a meeting is selected
    if selected_meeting and st.button("🚀 Start Analysis", type="primary", width='stretch', key="start_analysis_button") and st.session_state.active_workflow_task is None:
        # Initialize request
        request_id = session_manager.start_request(selected_meeting)
        st.session_state.current_request_id = request_id

        # Progress tracking
        progress_bar = st.progress(0, text="Initializing workflow...")
        status_text = st.empty()
        log_container = st.container()

        async def run_analysis_workflow():
            workflow = create_fomc_workflow()
            initial_state = AgentState(
                meeting_date=selected_meeting,
                request_id=request_id,
                requested_at=datetime.now(),
                config={
                    "include_historical_context": include_historical_context,
                    "num_themes_to_extract": num_themes_to_extract,
                    "auto_validate_citations": auto_validate_citations,
                    "minimum_confidence_threshold": minimum_confidence_threshold,
                }
            )

            step_count = 0
            total_steps = 10
            final_state = None
            logged_nodes = set()  # Track which nodes we've already logged

            try:
                # Use astream_events for more granular control and status updates
                async for event in workflow.astream_events(initial_state, {"configurable": {"thread_id": request_id}}, version="v1"):
                    event_type = event["event"]
                    current_node = None  # Initialize current_node for each event
                    is_start_event = False

                    if event_type == "on_chain_start":
                        # This is usually the chain name
                        current_node = event["name"]
                        is_start_event = True
                    elif event_type == "on_tool_start":
                        current_node = event["name"]  # Tool name
                        is_start_event = True
                    elif event_type == "on_agent_action":
                        current_node = event["name"]  # Agent action
                        is_start_event = True
                    elif event_type == "on_chain_end":
                        # This is usually the chain name
                        current_node = event["name"]
                        is_start_event = False

                    # Update progress and status only for start events to avoid duplicates
                    if current_node and is_start_event and current_node not in logged_nodes:
                        logged_nodes.add(current_node)
                        step_count += 1
                        progress = min(step_count / total_steps, 0.95)
                        progress_bar.progress(
                            progress, text=f"Running: **{current_node.replace('_', ' ').title()}**")
                        status_text.markdown(f"Current Step: `{current_node}`")
                        with log_container:
                            st.markdown(f"✓ Started: `{current_node}`")

                        # Store detailed events and audit entries in session_manager
                        audit_entry = event.get("audit_entry")
                        session_manager.update_step(
                            request_id, current_node, progress, audit_entry)

                    # Assuming 'workflow' is the top-level chain name
                    if event_type == "on_chain_end" and event["name"] == "workflow":
                        # Extract final state from workflow output
                        final_state = event["data"]["output"]["output"]

                progress_bar.progress(1.0, text="Analysis Complete!")
                status_text.success("Workflow finished successfully!")

                # Update request history and latest memo data
                if final_state and final_state.get("draft_memo"):
                    st.session_state.latest_memo_data = FOMCMemo(
                        **final_state["draft_memo"])

                    found = False
                    for i, req in enumerate(st.session_state.request_history):
                        if req["request_id"] == request_id:
                            # Status is a string literal, not an enum
                            st.session_state.request_history[i]["status"] = st.session_state.latest_memo_data.status
                            st.session_state.request_history[i]["memo_id"] = st.session_state.latest_memo_data.memo_id
                            found = True
                            break
                    if not found:
                        st.session_state.request_history.append({
                            "request_id": request_id,
                            "meeting_date": selected_meeting,
                            "status": st.session_state.latest_memo_data.status,  # Status is a string literal
                            "memo_id": st.session_state.latest_memo_data.memo_id,
                            "generated_at": datetime.now()
                        })

                    if st.session_state.latest_memo_data.validation_report.all_checks_passed:
                        st.success("Analysis complete! Ready for review.")
                    else:
                        st.warning(
                            "Analysis complete with warnings/errors. Review required.")

            except Exception as e:
                st.error(f"Analysis workflow failed: {e}")
                status_text.error("Workflow failed!")
                progress_bar.progress(0.0, text="Failed!")
                # Mark the request in history as failed if it was created
                if current_request_id:
                    for i, req in enumerate(st.session_state.request_history):
                        if req["request_id"] == current_request_id:
                            st.session_state.request_history[i]["status"] = "failed"
                            break
            finally:
                st.session_state.active_workflow_task = None

        # Run the workflow synchronously
        asyncio.run(run_analysis_workflow())
        st.rerun()

    elif st.session_state.active_workflow_task is not None:
        st.info(
            "Analysis is currently running. Please wait or navigate to another page.")
        if current_request_id and current_request_id in st.session_state.workflow_state:
            current_workflow_info = st.session_state.workflow_state[current_request_id]
            # The progress and current_step updates might need more granular data from astream_events
            # For simplicity, we can show a general progress and last known step
            # This 'progress' needs to be managed in session_manager.update_step
            progress = current_workflow_info.get("progress", 0)
            current_step = current_workflow_info.get(
                "current_step", "Initializing...")

            progress_bar = st.progress(
                progress, text=f"Running: **{current_step.replace('_', ' ').title()}**")
            status_text = st.empty()
            log_container = st.container()
            status_text.markdown(f"Current Step: `{current_step}`")
            with log_container:
                # Display last few log entries or a summarized view
                # Show last 5 completed steps
                for log_entry in current_workflow_info.get("steps_completed", [])[-5:]:
                    st.markdown(f"✓ Completed: `{log_entry['step']}`")
        st.stop()  # Stop further execution while task is running

    if st.session_state.latest_memo_data:
        st.markdown(f"---")
        st.markdown(f"### Last Generated Memo Preview")
        memo = st.session_state.latest_memo_data

        st.markdown(
            f"**Meeting Date:** {memo.meeting_date.strftime('%B %d, %Y')}")
        # Status is a Literal string, not an enum
        st.markdown(
            f"**Status:** <span class='status-{memo.status}'>{memo.status.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
        st.markdown(
            f"**Overall Confidence:** {memo.confidence_assessment.overall_confidence:.0%}")
        st.markdown(
            f"**Validation Status:** {'✅ Passed' if memo.validation_report.all_checks_passed else '❌ Failed'}")

        st.markdown(f"#### Executive Summary")
        st.markdown(f"{memo.executive_summary}")

        st.markdown(f"#### Key Themes")
        for theme in memo.themes:
            st.markdown(
                f"- **{theme.theme_name}** (Confidence: {theme.confidence:.0%}): {theme.description}")

        if st.button("📝 Go to Review for full details", key="go_to_review_btn"):
            st.session_state.selected_review_memo_id = memo.memo_id
            st.session_state.current_request_id = current_request_id
            st.session_state.current_page = "review"
            st.rerun()


def render_review():
    st.markdown(f"## 📝 Review Pending Analysis")

    pending_reviews = get_pending_reviews()

    selected_review = None
    selected_index = 0

    # Try to find current selection in the list
    if st.session_state.current_request_id and pending_reviews:
        for i, review_item in enumerate(pending_reviews):
            if review_item["request_id"] == st.session_state.current_request_id:
                selected_index = i
                break
    # If memo ID is set but request ID isn't directly matching
    elif st.session_state.selected_review_memo_id and pending_reviews:
        for i, review_item in enumerate(pending_reviews):
            if review_item["memo_id"] == st.session_state.selected_review_memo_id:
                selected_index = i
                break

    if pending_reviews:
        selected_review = st.selectbox(
            "Select Analysis to Review",
            options=pending_reviews,
            format_func=lambda r: f"{r['meeting_date'].strftime('%B %d, %Y')} - {r['status'].replace('_', ' ').title()}",
            index=selected_index,
            key="selected_review_selectbox"
        )

        if selected_review:
            st.session_state.selected_review_memo_id = selected_review["memo_id"]
            st.session_state.current_request_id = selected_review["request_id"]

    else:
        st.info("No analyses pending review.")
        # If no pending reviews, clear any lingering selections
        st.session_state.selected_review_memo_id = None
        st.session_state.current_request_id = None
        st.stop()  # Stop execution if nothing to review

    st.divider()

    if st.session_state.selected_review_memo_id:
        memo = load_memo(st.session_state.selected_review_memo_id)
        # documents = load_source_documents(st.session_state.current_request_id) # kept for completeness
    else:
        st.error("Please select an analysis to review.")
        st.stop()

    st.markdown(f"### FOMC Memo: {memo.meeting_date.strftime('%B %d, %Y')}")
    # Status is a string literal, not an enum
    st.markdown(
        f"**Status:** <span class='status-{memo.status}'>{memo.status.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
    st.markdown(
        f"**Generated At:** {memo.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Summary", "Themes", "Tone Analysis", "Surprises", "Validation", "Audit Trail"])

    with tab1:
        st.markdown(f"#### Executive Summary")
        st.markdown(f"{memo.executive_summary}")
        st.markdown(f"#### Market Implications")
        st.markdown(f"{memo.market_implications}")
        if memo.historical_context_summary:
            st.markdown(f"#### Historical Context Summary")
            st.markdown(f"{memo.historical_context_summary}")

    with tab2:
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

    with tab3:
        st.markdown(f"#### Overall Tone Assessment")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            tone_label = "Hawkish" if memo.tone_analysis.score > 0 else (
                "Dovish" if memo.tone_analysis.score < 0 else "Neutral")
            st.metric("Overall Tone",
                      f"{memo.tone_analysis.score:+.2f}", tone_label)
        with col_b:
            st.metric("Confidence", f"{memo.tone_analysis.confidence:.0%}")
        with col_c:
            if memo.tone_delta:
                st.metric("Delta vs Prior", f"{memo.tone_delta.delta:+.2f}",
                          memo.tone_delta.delta_significance.title())

        st.markdown(f"##### Tone Components Breakdown")
        components = memo.tone_analysis.components
        component_scores = [
            components.inflation_stance,
            components.employment_stance,
            components.growth_outlook,
            components.policy_bias,
            components.uncertainty_level
        ]
        component_names = ['Inflation Stance', 'Employment Outlook',
                           'Growth Outlook', 'Policy Bias', 'Uncertainty Level']

        fig = go.Figure(go.Bar(
            x=component_scores,
            y=component_names,
            orientation='h',
            marker_color=[
                '#ff6b6b' if v > 0.05 else ('#4ade80' if v < -0.05 else '#cccccc') for v in component_scores
            ],
            name="Tone Component Scores"
        ))
        fig.update_layout(
            title="Tone Components: (-1=Dovish, +1=Hawkish)",
            xaxis_title="Score",
            height=350,
            margin=dict(l=150),
            template="plotly_dark"
        )
        st.plotly_chart(fig, width='stretch')

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

    with tab4:
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

    with tab5:
        st.markdown(f"#### Automated Validation Report")
        validation_report = memo.validation_report
        if validation_report.all_checks_passed:
            st.success("✅ All automated validation checks passed!")
        else:
            st.warning(
                "⚠️ Some automated validation checks failed or raised warnings. Review required.")

        st.markdown(
            f"**Validation Timestamp:** {validation_report.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(f"---")

        for check in validation_report.checks:
            icon = "✅" if check.passed else (
                "⚠️" if check.severity == "warning" else "❌")
            with st.expander(f"{icon} **{check.check_name.replace('_', ' ').title()}** (Severity: `{check.severity.upper()}`)", expanded=not check.passed):
                st.markdown(f"{check.details}")

        st.markdown(f"---")
        st.markdown(f"#### Confidence Assessment")
        conf = memo.confidence_assessment
        st.progress(conf.overall_confidence,
                    text=f"Overall Confidence: **{conf.overall_confidence:.0%}**")
        st.progress(conf.evidence_strength,
                    text=f"Evidence Strength: **{conf.evidence_strength:.0%}**")
        st.progress(conf.citation_coverage,
                    text=f"Citation Coverage: **{conf.citation_coverage:.0%}**")

        if conf.flags:
            st.markdown(f"**Flags/Concerns:**")
            for flag in conf.flags:
                st.markdown(f"- ⚠️ {flag}")

    with tab6:
        st.markdown(f"#### Audit Trail (Last 5 Entries)")
        workflow_details = session_manager.get_state(
            st.session_state.current_request_id)
        if workflow_details and workflow_details.get("execution_log"):
            # Assuming execution_log stores pydantic models, use model_dump
            audit_log_data = [entry.model_dump() if hasattr(
                entry, 'model_dump') else entry for entry in workflow_details["execution_log"]]
            if audit_log_data:
                st.json(audit_log_data[-5:])
            else:
                st.info("No audit log entries found yet.")
        else:
            st.info("No audit log available for this analysis.")

    st.divider()
    st.markdown(f"### 💬 Review Decision")

    col1,  col3 = st.columns(2)

    with col1:
        if st.button("✅ Approve", type="primary", width='stretch', key="approve_button"):
            st.session_state.review_action = "approved"
            st.rerun()
    with col3:
        if st.button("❌ Reject", width='stretch', key="reject_button"):
            st.session_state.review_action = "rejected"
            st.rerun()

    if st.session_state.review_action:
        action_title = st.session_state.review_action.replace("_", " ").title()
        st.markdown(f"#### {action_title} - Add Comments")

        comments = st.text_area(
            "Comments (required)",
            value=st.session_state.reviewer_comments,
            placeholder="Explain your decision and provide feedback...",
            height=150,
            key="review_comments_textarea"
        )
        st.session_state.reviewer_comments = comments

        if st.button("Submit Decision", key="submit_decision_button"):
            if not comments.strip():
                st.error("Comments are required to submit your decision.")
            else:
                request_id = st.session_state.current_request_id
                decision_action = st.session_state.review_action

                # Map UI action names to ReviewAction model values
                action_mapping = {
                    "approved": "approve",
                    "revision_requested": "revise",
                    "rejected": "reject"
                }
                review_action_value = action_mapping.get(
                    decision_action, "approve")

                # Map UI action names to FOMCMemo status values
                status_mapping = {
                    "approved": "approved",
                    "revision_requested": "pending_review",  # Revision -> back to pending
                    "rejected": "rejected"
                }
                status_value = status_mapping.get(
                    decision_action, "pending_review")

                session_manager.record_human_decision(
                    request_id=request_id,
                    decision=decision_action,
                    comments=comments,
                    reviewer_id="Alex Chen"
                )

                update_memo_status(
                    memo_id=memo.memo_id,
                    status=status_value,  # Use mapped status value
                    review_action=ReviewAction(
                        timestamp=datetime.now(),
                        reviewer_id="Alex Chen",
                        action=review_action_value,  # ReviewAction uses mapped value
                        comments=comments,
                        version_reviewed=memo.version
                    )
                )

                st.success(
                    f"Decision '{action_title}' recorded for memo ID: {memo.memo_id}")
                st.balloons()

                del st.session_state.review_action
                st.session_state.reviewer_comments = ""
                st.session_state.selected_review_memo_id = None
                st.session_state.current_request_id = None

                st.rerun()


def render_history():
    st.markdown(f"## 📚 Analysis History")

    history_items = [
        item for item in st.session_state.request_history
        # 'completed' status might be pending review
        if item["status"] in ["approved", "rejected", "revision_requested", "completed"]
    ]

    # Sort history items by meeting date, newest first
    history_items = sorted(
        history_items, key=lambda x: x["meeting_date"], reverse=True)

    if history_items:
        # Determine initial selection for history
        initial_selection_index = 0
        if st.session_state.selected_review_memo_id:
            for i, item in enumerate(history_items):
                if item["memo_id"] == st.session_state.selected_review_memo_id:
                    initial_selection_index = i
                    break

        selected_history_item = st.selectbox(
            "Select Past Analysis",
            options=history_items,
            format_func=lambda r: f"{r['meeting_date'].strftime('%B %d, %Y')} - {r['status'].replace('_', ' ').title()}",
            index=initial_selection_index,
            key="selected_history_selectbox"
        )

        if selected_history_item:
            st.session_state.selected_review_memo_id = selected_history_item["memo_id"]
            st.session_state.current_request_id = selected_history_item["request_id"]
    else:
        st.info("No past analyses found in history.")
        st.session_state.selected_review_memo_id = None
        st.session_state.current_request_id = None
        st.stop()

    st.divider()

    if st.session_state.selected_review_memo_id:
        memo = load_memo(st.session_state.selected_review_memo_id)

        st.markdown(
            f"### FOMC Memo: {memo.meeting_date.strftime('%B %d, %Y')}")
        # Status is a string literal, not an enum
        st.markdown(
            f"**Status:** <span class='status-{memo.status}'>{memo.status.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
        st.markdown(
            f"**Generated At:** {memo.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")

        last_review_info = "N/A"
        if memo.review_history:
            last_review = memo.review_history[-1]
            last_review_info = f"{last_review.timestamp.strftime('%Y-%m-%d %H:%M:%S')} by {last_review.reviewer_id}"
        st.markdown(f"**Last Reviewed:** {last_review_info}")

        st.markdown(
            f"**Overall Confidence:** {memo.confidence_assessment.overall_confidence:.0%}")
        st.markdown(
            f"**Validation Status:** {'✅ Passed' if memo.validation_report.all_checks_passed else '❌ Failed'}")

        st.divider()

        tabs_history = st.tabs(
            ["Tone Trajectory", "Citation-Theme Network", "Full Memo View", "Audit Log"])

        with tabs_history[0]:
            st.markdown(f"#### Historical Tone Trajectory")

            # Filter out the current memo's date from mock_prior_tone_scores if it exists there
            historical_scores_for_plot = []
            if mock_prior_tone_scores:  # Ensure mock_prior_tone_scores is not empty
                # Renamed 'score' to 'score_obj' to avoid conflict with `_` in sorted_data
                for score_obj in mock_prior_tone_scores:
                    if score_obj.citations and len(score_obj.citations) > 0:
                        try:
                            # Document_id format is "statement_YYYYMMDD" or "minutes_YYYYMMDD"
                            # Split by '_' and get the last part which is the date
                            doc_id_parts = score_obj.citations[0].document_id.split(
                                '_')
                            # Get last part (the date)
                            doc_id_part = doc_id_parts[-1]
                            score_date = datetime.strptime(
                                doc_id_part, '%Y%m%d').date()
                            if score_date != memo.meeting_date:
                                historical_scores_for_plot.append(
                                    (score_date, score_obj))
                        except (ValueError, IndexError):
                            # Handle cases where document_id might not match expected format
                            st.warning(
                                f"Could not parse date from document_id: {score_obj.citations[0].document_id}. Skipping this score for plot.")
                    else:
                        st.warning(
                            f"ToneAnalysis object without citations found in mock_prior_tone_scores. Skipping this score for plot.")

            # Add the current memo's tone analysis
            historical_scores_for_plot.append(
                (memo.meeting_date, memo.tone_analysis))

            # Sort scores and dates together by date for correct plotting order
            sorted_data = sorted(
                historical_scores_for_plot, key=lambda x: x[0])
            plot_meeting_dates = [d for d, _ in sorted_data]
            scores_to_plot = [s for _, s in sorted_data]

            st.markdown(
                r"$$ S_{tone} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot s_i $$")
            st.markdown(
                r"where $S_{tone}$ is the overall tone score, $N$ is the number of components, $w_i$ is the weighting of component $i$, and $s_i$ is the sentiment score for component $i$.")

            render_tone_trajectory(
                scores_to_plot, memo.tone_analysis, plot_meeting_dates)

        with tabs_history[1]:
            st.markdown(f"#### Citation-Theme Relationship Network")
            render_citation_network(memo.all_citations, memo.themes)
            st.markdown(
                f"Visualize the interconnectedness of extracted themes and their supporting evidence. ")

        with tabs_history[2]:
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
            st.markdown(
                f"Score: {memo.tone_analysis.score:+.2f} (Confidence: {memo.tone_analysis.confidence:.0%})")
            if memo.tone_delta:
                st.markdown(
                    f"Delta vs Prior: {memo.tone_delta.delta:+.2f} ({memo.tone_delta.delta_significance.title()})")
            st.markdown(f"Explanation: {memo.tone_analysis.explanation}")

            st.markdown(f"##### Detected Policy Surprises")
            if memo.surprises:
                for s in memo.surprises:
                    st.markdown(
                        f"- **{s.category.replace('_', ' ').title()}** (Relevance: {s.market_relevance.title()}, Confidence: {s.confidence:.0%}): {s.description}")
            else:
                st.info("No notable surprises detected.")

            st.markdown(f"##### Market Implications")
            st.markdown(f"{memo.market_implications}")

            st.markdown(f"##### Automated Validation Report")
            if memo.validation_report.all_checks_passed:
                st.success("✅ All automated validation checks passed!")
            else:
                st.warning(
                    "⚠️ Some automated validation checks failed or raised warnings.")
            for check in memo.validation_report.checks:
                icon = "✅" if check.passed else (
                    "⚠️" if check.severity == "warning" else "❌")
                with st.expander(f"{icon} **{check.check_name.replace('_', ' ').title()}** (Severity: `{check.severity.upper()}`)"):
                    st.markdown(f"{check.details}")

        with tabs_history[3]:
            st.markdown(
                f"#### Full Audit Log for Request ID: `{st.session_state.current_request_id}`")
            workflow_details = session_manager.get_state(
                st.session_state.current_request_id)
            if workflow_details and workflow_details.get("execution_log"):
                audit_log_data = [entry.model_dump() if hasattr(
                    entry, 'model_dump') else entry for entry in workflow_details["execution_log"]]
                st.json(audit_log_data)
            else:
                st.info("No detailed audit log entries found for this request.")

    else:
        st.info("Please select an analysis from the history to view its details.")


def render_settings():
    st.markdown(f"## ⚙️ Settings and Configuration")
    st.markdown(
        f"This page would allow configuration of API keys, model parameters, and other system settings.")
    st.markdown(
        f"**API Keys (e.g., OpenAI, Anthropic)**: Input fields to set/update API keys, likely stored securely.")
    st.markdown(f"**Model Configuration**: Dropdowns/sliders for default LLM model (`gpt-4-turbo`), embedding model, and temperature settings for different agents (planner, analyst, validator).")
    st.markdown(
        f"**Memory Persistence**: Options to configure ChromaDB persistence directory.")
    st.markdown(
        f"**Validation Thresholds**: Sliders for minimum confidence thresholds, max retries.")
    st.markdown(
        f"**Logging**: Options to adjust log level (`INFO`, `DEBUG`) and format (`JSON`, `console`).")
    st.info("Settings functionality is conceptual and not fully implemented in this pedagogical demo.")


# Main Routing
if st.session_state.current_page == "home":
    render_home()
elif st.session_state.current_page == "analysis":
    render_new_analysis()
elif st.session_state.current_page == "review":
    render_review()
elif st.session_state.current_page == "history":
    render_history()
elif st.session_state.current_page == "settings":
    render_settings()


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
