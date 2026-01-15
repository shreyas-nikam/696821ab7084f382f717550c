
import pytest
import sys
from streamlit.testing.v1 import AppTest
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

# --- MOCK CLASSES AND FUNCTIONS FOR source.py ---
# These mock objects mimic the structure and methods expected by the Streamlit application
# from the (unprovided) 'source.py' file.

class MockCitation:
    def __init__(self, document_id, section_id, paragraph_number, quote):
        self.document_id = document_id
        self.section_id = section_id
        self.paragraph_number = paragraph_number
        self.quote = quote
    
    def model_dump(self):
        return {
            "document_id": self.document_id,
            "section_id": self.section_id,
            "paragraph_number": self.paragraph_number,
            "quote": self.quote
        }

class MockTheme:
    def __init__(self, theme_name, description, keywords, confidence, citations):
        self.theme_name = theme_name
        self.description = description
        self.keywords = keywords
        self.confidence = confidence
        self.citations = citations

class MockToneComponents:
    def __init__(self, inflation_stance, employment_stance, growth_outlook, policy_bias, uncertainty_level):
        self.inflation_stance = inflation_stance
        self.employment_stance = employment_stance
        self.growth_outlook = growth_outlook
        self.policy_bias = policy_bias
        self.uncertainty_level = uncertainty_level

class MockToneAnalysis:
    def __init__(self, score, explanation, confidence, components, citations):
        self.score = score
        self.explanation = explanation
        self.confidence = confidence
        self.components = components if components is not None else MockToneComponents(0,0,0,0,0)
        self.citations = citations

class MockToneDelta:
    def __init__(self, delta, delta_significance):
        self.delta = delta
        self.delta_significance = delta_significance

class MockSurprise:
    def __init__(self, category, description, market_relevance, confidence, citations):
        self.category = category
        self.description = description
        self.market_relevance = market_relevance
        self.confidence = confidence
        self.citations = citations

class MockValidationCheck:
    def __init__(self, check_name, passed, severity, details):
        self.check_name = check_name
        self.passed = passed
        self.severity = severity
        self.details = details

class MockValidationReport:
    def __init__(self, all_checks_passed, validation_timestamp, checks):
        self.all_checks_passed = all_checks_passed
        self.validation_timestamp = validation_timestamp
        self.checks = checks

class MockConfidenceAssessment:
    def __init__(self, overall_confidence, evidence_strength, citation_coverage, flags):
        self.overall_confidence = overall_confidence
        self.evidence_strength = evidence_strength
        self.citation_coverage = citation_coverage
        self.flags = flags

class MockReviewAction:
    def __init__(self, timestamp, reviewer_id, action, comments, version_reviewed):
        self.timestamp = timestamp
        self.reviewer_id = reviewer_id
        self.action = action
        self.comments = comments
        self.version_reviewed = version_reviewed
    
    def model_dump(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "reviewer_id": self.reviewer_id,
            "action": self.action,
            "comments": self.comments,
            "version_reviewed": self.version_reviewed
        }

class MockFOMCMemo:
    def __init__(self, memo_id, meeting_date, generated_at, status, version,
                 executive_summary, market_implications, historical_context_summary,
                 themes, tone_analysis, tone_delta, surprises,
                 validation_report, confidence_assessment, review_history=None, **kwargs):
        # kwargs are to catch any extra fields from dict unpacking for easier mocking
        self.memo_id = memo_id
        self.meeting_date = meeting_date
        self.generated_at = generated_at
        self.status = status
        self.version = version
        self.executive_summary = executive_summary
        self.market_implications = market_implications
        self.historical_context_summary = historical_context_summary
        self.themes = themes
        self.tone_analysis = tone_analysis
        self.tone_delta = tone_delta
        self.surprises = surprises
        self.validation_report = validation_report
        self.confidence_assessment = confidence_assessment
        self.review_history = review_history if review_history is not None else []
        self.all_citations = self._get_all_citations()

    def _get_all_citations(self):
        citations = []
        for theme in self.themes:
            citations.extend(theme.citations)
        if self.tone_analysis:
            citations.extend(self.tone_analysis.citations)
        for surprise in self.surprises:
            citations.extend(surprise.citations)
        return citations

class MockAgentState:
    def __init__(self, meeting_date, request_id, requested_at, config):
        self.meeting_date = meeting_date
        self.request_id = request_id
        self.requested_at = requested_at
        self.config = config

class MockSessionStateManager:
    def __init__(self):
        self.requests = {}
        self.workflow_states = {}

    def start_request(self, meeting_date):
        request_id = f"req_{len(self.requests) + 1}"
        self.requests[request_id] = {
            "meeting_date": meeting_date,
            "status": "pending",
            "request_id": request_id,
            "generated_at": datetime.now(),
            "memo_id": None # Will be set later
        }
        self.workflow_states[request_id] = {
            "progress": 0,
            "current_step": "Initializing...",
            "steps_completed": [],
            "execution_log": []
        }
        return request_id

    def update_step(self, request_id, step_name, event):
        if request_id in self.workflow_states:
            self.workflow_states[request_id]["steps_completed"].append({"step": step_name, "event": event})
            self.workflow_states[request_id]["current_step"] = step_name
            self.workflow_states[request_id]["progress"] = min(self.workflow_states[request_id].get("progress", 0) + 0.1, 0.95)
            self.workflow_states[request_id]["execution_log"].append(
                MockReviewAction(datetime.now(), "System", f"Step {step_name} completed", f"Event data: {str(event)}", "1.0")
            )

    def record_human_decision(self, request_id, decision, comments, reviewer_id):
        if request_id in self.requests:
            self.requests[request_id]["status"] = decision
            # Simulate the app's logic where review history is part of the memo, not the request.
            # This mock might need refinement if review history is stored per request in reality.
            # For now, we update the status of the request itself.
            pass # The actual update happens in update_memo_status for the memo object.

    def get_state(self, request_id):
        return self.workflow_states.get(request_id)

    def get_all_requests(self):
        return list(self.requests.values())

# Mock data for various scenarios
mock_meeting_dates = [date(2023, 12, 13), date(2023, 11, 1), date(2023, 9, 20)]
mock_docs_available = {"statement": True, "minutes": True, "press_conference": True}
mock_citation = MockCitation("fomc_20231213_statement", "paragraph_5", 1, "Inflation remains elevated.")
mock_theme = MockTheme("Inflation Concerns", "The committee discussed persistent inflation pressures.", ["inflation", "PCE"], 0.85, [mock_citation])
mock_tone_components = MockToneComponents(0.6, 0.2, 0.1, 0.3, -0.1)
mock_tone_analysis = MockToneAnalysis(0.4, "Overall hawkish due to inflation focus.", 0.9, mock_tone_components, [mock_citation])
mock_tone_delta = MockToneDelta(0.1, "slight increase")
mock_surprise = MockSurprise("Rate Hike", "Unexpected rate hike signals hawkish shift.", "high", 0.75, [mock_citation])
mock_validation_check = MockValidationCheck("Citation Presence", True, "info", "All citations found.")
mock_validation_report = MockValidationReport(True, datetime.now(), [mock_validation_check])
mock_confidence_assessment = MockConfidenceAssessment(0.9, 0.8, 0.95, [])

mock_memo_id = "memo_123"
mock_fomc_memo = MockFOMCMemo(
    memo_id=mock_memo_id,
    meeting_date=mock_meeting_dates[0],
    generated_at=datetime.now() - timedelta(hours=1),
    status="pending_review",
    version="1.0",
    executive_summary="The December FOMC meeting emphasized a cautious but hawkish stance...",
    market_implications="Markets reacted with slight volatility...",
    historical_context_summary="Compared to previous meetings, the tone was incrementally more hawkish.",
    themes=[mock_theme],
    tone_analysis=mock_tone_analysis,
    tone_delta=mock_tone_delta,
    surprises=[mock_surprise],
    validation_report=mock_validation_report,
    confidence_assessment=mock_confidence_assessment,
    review_history=[]
)

# Mock prior tone scores for history page
mock_prior_tone_scores = [
    MockToneAnalysis(0.3, "Mildly hawkish", 0.8, mock_tone_components, [MockCitation("fomc_20230920_statement", "", 0, "")]),
    MockToneAnalysis(0.2, "Neutral leaning hawkish", 0.75, mock_tone_components, [MockCitation("fomc_20231101_statement", "", 0, "")]),
]

# Mock functions for source.py
def get_fomc_meeting_dates(start, end):
    return mock_meeting_dates

def check_document_availability(meeting_date):
    return mock_docs_available

def get_pending_review_count():
    global mock_session_manager
    return sum(1 for req in mock_session_manager.get_all_requests() if req.get("status") == "pending_review")

def load_memo(memo_id):
    if memo_id == mock_memo_id:
        return mock_fomc_memo
    return None

def update_memo_status(memo_id, status, review_action):
    if memo_id == mock_fomc_memo.memo_id:
        mock_fomc_memo.status = status
        mock_fomc_memo.review_history.append(review_action)
        # Also ensure the session_manager's request status is updated
        for req_id, req_data in mock_session_manager.requests.items():
            if req_data.get("memo_id") == memo_id:
                mock_session_manager.requests[req_id]["status"] = status
                break
    return True

# Mock async generator for workflow
async def mock_workflow_astream_generator(initial_state, config):
    yield {"__node__": "step_1"}
    yield {"__node__": "step_2"}
    final_event = {"draft_memo": mock_fomc_memo.__dict__}
    yield final_event

class MockWorkflow:
    def astream(self, initial_state, config):
        return mock_workflow_astream_generator(initial_state, config)

def create_fomc_workflow():
    return MockWorkflow()

def get_pending_reviews():
    global mock_session_manager
    pending = []
    for req in mock_session_manager.get_all_requests():
        if req.get("status") == "pending_review":
            pending.append({
                "request_id": req["request_id"],
                "meeting_date": req["meeting_date"],
                "status": req["status"],
                "memo_id": req.get("memo_id")
            })
    return pending

def render_tone_trajectory(historical_scores, current_tone, meeting_dates):
    pass

def render_citation_network(citations, themes):
    pass

# Global mock session manager
mock_session_manager = MockSessionStateManager()

@pytest.fixture(autouse=True)
def patch_source_module(monkeypatch):
    """Fixture to patch the 'source' module imports for AppTest."""
    mock_source = MagicMock()
    mock_source.get_fomc_meeting_dates = get_fomc_meeting_dates
    mock_source.check_document_availability = check_document_availability
    mock_source.get_pending_review_count = get_pending_review_count
    mock_source.SessionStateManager = MockSessionStateManager
    mock_source.create_fomc_workflow = create_fomc_workflow
    mock_source.AgentState = MockAgentState
    mock_source.FOMCMemo = MockFOMCMemo
    mock_source.get_pending_reviews = get_pending_reviews
    mock_source.load_memo = load_memo
    mock_source.update_memo_status = update_memo_status
    mock_source.mock_prior_tone_scores = mock_prior_tone_scores
    mock_source.render_tone_trajectory = render_tone_trajectory
    mock_source.render_citation_network = render_citation_network
    mock_source.ReviewAction = MockReviewAction

    monkeypatch.setitem(sys.modules, 'source', mock_source)

    # Reset global mock_session_manager and mock_fomc_memo for each test
    global mock_session_manager, mock_fomc_memo
    mock_session_manager = MockSessionStateManager()
    # Ensure a pending review exists for tests that expect it
    req_id = mock_session_manager.start_request(date(2023, 12, 13))
    mock_session_manager.requests[req_id]["status"] = "pending_review"
    mock_session_manager.requests[req_id]["memo_id"] = mock_memo_id
    
    # Reset mock_fomc_memo to its initial state
    mock_fomc_memo = MockFOMCMemo(
        memo_id=mock_memo_id,
        meeting_date=mock_meeting_dates[0],
        generated_at=datetime.now() - timedelta(hours=1),
        status="pending_review",
        version="1.0",
        executive_summary="The December FOMC meeting emphasized a cautious but hawkish stance...",
        market_implications="Markets reacted with slight volatility...",
        historical_context_summary="Compared to previous meetings, the tone was incrementally more hawkish.",
        themes=[mock_theme],
        tone_analysis=mock_tone_analysis,
        tone_delta=mock_tone_delta,
        surprises=[mock_surprise],
        validation_report=mock_validation_report,
        confidence_assessment=mock_confidence_assessment,
        review_history=[]
    )


# --- Test Functions ---
def test_initial_page_load_and_sidebar_navigation():
    at = AppTest.from_file("app.py").run()
    
    assert at.session_state["current_page"] == "home"
    assert "Welcome to the FOMC Insights Dashboard" in at.markdown[0].value

    at.button(key="nav_new_analysis_sidebar").click().run()
    assert at.session_state["current_page"] == "analysis"
    assert "New FOMC Analysis" in at.markdown[0].value

    at.button(key="nav_pending_reviews_sidebar").click().run()
    assert at.session_state["current_page"] == "review"
    assert "Review Pending Analysis" in at.markdown[0].value
    assert at.selectbox[0].options[0]["memo_id"] == mock_memo_id

    at.button(key="nav_analysis_history_sidebar").click().run()
    assert at.session_state["current_page"] == "history"
    assert "Analysis History" in at.markdown[0].value

    at.button(key="nav_settings_sidebar").click().run()
    assert at.session_state["current_page"] == "settings"
    assert "Settings and Configuration" in at.markdown[0].value

def test_home_page_card_navigation():
    at = AppTest.from_file("app.py").run()

    at.button(key="nav_analysis_card").click().run()
    assert at.session_state["current_page"] == "analysis"

    at = AppTest.from_file("app.py").run()
    at.button(key="nav_review_card").click().run()
    assert at.session_state["current_page"] == "review"

    at = AppTest.from_file("app.py").run()
    at.button(key="nav_history_card").click().run()
    assert at.session_state["current_page"] == "history"

def test_new_analysis_page_initial_render():
    at = AppTest.from_file("app.py").run()
    at.session_state["current_page"] = "analysis"
    at.run()

    assert "New FOMC Analysis" in at.markdown[0].value
    assert at.selectbox[0].options == mock_meeting_dates
    assert at.checkbox[0].value is True
    assert at.slider[0].value == 5
    assert at.checkbox[1].value is True
    assert at.slider[1].value == 0.7

def test_new_analysis_page_options_change():
    at = AppTest.from_file("app.py").run()
    at.session_state["current_page"] = "analysis"
    at.run()

    at.checkbox[0].set_value(False).run()
    assert at.checkbox[0].value is False

    at.slider[0].set_value(7).run()
    assert at.slider[0].value == 7

def test_new_analysis_start_button_and_preview_display():
    at = AppTest.from_file("app.py").run()
    at.session_state["current_page"] = "analysis"
    at.run()

    # Intercept asyncio.create_task to prevent actual async execution in tests
    with patch("asyncio.create_task") as mock_create_task:
        # Mock the return value of create_task, as it's assigned to active_workflow_task
        mock_create_task.return_value = MagicMock()
        at.button(key="start_analysis_button").click().run()

        # Verify that an async task was conceptually created
        assert at.session_state["active_workflow_task"] is not None
        assert mock_create_task.called
        
        # Manually simulate the workflow's successful completion by setting session state
        at.session_state["latest_memo_data"] = mock_fomc_memo
        # Get the request_id created by start_request in the mock_session_manager
        current_req_id = list(mock_session_manager.requests.keys())[-1]
        at.session_state["current_request_id"] = current_req_id
        
        # Update the mock_session_manager to reflect memo_id and status
        mock_session_manager.requests[current_req_id]["status"] = "completed"
        mock_session_manager.requests[current_req_id]["memo_id"] = mock_memo_id

        # Update request_history, as the app's workflow would
        if "request_history" not in at.session_state or not isinstance(at.session_state["request_history"], list):
            at.session_state["request_history"] = []
        # Find and update the existing request or append if new
        found_in_history = False
        for i, req_item in enumerate(at.session_state["request_history"]):
            if req_item.get("request_id") == current_req_id:
                at.session_state["request_history"][i].update({
                    "status": "completed",
                    "memo_id": mock_memo_id
                })
                found_in_history = True
                break
        if not found_in_history:
             at.session_state["request_history"].append({
                "request_id": current_req_id,
                "meeting_date": mock_fomc_memo.meeting_date,
                "status": "completed",
                "memo_id": mock_memo_memo.memo_id,
                "generated_at": datetime.now()
            })

        at.session_state["active_workflow_task"] = None # Simulate task completion
        at.run() # Rerun the app to reflect the completed state and display the preview

        # Verify the "Last Generated Memo Preview" is displayed
        assert "Last Generated Memo Preview" in at.markdown[7].value
        assert f"Meeting Date: {mock_fomc_memo.meeting_date.strftime('%B %d, %Y')}" in at.markdown[8].value
        assert "Status: <span class='status-pending_review'>Pending Review</span>" in at.markdown[9].value # Mock memo status
        assert "Overall Confidence: 90%" in at.markdown[10].value
        assert "Validation Status: ✅ Passed" in at.markdown[11].value
        assert at.button(key="go_to_review_btn").value == "📝 Go to Review for full details"

def test_review_page_no_pending_reviews():
    # Clear pending reviews for this test
    global mock_session_manager
    mock_session_manager = MockSessionStateManager() # Reset, so no pending reviews

    at = AppTest.from_file("app.py").run()
    at.session_state["current_page"] = "review"
    at.run()

    assert "No analyses pending review." in at.info[0].value
    assert not at.selectbox # No selectbox should be present

def test_review_page_with_pending_reviews_selection_and_memo_display():
    at = AppTest.from_file("app.py").run()
    at.session_state["current_page"] = "review"
    at.run()

    assert at.selectbox[0].options[0]["memo_id"] == mock_memo_id
    assert f"FOMC Memo: {mock_fomc_memo.meeting_date.strftime('%B %d, %Y')}" in at.markdown[1].value
    assert "Status: <span class='status-pending_review'>Pending Review</span>" in at.markdown[2].value
    assert at.tabs[0].label == "Summary"
    
    at.tabs[0].click().run()
    assert mock_fomc_memo.executive_summary in at.markdown[6].value

    at.tabs[1].click().run() # Themes tab
    assert at.expander[0].label == f"**{mock_theme.theme_name}** (Confidence: {mock_theme.confidence:.0%})"

def test_review_page_decision_workflow_approve_reject():
    at = AppTest.from_file("app.py").run()
    at.session_state["current_page"] = "review"
    at.run()

    # Test approve workflow
    at.button(key="approve_button").click().run()
    assert at.session_state["review_action"] == "approved"
    assert at.text_area[0].label == "Comments (required)"

    # Test submitting without comments
    at.button(key="submit_decision_button").click().run()
    assert "Comments are required to submit your decision." in at.error[0].value

    # Test submitting with comments
    at.text_area[0].set_value("Looks good, approved.").run()
    at.button(key="submit_decision_button").click().run()

    assert "review_action" not in at.session_state
    assert at.session_state["reviewer_comments"] == ""
    assert at.session_state["selected_review_memo_id"] is None
    assert at.session_state["current_request_id"] is None
    assert mock_fomc_memo.status == "approved"
    assert len(mock_fomc_memo.review_history) == 1
    assert mock_fomc_memo.review_history[0].action == "approved"

    # Reset for reject test
    at = AppTest.from_file("app.py").run()
    at.session_state["current_page"] = "review"
    at.run()
    
    # Test reject workflow
    at.button(key="reject_button").click().run()
    assert at.session_state["review_action"] == "rejected"
    at.text_area[0].set_value("Rejected due to factual errors.").run()
    at.button(key="submit_decision_button").click().run()

    assert mock_fomc_memo.status == "rejected"
    assert len(mock_fomc_memo.review_history) == 1
    assert mock_fomc_memo.review_history[0].action == "rejected"

def test_history_page_no_history():
    at = AppTest.from_file("app.py").run()
    at.session_state["current_page"] = "history"
    at.session_state["request_history"] = [] # Ensure empty history
    at.run()

    assert "No past analyses found in history." in at.info[0].value
    assert not at.selectbox

def test_history_page_with_history_selection():
    at = AppTest.from_file("app.py").run()
    at.session_state["current_page"] = "history"
    
    history_item = {
        "request_id": "hist_req_1",
        "meeting_date": date(2023, 10, 26),
        "status": "approved",
        "memo_id": mock_memo_id,
        "generated_at": datetime.now() - timedelta(days=5)
    }
    at.session_state["request_history"] = [history_item]
    
    # Ensure mock_fomc_memo is in an 'approved' state with a review history for display
    mock_fomc_memo.status = "approved"
    mock_fomc_memo.review_history = [
        MockReviewAction(datetime.now() - timedelta(days=1), "TestReviewer", "approved", "Good analysis", "1.0")
    ]
    at.run()

    assert at.selectbox[0].options[0] == history_item
    assert f"FOMC Memo: {mock_fomc_memo.meeting_date.strftime('%B %d, %Y')}" in at.markdown[1].value
    assert "Status: <span class='status-approved'>Approved</span>" in at.markdown[2].value
    
    assert at.tabs[0].label == "Tone Trajectory"
    assert at.tabs[2].label == "Full Memo View"

    at.tabs[2].click().run() # Full Memo View tab
    assert mock_fomc_memo.executive_summary in at.markdown[6].value

def test_settings_page():
    at = AppTest.from_file("app.py").run()
    at.session_state["current_page"] = "settings"
    at.run()

    assert "Settings and Configuration" in at.markdown[0].value
    assert "This page would allow configuration of API keys" in at.markdown[1].value
    assert "Settings functionality is conceptual and not fully implemented" in at.info[0].value
