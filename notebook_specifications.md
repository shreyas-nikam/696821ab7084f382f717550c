
# FOMC Research Insights: A Quantitative Analyst's Workflow

As a Quantitative Analyst at **Global Macro Alpha Fund (GMAF)**, your ability to rapidly and objectively interpret Federal Reserve communications is paramount. The Federal Open Market Committee (FOMC) meetings produce a deluge of information – statements, minutes, and press conference transcripts – that can move markets significantly. Manually sifting through these documents is time-consuming and prone to subjective bias, making it challenging to identify subtle policy shifts or unexpected signals.

This Jupyter Notebook simulates a real-world workflow where you, playing the role of **Alex Chen**, a Quantitative Analyst, leverage an AI-powered research agent to transform raw FOMC documents into structured, auditable investment research. The goal is to quantify the Fed's hawkish/dovish stance, detect crucial policy surprises, and extract key themes, providing GMAF with a competitive edge in macro trading decisions. We will focus on the **January 31, 2024 FOMC meeting** as our case study.

By the end of this notebook, you will have seen how to:
*   Set up a Python environment for advanced text analysis.
*   Ingest and parse complex financial documents programmatically.
*   Build a historical context using a vector database for comparative analysis.
*   Apply Large Language Models (LLMs) to extract key themes and quantify policy tone.
*   Identify "surprises" by detecting divergences from historical communication patterns.
*   Implement rigorous validation checks to ensure the reliability and auditability of AI-generated insights.
*   Synthesize all findings into a structured research memo, ready for strategic decision-making.

---

## 1. Setting Up the Research Environment

Alex begins by configuring his Python environment, installing the necessary libraries, and defining the structured data models (using Pydantic) that will underpin all subsequent analysis. This ensures that the outputs from different analytical steps are consistent, type-safe, and ready for an auditable workflow, a critical requirement in regulated financial environments.

### 1.1. Install Required Libraries

We'll install `openai` for LLM interaction, `langchain` for core components (though we'll use direct API calls for most LLM interactions for explicit control), `langgraph` for conceptual workflow state management (though explicit function calls will demonstrate the flow), `chromadb` for our vector store, `pydantic` for data modeling, `httpx` and `beautifulsoup4` for web scraping, and `structlog` for structured logging.

```python
!pip install openai~=1.10.0 langchain~=0.1.0 langgraph~=0.0.20 chromadb~=0.4.20 pydantic~=2.5.0 httpx~=0.26.0 beautifulsoup4~=4.12.0 structlog~=24.1.0 python-dotenv~=1.0.0
```

### 1.2. Import Dependencies and Configure API Key

Next, Alex imports the necessary modules and sets up his OpenAI API key. This key is vital for accessing the powerful LLMs that will drive much of the analysis.

```python
import os
import json
import httpx
import hashlib
import structlog
import asyncio
from datetime import date, datetime, timedelta
from typing import List, Optional, Literal, Dict, Any
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Load environment variables from a .env file (e.g., FOMC_OPENAI_API_KEY=sk-...)
load_dotenv()

# Configure OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("FOMC_OPENAI_API_KEY")
if not os.environ["OPENAI_API_KEY"]:
    raise ValueError("FOMC_OPENAI_API_KEY not set in environment variables. Please set it in a .env file or directly.")

import openai

# Initialize structured logger
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# Global LLM client
llm_client = openai.OpenAI()
```

### 1.3. Define Pydantic Data Models and Utility Functions

Alex defines the core data structures that will hold the ingested documents, extracted insights, and validation results. Pydantic ensures data integrity and helps enforce a consistent schema throughout the workflow. He also defines helper functions for URL construction, basic parsing, and citation validation that simplify later steps.

```python
# --- Pydantic Data Models from Specification ---

class FOMCDocumentType(str, Enum):
    """Types of FOMC documents we process."""
    STATEMENT = "statement"
    MINUTES = "minutes"
    PRESS_CONFERENCE = "press_conference"
    SPEECH = "speech"
    TESTIMONY = "testimony"
    SEP = "summary_of_economic_projections"

class DocumentSection(BaseModel):
    """A section within an FOMC document."""
    section_id: str
    heading: Optional[str] = None
    text: str
    start_position: int # Character offset in raw_text
    end_position: int
    speaker: Optional[str] = None # For press conferences
    paragraph_numbers: List[int]

class DocumentMetadata(BaseModel):
    """Metadata for audit trail."""
    ingestion_timestamp: datetime
    source_hash: str # SHA-256 of original file
    parser_version: str
    word_count: int
    language: str = "en"

class FOMCDocument(BaseModel):
    """Base model for all FOMC documents."""
    document_id: str = Field(..., description="Unique identifier")
    document_type: FOMCDocumentType
    meeting_date: date
    publication_date: date
    title: str
    source_url: str
    raw_text: str
    sections: List[DocumentSection]
    metadata: DocumentMetadata

    class Config:
        frozen = True # Immutable after creation
        arbitrary_types_allowed = True # Allow datetime.date

class Citation(BaseModel):
    """A citation linking a claim to source text."""
    citation_id: str
    document_id: str
    section_id: str
    paragraph_number: int
    quote: str # The exact quoted text
    quote_start: int # Character offset
    quote_end: int

class ThemeExtraction(BaseModel):
    """An extracted theme from FOMC materials."""
    theme_id: str
    theme_name: str
    description: str
    keywords: List[str]
    citations: List[Citation]
    confidence: float = Field(..., ge=0.0, le=1.0)

class ToneComponents(BaseModel):
    """Breakdown of tone score components."""
    inflation_stance: float
    employment_stance: float
    growth_outlook: float
    policy_bias: float
    uncertainty_level: float

class ToneScore(BaseModel):
    """Hawkish/Dovish tone assessment."""
    score: float = Field(..., ge=-1.0, le=1.0) # -1=dovish, +1=hawkish
    confidence: float = Field(..., ge=0.0, le=1.0)
    components: ToneComponents
    citations: List[Citation]
    explanation: str

class ToneDelta(BaseModel):
    """Change in tone between meetings."""
    current_meeting: date
    prior_meeting: date
    current_score: ToneScore
    prior_score: ToneScore
    delta: float
    delta_significance: Literal["minimal", "notable", "significant", "major"]
    key_drivers: List[str]
    citations: List[Citation]

class Surprise(BaseModel):
    """An unexpected element in FOMC communication."""
    surprise_id: str
    category: Literal[
        "policy_change",
        "language_shift",
        "forecast_revision",
        "dissent",
        "new_concern",
        "omission"
    ]
    description: str
    market_relevance: Literal["low", "medium", "high"]
    citations: List[Citation]
    confidence: float

class ValidationCheck(BaseModel):
    """Individual validation check result."""
    check_name: str
    passed: bool
    details: str
    severity: Literal["info", "warning", "error"]

class ValidationReport(BaseModel):
    """Results of automated validation."""
    validation_timestamp: datetime
    all_checks_passed: bool
    checks: List[ValidationCheck]

class ConfidenceAssessment(BaseModel):
    """Overall confidence in the memo."""
    overall_confidence: float
    evidence_strength: float
    citation_coverage: float # % of claims with citations
    model_agreement: float # If multiple models used
    flags: List[str] # Any concerns

class ReviewAction(BaseModel):
    """Human review action record."""
    timestamp: datetime
    reviewer_id: str
    action: Literal["approve", "revise", "reject", "escalate"]
    comments: str
    version_reviewed: int

class FOMCMemo(BaseModel):
    """The final research memo output."""
    memo_id: str
    meeting_date: date
    generated_at: datetime
    version: int
    status: Literal["draft", "pending_review", "approved", "rejected"]
    executive_summary: str
    themes: List[ThemeExtraction]
    tone_analysis: ToneScore
    tone_delta: Optional[ToneDelta] = None
    surprises: List[Surprise]
    historical_context_summary: str
    market_implications: str
    all_citations: List[Citation]
    confidence_assessment: ConfidenceAssessment
    validation_report: ValidationReport
    review_history: List[ReviewAction]

class AuditEntry(BaseModel):
    """Single audit trail entry."""
    entry_id: str
    request_id: str
    timestamp: datetime
    agent: str
    action: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    duration_ms: int
    reviewer_id: Optional[str] = None # For human decisions

# --- Utility Functions (simplified for notebook demo) ---

class DocumentNotFoundError(Exception):
    """Raised when a specific FOMC document for a date is not found."""
    pass

class FetchError(Exception):
    """Raised when document retrieval fails."""
    pass

def construct_fed_url(meeting_date: date, doc_type: str) -> str:
    """
    Constructs a mock URL for FOMC documents.
    In a real system, this would involve more complex logic to handle different
    document types and release schedules.
    """
    date_str = meeting_date.strftime("%Y%m%d")
    if doc_type == "statement":
        return f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{date_str}a.htm"
    elif doc_type == "minutes":
        # Minutes are typically released 3 weeks after the meeting
        release_date = meeting_date + timedelta(weeks=3)
        release_date_str = release_date.strftime("%Y%m%d")
        return f"https://www.federalreserve.gov/monetarypolicy/fomcminutes{release_date_str}.htm"
    elif doc_type == "press_conference":
        return f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{date_str}b.htm" # Mock for presser
    return "mock_url_not_found"

def parse_fed_statement_html(html_content: str) -> str:
    """Mock parser for FOMC statement HTML."""
    # In a real scenario, use BeautifulSoup4 to parse specific divs/sections
    return f"""Recent indicators suggest that economic activity has been expanding at a solid pace. Job gains have moderated since early last year but remain strong, and the unemployment rate has remained low. Inflation has eased over the past year but remains elevated. The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run.

The Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent. In considering any adjustments to the target range for the federal funds rate, the Committee will carefully assess incoming data, the evolving outlook, and the balance of risks.

The Committee's assessments of the appropriate stance of monetary policy and its economic projections will be summarized in the Summary of Economic Projections (SEP) to be released on {date(2024, 3, 20).strftime('%B %d, %Y')}.
"""

def extract_minutes_sections(html_content: str) -> List[DocumentSection]:
    """Mock parser for FOMC minutes sections."""
    # In reality, this would use BeautifulSoup4 to find headings like "Participants", "Developments in Financial Markets" etc.
    return [
        DocumentSection(section_id="overview", heading="Developments in Financial Markets and the Federal Funds Rate", text="""Participants noted that market participants generally expected the Committee to maintain the target range for the federal funds rate at this meeting. The market-implied path for the federal funds rate had flattened somewhat in recent weeks, suggesting that participants expected fewer rate cuts in the coming year.""", start_position=0, end_position=300, paragraph_numbers=[1]),
        DocumentSection(section_id="economic_outlook", heading="Staff Review of the Economic Situation", text="""The staff continued to project that real GDP would decelerate significantly in 2024, reflecting the lagged effects of tighter monetary policy and a softening in labor market conditions. Inflation was projected to decline further, approaching the Committee’s 2 percent objective over the next two years.""", start_position=301, end_position=600, paragraph_numbers=[1, 2]),
        DocumentSection(section_id="policy_discussion", heading="Committee Discussion", text="""In their discussion of monetary policy, members generally agreed that the current stance of monetary policy was restrictive and was contributing to downward pressure on inflation. Most members judged that further rate increases would likely be unnecessary, but that the Committee should remain prepared to respond appropriately to evolving economic conditions.""", start_position=601, end_position=900, paragraph_numbers=[1, 2])
    ]

def extract_speaker_sections(html_content: str) -> List[DocumentSection]:
    """Mock parser for press conference transcript with speaker diarization."""
    # In a real scenario, this involves regex or LLM to identify speakers and their turns.
    return [
        DocumentSection(section_id="powell_opening", speaker="Chair Powell", heading="Opening Remarks", text="""Good afternoon. My colleagues and I are strongly committed to bringing inflation down to our 2 percent goal. We have tightened the stance of monetary policy significantly, and we are seeing the effects of our actions on demand in the most interest-sensitive sectors of the economy.""", start_position=0, end_position=300, paragraph_numbers=[1]),
        DocumentSection(section_id="q_and_a_1", speaker="Reporter A", text="""Chair Powell, can you elaborate on the phrase 'further tightening may be appropriate'? Does this imply a stronger bias towards rate hikes than previously communicated?""", start_position=301, end_position=450, paragraph_numbers=[1]),
        DocumentSection(section_id="powell_response_1", speaker="Chair Powell", text="""We are committed to achieving a stance of monetary policy that is sufficiently restrictive to return inflation to 2 percent over time. The extent of future policy adjustments will depend on the totality of incoming data, the evolving outlook, and the balance of risks.""", start_position=451, end_position=700, paragraph_numbers=[1, 2])
    ]

def count_paragraphs(text: str) -> int:
    """Simple paragraph counter for mock data."""
    return len([p for p in text.split('\n') if p.strip()])

def validate_citation_exists(citation_dict: Dict, documents: List[FOMCDocument]) -> bool:
    """
    Checks if a citation (dict representation) refers to existing content in documents.
    Simplified for demo: just checks if doc_id and section_id exist and quote is in text.
    """
    try:
        citation = Citation(**citation_dict)
    except ValidationError:
        return False
        
    for doc in documents:
        if doc.document_id == citation.document_id:
            for section in doc.sections:
                if section.section_id == citation.section_id:
                    if citation.quote in section.text: # Basic check if quote is in text
                        return True
    return False

def extract_factual_claims(text: str) -> List[str]:
    """
    Extracts simple factual claims from text.
    Simplified for this notebook. In a real scenario, this would be an LLM call.
    """
    # Placeholder: return sentences as claims
    # This is a very basic split; a real LLM-based claim extraction would be more robust.
    return [s.strip() for s in text.split('.') if s.strip()]


# --- ChromaDB Integration (Mock/Simplified) ---
import chromadb
from chromadb.utils import embedding_functions

# Mock ChromaDB client setup
class MockChromaCollection:
    def __init__(self, name: str, embedding_function):
        self.name = name
        self._items = [] # Store added items
        self.embedding_function = embedding_function

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
        for i in range(len(ids)):
            # Generate dummy embedding as a list of floats
            dummy_embedding = [(j % self.embedding_function.dimension) / self.embedding_function.dimension for j in range(self.embedding_function.dimension)]
            self._items.append({
                "id": ids[i],
                "document": documents[i],
                "metadata": metadatas[i],
                "embedding": dummy_embedding # Add dummy embedding
            })
        logger.info("Mock ChromaDB add executed", collection_name=self.name, num_documents=len(ids))

    def query(self, query_embeddings: List[List[float]], n_results: int = 5, where: Optional[Dict] = None, include: List[str] = ["documents", "metadatas", "distances"]):
        # Mock query logic - in reality this queries ChromaDB
        # For demo, just return a fixed mock result for a known query
        # This mock data corresponds to the sample historical documents defined later.
        mock_results = {
            "ids": [["statement_2023_10_31_main_1", "minutes_2023_07_26_overview_1", "statement_2023_04_27_main_1"]],
            "embeddings": None, # Not returned by default for all queries
            "documents": [
                ["Inflation remains stubbornly high, and risks are skewed to the upside.",
                 "The Committee discussed the persistent elevated inflation and the need for a restrictive stance.",
                 "Economic growth has shown signs of slowing, and inflation pressures, while still elevated, have begun to moderate."
                ]
            ],
            "metadatas": [
                [
                    {"document_id": "statement_2023_10_31", "document_type": "statement", "meeting_date": "2023-10-31", "section_id": "main_1", "heading": "Statement", "speaker": None, "paragraph_numbers": [1], "word_count": 10},
                    {"document_id": "minutes_2023_07_26", "document_type": "minutes", "meeting_date": "2023-07-26", "section_id": "overview_1", "heading": "Developments in Financial Markets", "speaker": None, "paragraph_numbers": [1], "word_count": 12},
                    {"document_id": "statement_2023_04_27", "document_type": "statement", "meeting_date": "2023-04-27", "section_id": "main_1", "heading": "Statement", "speaker": None, "paragraph_numbers": [1], "word_count": 15}
                ]
            ],
            "distances": [[0.1, 0.15, 0.2]] # Mock distances
        }
        logger.info("Mock ChromaDB query executed", query_length=len(query_embeddings[0]), n_results=n_results, where=where)
        
        # Filter mock results based on 'where' clause (simplified)
        filtered_docs = []
        filtered_metadatas = []
        filtered_distances = []
        
        if where and "meeting_date" in where and "$lt" in where["meeting_date"]:
            query_date_str = where["meeting_date"]["$lt"]
            query_date = date.fromisoformat(query_date_str)
            
            for i in range(len(mock_results["documents"][0])):
                doc_meta_date_str = mock_results["metadatas"][0][i]["meeting_date"]
                doc_meta_date = date.fromisoformat(doc_meta_date_str)
                if doc_meta_date < query_date:
                    filtered_docs.append(mock_results["documents"][0][i])
                    filtered_metadatas.append(mock_results["metadatas"][0][i])
                    filtered_distances.append(mock_results["distances"][0][i])
            
            # Limit to n_results
            return {
                "ids": [mock_results["ids"][0][:len(filtered_docs)]], # Keep IDs for filtered documents only
                "documents": [filtered_docs[:n_results]],
                "metadatas": [filtered_metadatas[:n_results]],
                "distances": [filtered_distances[:n_results]]
            }

        return mock_results

class MockChromaClient:
    def __init__(self, persist_directory: str = "./fomc_memory"):
        self.persist_directory = persist_directory
        self._collections = {}

    def get_or_create_collection(self, name: str, metadata: Optional[Dict] = None, embedding_function=None):
        if name not in self._collections:
            logger.info("Creating ChromaDB collection", collection_name=name)
            self._collections[name] = MockChromaCollection(name=name, embedding_function=embedding_function)
        return self._collections[name]

# Dummy embedding function for ChromaDB without real OpenAI API calls.
class DummyOpenAIEmbeddingFunction:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.dimension = 1536 # OpenAI text-embedding-3-small dimension

    def __call__(self, texts: List[str]) -> List[List[float]]:
        # Generate dummy embeddings for demonstration purposes
        logger.info("Generating dummy embeddings", model=self.model_name, num_texts=len(texts))
        return [[(i % self.dimension) / self.dimension for i in range(self.dimension)] for _ in texts]

chroma_client = MockChromaClient(persist_directory="./fomc_memory")

async def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Retrieves an embedding for the given text."""
    # In a real scenario, this would call openai.embeddings.create
    logger.info("Getting dummy embedding", text_length=len(text), model=model)
    return [(i % 1536) / 1536 for i in range(1536)] # Dummy embedding of 1536 dimensions

```

### 1.4. (Conceptual) Tool Definitions

Alex understands that for an AI agent to be truly useful, it needs to interact with external systems and perform specific, well-defined tasks. These "tools" provide structured interfaces for capabilities like fetching documents, computing metrics, or searching databases. While we won't implement a full LangChain tool agent here, defining these tools conceptually is key to the agentic design.

```python
# --- Conceptual Tool Definitions from Specification ---
# Note: For this notebook, we directly call the functions that these tools represent,
# rather than orchestrating them via a LangChain AgentExecutor, to keep the focus
# on the data flow and analytical steps.

class ToolParameter(BaseModel):
    type: str
    required: bool = True
    default: Optional[Any] = None

class Tool(BaseModel):
    """Base class for all agent tools, with explicit schemas and audit requirements."""
    name: str
    description: str
    parameters: Dict[str, ToolParameter]
    returns: str
    requires_approval: bool = False
    audit_level: Literal["none", "input", "full"] = "full"

FOMC_TOOLS = [
    Tool(
        name="fetch_fomc_statement",
        description="Retrieve official FOMC statement for a given meeting date",
        parameters={"meeting_date": ToolParameter(type="date", required=True)},
        returns="FOMCDocument object with full text and metadata",
        audit_level="full"
    ),
    Tool(
        name="compute_tone_score",
        description="Calculate hawkish/dovish tone score for text passage",
        parameters={
            "documents": ToolParameter(type="List[FOMCDocument]", required=True),
            "prior_scores": ToolParameter(type="List[ToneScore]", required=False),
        },
        returns="ToneScore with value, confidence, and explanation",
        audit_level="full"
    ),
    Tool(
        name="search_historical_context",
        description="Retrieve similar passages from historical FOMC materials",
        parameters={
            "query": ToolParameter(type="string", required=True),
            "n_results": ToolParameter(type="int", default=5),
            "date_range": ToolParameter(type="date_range", required=False),
        },
        returns="List of HistoricalPassage with similarity scores",
        audit_level="full"
    ),
    # ... other tools like extract_themes, detect_surprises, validate_citation, check_hallucination
]
```

---

## 2. Ingesting and Parsing FOMC Documents

To begin his analysis of the January 31, 2024 FOMC meeting, Alex needs to gather the raw materials: the official FOMC Statement, the Meeting Minutes (which are released later), and the Press Conference transcript. This step is crucial for establishing the data's provenance and ensuring the analysis is based on the authoritative source text.

### 2.1. Fetching FOMC Statement, Minutes, and Press Conference Transcript

Alex uses specialized `fetch` functions to retrieve the documents from the Federal Reserve's website. These functions emulate web scraping (using `httpx` and `BeautifulSoup4` internally for real systems) and return structured `FOMCDocument` objects.

```python
async def fetch_fomc_statement(meeting_date: date) -> FOMCDocument:
    """
    Retrieves the official FOMC statement for a given meeting date.
    Assumes successful retrieval and parsing for this demo.
    """
    url = construct_fed_url(meeting_date, "statement")
    # In a real scenario, use httpx to fetch and BeautifulSoup4 to parse.
    # For this demo, we use a mock content.
    raw_text = parse_fed_statement_html("mock_html_content_statement")
    sections = [
        DocumentSection(
            section_id="main",
            heading="FOMC Statement",
            text=raw_text,
            start_position=0,
            end_position=len(raw_text),
            speaker=None,
            paragraph_numbers=list(range(1, count_paragraphs(raw_text) + 1))
        )
    ]
    doc_id = f"statement_{meeting_date.isoformat()}"
    logger.info("Fetched FOMC Statement", document_id=doc_id, meeting_date=meeting_date)
    return FOMCDocument(
        document_id=doc_id,
        document_type=FOMCDocumentType.STATEMENT,
        meeting_date=meeting_date,
        publication_date=meeting_date,
        title=f"FOMC Statement - {meeting_date.strftime('%B %d, %Y')}",
        source_url=url,
        raw_text=raw_text,
        sections=sections,
        metadata=DocumentMetadata(
            ingestion_timestamp=datetime.now(),
            source_hash=hashlib.sha256(raw_text.encode()).hexdigest(),
            parser_version="1.0.0",
            word_count=len(raw_text.split()),
        )
    )

async def fetch_fomc_minutes(meeting_date: date) -> FOMCDocument:
    """
    Retrieves the FOMC meeting minutes for a given meeting date.
    Minutes are typically released ~3 weeks after the meeting.
    Assumes successful retrieval and parsing for this demo.
    """
    url = construct_fed_url(meeting_date, "minutes")
    # For this demo, we use a mock content.
    raw_text = """Participants noted that market participants generally expected the Committee to maintain the target range for the federal funds rate at this meeting. The market-implied path for the federal funds rate had flattened somewhat in recent weeks, suggesting that participants expected fewer rate cuts in the coming year.

The staff continued to project that real GDP would decelerate significantly in 2024, reflecting the lagged effects of tighter monetary policy and a softening in labor market conditions. Inflation was projected to decline further, approaching the Committee’s 2 percent objective over the next two years.

In their discussion of monetary policy, members generally agreed that the current stance of monetary policy was restrictive and was contributing to downward pressure on inflation. Most members judged that further rate increases would likely be unnecessary, but that the Committee should remain prepared to respond appropriately to evolving economic conditions."""
    sections = extract_minutes_sections("mock_html_content_minutes") # Mock parsing to get structured sections
    doc_id = f"minutes_{meeting_date.isoformat()}"
    logger.info("Fetched FOMC Minutes", document_id=doc_id, meeting_date=meeting_date)
    return FOMCDocument(
        document_id=doc_id,
        document_type=FOMCDocumentType.MINUTES,
        meeting_date=meeting_date,
        publication_date=meeting_date + timedelta(weeks=3), # Mock publication date
        title=f"FOMC Minutes - {meeting_date.strftime('%B %d, %Y')}",
        source_url=url,
        raw_text=raw_text,
        sections=sections,
        metadata=DocumentMetadata(
            ingestion_timestamp=datetime.now(),
            source_hash=hashlib.sha256(raw_text.encode()).hexdigest(),
            parser_version="1.0.0",
            word_count=len(raw_text.split()),
        )
    )

async def fetch_press_conference(meeting_date: date) -> FOMCDocument:
    """
    Retrieves the press conference transcript for a given meeting.
    Not all meetings have press conferences. Assumes successful retrieval and parsing for this demo.
    """
    url = construct_fed_url(meeting_date, "press_conference")
    # For this demo, we use a mock content.
    raw_text = """Chair Powell: Good afternoon. My colleagues and I are strongly committed to bringing inflation down to our 2 percent goal. We have tightened the stance of monetary policy significantly, and we are seeing the effects of our actions on demand in the most interest-sensitive sectors of the economy.

Reporter A: Chair Powell, can you elaborate on the phrase 'further tightening may be appropriate'? Does this imply a stronger bias towards rate hikes than previously communicated?

Chair Powell: We are committed to achieving a stance of monetary policy that is sufficiently restrictive to return inflation to 2 percent over time. The extent of future policy adjustments will depend on the totality of incoming data, the evolving outlook, and the balance of risks."""
    sections = extract_speaker_sections("mock_html_content_presser") # Mock parsing for speaker diarization
    doc_id = f"presser_{meeting_date.isoformat()}"
    logger.info("Fetched FOMC Press Conference", document_id=doc_id, meeting_date=meeting_date)
    return FOMCDocument(
        document_id=doc_id,
        document_type=FOMCDocumentType.PRESS_CONFERENCE,
        meeting_date=meeting_date,
        publication_date=meeting_date, # Press conference released same day
        title=f"Press Conference - {meeting_date.strftime('%B %d, %Y')}",
        source_url=url,
        raw_text=raw_text,
        sections=sections,
        metadata=DocumentMetadata(
            ingestion_timestamp=datetime.now(),
            source_hash=hashlib.sha256(raw_text.encode()).hexdigest(),
            parser_version="1.0.0",
            word_count=len(raw_text.split()),
        )
    )

# Execution for January 31, 2024 FOMC meeting
fomc_date = date(2024, 1, 31)
logger.info("Initiating document ingestion for FOMC meeting", meeting_date=fomc_date)

# Simulate fetching documents
fomc_statement = await fetch_fomc_statement(fomc_date)
fomc_minutes = await fetch_fomc_minutes(fomc_date) # Note: In real life, minutes are later. For demo, we assume they are available.
fomc_press_conference = await fetch_press_conference(fomc_date)

current_fomc_documents = [fomc_statement, fomc_minutes, fomc_press_conference]
```

### 2.2. Explanation of Document Ingestion Output

Alex now has a list of `FOMCDocument` objects, each representing a key piece of communication from the January 2024 FOMC meeting. Each document is not just raw text, but a structured object containing metadata (like publication date and source hash) and segmented sections, including speaker attribution for the press conference.

This structured ingestion is crucial because:
*   **Auditability:** Every piece of information has a clear source URL, ingestion timestamp, and content hash, forming an initial audit trail.
*   **Granularity:** Breaking documents into sections allows for more targeted analysis and precise citation.
*   **Preparation for LLMs:** Structured data is easier for LLMs to process consistently and extract information with higher accuracy.

The next step for Alex is to build a foundation for comparative analysis by establishing a historical context in a vector store.

---

## 3. Building Historical Context with a Vector Store

To properly assess the nuance of the current FOMC communication, Alex needs to compare it against historical patterns. A vector store provides a powerful "memory" for the agent, allowing Alex to retrieve semantically similar passages from past meetings. This helps distinguish routine language from significant policy shifts.

### 3.1. Initializing ChromaDB and Indexing Documents

Alex sets up an in-memory ChromaDB instance. He then indexes the fetched current FOMC documents and several mock historical documents, turning their sections into numerical vector embeddings. These embeddings allow for efficient semantic search.

```python
# --- ChromaDB Initialization and Indexing ---

async def index_document(document: FOMCDocument, collection) -> None: # collection type is MockChromaCollection
    """Indexes an FOMC document into the vector store, chunked by section."""
    ids = []
    documents_to_add = []
    metadatas = []

    for section in document.sections:
        section_id_full = f"{document.document_id}_{section.section_id}"
        ids.append(section_id_full)
        documents_to_add.append(section.text)
        metadatas.append({
            "document_id": document.document_id,
            "document_type": document.document_type.value,
            "meeting_date": document.meeting_date.isoformat(),
            "section_id": section.section_id,
            "heading": section.heading or "",
            "speaker": section.speaker or "",
            "paragraph_numbers": json.dumps(section.paragraph_numbers),
            "word_count": len(section.text.split()),
        })

    if ids:
        collection.add(
            ids=ids,
            documents=documents_to_add,
            metadatas=metadatas
        )
        logger.info("Document indexed", document_id=document.document_id, sections_indexed=len(ids))


# Initialize the ChromaDB client and collections
# Using DummyOpenAIEmbeddingFunction for mock client
openai_ef = DummyOpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"])
fomc_documents_collection = chroma_client.get_or_create_collection(
    name="fomc_documents",
    metadata={
        "description": "Embedded FOMC document sections for retrieval",
        "embedding_model": "text-embedding-3-small",
        "distance_metric": "cosine",
    },
    embedding_function=openai_ef
)

historical_memos_collection = chroma_client.get_or_create_collection(
    name="historical_memos", # For storing summary memos, not individual sections
    metadata={
        "description": "Past research memos for context",
        "embedding_model": "text-embedding-3-small",
    },
    embedding_function=openai_ef
)

# Index current FOMC documents
logger.info("Indexing current FOMC documents into ChromaDB.")
for doc in current_fomc_documents:
    await index_document(doc, fomc_documents_collection)

# Create and index mock historical FOMC documents (simplified for demo)
mock_historical_documents = [
    FOMCDocument(
        document_id="statement_2023_10_31",
        document_type=FOMCDocumentType.STATEMENT,
        meeting_date=date(2023, 10, 31),
        publication_date=date(2023, 10, 31),
        title="FOMC Statement - October 31, 2023",
        source_url="https://federalreserve.gov/mock_20231031a.htm",
        raw_text="""Inflation remains stubbornly high, and risks are skewed to the upside. The Committee maintains its commitment to price stability. Further tightening actions may be warranted if inflation does not continue to move sustainably towards 2 percent. Economic activity expanded at a solid pace.""",
        sections=[
            DocumentSection(section_id="main_1", heading="Statement", text="Inflation remains stubbornly high, and risks are skewed to the upside.", start_position=0, end_position=70, paragraph_numbers=[1]),
            DocumentSection(section_id="main_2", heading="Statement", text="The Committee maintains its commitment to price stability. Further tightening actions may be warranted if inflation does not continue to move sustainably towards 2 percent.", start_position=71, end_position=250, paragraph_numbers=[2]),
            DocumentSection(section_id="main_3", heading="Statement", text="Economic activity expanded at a solid pace.", start_position=251, end_position=290, paragraph_numbers=[3])
        ],
        metadata=DocumentMetadata(
            ingestion_timestamp=datetime.now() - timedelta(days=90),
            source_hash="hist_hash_oct23", parser_version="1.0.0", word_count=50
        )
    ),
    FOMCDocument(
        document_id="minutes_2023_07_26",
        document_type=FOMCDocumentType.MINUTES,
        meeting_date=date(2023, 7, 26),
        publication_date=date(2023, 8, 16),
        title="FOMC Minutes - July 26, 2023",
        source_url="https://federalreserve.gov/mock_20230726.htm",
        raw_text="""The Committee discussed the persistent elevated inflation and the need for a restrictive stance. Some participants noted the importance of monitoring financial conditions. Labor market conditions remained tight.""",
        sections=[
            DocumentSection(section_id="overview_1", heading="Developments in Financial Markets", text="The Committee discussed the persistent elevated inflation and the need for a restrictive stance.", start_position=0, end_position=110, paragraph_numbers=[1]),
            DocumentSection(section_id="overview_2", heading="Developments in Financial Markets", text="Some participants noted the importance of monitoring financial conditions. Labor market conditions remained tight.", start_position=111, end_position=250, paragraph_numbers=[2])
        ],
        metadata=DocumentMetadata(
            ingestion_timestamp=datetime.now() - timedelta(days=180),
            source_hash="hist_hash_jul23", parser_version="1.0.0", word_count=40
        )
    ),
    FOMCDocument(
        document_id="statement_2023_04_27",
        document_type=FOMCDocumentType.STATEMENT,
        meeting_date=date(2023, 4, 27),
        publication_date=date(2023, 4, 27),
        title="FOMC Statement - April 27, 2023",
        source_url="https://federalreserve.gov/mock_20230427a.htm",
        raw_text="""Economic growth has shown signs of slowing, and inflation pressures, while still elevated, have begun to moderate. The Committee is prepared to adjust policy as appropriate.""",
        sections=[
            DocumentSection(section_id="main_1", heading="Statement", text="Economic growth has shown signs of slowing, and inflation pressures, while still elevated, have begun to moderate.", start_position=0, end_position=150, paragraph_numbers=[1]),
            DocumentSection(section_id="main_2", heading="Statement", text="The Committee is prepared to adjust policy as appropriate.", start_position=151, end_position=200, paragraph_numbers=[2])
        ],
        metadata=DocumentMetadata(
            ingestion_timestamp=datetime.now() - timedelta(days=270),
            source_hash="hist_hash_apr23", parser_version="1.0.0", word_count=35
        )
    )
]

logger.info("Indexing mock historical FOMC documents into ChromaDB.")
for doc in mock_historical_documents:
    await index_document(doc, fomc_documents_collection)

all_historical_documents = mock_historical_documents + current_fomc_documents # For full context in later steps
```

### 3.2. Retrieving Historical Context for Specific Queries

Now Alex can query the vector store for passages semantically related to a topic of interest, filtering by date to ensure relevance. This allows him to understand how the Fed has discussed certain themes in the past.

```python
async def search_historical_context(
    query: str,
    meeting_date: date,
    n_results: int = 5,
    lookback_meetings: int = 8 # Not directly used in mock query but conceptually for filtering
) -> List[Dict]:
    """
    Searches historical FOMC materials for relevant context using semantic search.
    For demo, it uses a mock ChromaDB client.
    """
    embedding = await get_embedding(query) # Get dummy embedding for the query

    # In a real ChromaDB, this would query the actual fomc_documents_collection
    # and filter by meeting_date using the 'where' clause.
    
    results = chroma_client.query(
        collection_name="fomc_documents",
        query_embeddings=[embedding],
        n_results=n_results,
        where={"meeting_date": {"$lt": meeting_date.isoformat()}} # Filter to exclude current meeting
    )

    historical_passages = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc_text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            historical_passages.append({
                "text": doc_text,
                "meeting_date": meta["meeting_date"],
                "document_type": meta["document_type"],
                "similarity": 1 - results["distances"][0][i], # Convert distance to similarity
                "section_id": meta["section_id"],
                "document_id": meta["document_id"]
            })
    
    logger.info("Searched historical context", query=query, num_results=len(historical_passages))
    return historical_passages

# Example: Search for historical context on "inflation outlook"
query_inflation = "current inflation outlook and its implications"
historical_inflation_context = await search_historical_context(
    query=query_inflation,
    meeting_date=fomc_date,
    n_results=3
)

print("--- Retrieved Historical Inflation Context ---")
for i, passage in enumerate(historical_inflation_context):
    print(f"Passage {i+1} (Meeting: {passage['meeting_date']}, Similarity: {passage['similarity']:.2f}):")
    print(f"  '{passage['text'][:100]}...'")
    print("-" * 10)
```

### 3.3. Explanation of Historical Context Output

By querying the vector store, Alex has retrieved relevant historical passages concerning "inflation outlook". Each passage is accompanied by its source document ID, meeting date, and a similarity score.

This is a powerful application of **Memory** and **Tool Use (semantic search)**:
*   **Objectivity:** It prevents analysis from being conducted in a vacuum, grounding the understanding of current language in historical data.
*   **Trend Identification:** Alex can quickly see if phrases like "inflation remains elevated" are new or a continuation of past communication, helping to identify true shifts.
*   **Efficiency:** Instead of manually reviewing years of documents, the agent can instantly surface the most relevant comparisons.

This "memory" will now be used to inform the LLM's analysis of themes and tone, making the insights more nuanced and precise.

---

## 4. Analyzing the Fed's Stance: Themes and Tone

Now, with the latest documents ingested and historical context at hand, Alex instructs the AI agent to extract the primary themes discussed in the FOMC meeting and quantify the overall hawkish/dovish tone. This provides the fundamental insights for understanding the Fed's current policy posture.

### 4.1. Extracting Key Themes with Citations

Alex needs a concise summary of the key discussion points from the meeting, each backed by direct quotes from the source documents. This ensures that the extracted themes are factual and auditable.

```python
# --- LLM Prompts ---

THEME_EXTRACTION_PROMPT = """
You are an expert financial analyst. Your task is to extract the most important themes
from the provided FOMC documents (Statement, Minutes, Press Conference).
For each theme, provide:
1. A concise `theme_name`.
2. A brief `description` of the theme.
3. A list of `keywords` associated with the theme.
4. AT LEAST ONE `citation` directly linking to the source text that supports the theme.
   Citations must be in the exact format:
   {"document_id": "...", "section_id": "...", "paragraph_number": ..., "quote": "...", "quote_start": ..., "quote_end": ...}
   Ensure the `quote` is an exact substring from the source document, and `quote_start`/`quote_end` are its character offsets.

Combine the text from all documents for analysis, noting their types (e.g., [STATEMENT], [MINUTES], [PRESS_CONFERENCE]).
Focus on macroeconomic conditions, monetary policy, and outlook.

Respond in JSON format with a key "themes" which is a list of ThemeExtraction objects.
Example:
{
  "themes": [
    {
      "theme_name": "Inflation Moderation",
      "description": "Evidence suggests inflation has eased but remains above target.",
      "keywords": ["inflation", "moderation", "easing", "target"],
      "citations": [
        {"citation_id": "c1", "document_id": "statement_2024_01_31", "section_id": "main", "paragraph_number": 1, "quote": "Inflation has eased over the past year but remains elevated.", "quote_start": 100, "quote_end": 156}
      ],
      "confidence": 0.95
    }
  ]
}
"""

async def extract_themes(
    documents: List[FOMCDocument],
    n_themes: int = 5
) -> List[ThemeExtraction]:
    """
    Extracts major themes from FOMC documents using an LLM, ensuring citations.
    """
    combined_text = "\n\n".join([
        f"[{doc.document_type.value.upper()}]\n{doc.raw_text}"
        for doc in documents
    ])

    response = await llm_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": THEME_EXTRACTION_PROMPT},
            {"role": "user", "content": combined_text}
        ],
        response_format={"type": "json_object"},
        temperature=0.3 # Moderate temperature for balanced creativity and consistency
    )
    
    raw_themes_output = json.loads(response.choices[0].message.content)
    
    themes = []
    for theme_data in raw_themes_output.get("themes", [])[:n_themes]:
        validated_citations = []
        for cite_dict in theme_data.get("citations", []):
            if validate_citation_exists(cite_dict, documents):
                try:
                    validated_citations.append(Citation(**cite_dict))
                except ValidationError as e:
                    logger.warning("Invalid citation format in LLM output", citation=cite_dict, error=str(e))
            else:
                logger.warning("LLM generated unverified citation", citation=cite_dict)
        
        # Calculate confidence based on citation validity and quantity
        llm_confidence = theme_data.get("confidence", 0.7) # Use LLM's confidence if provided
        citation_coverage_confidence = len(validated_citations) / max(len(theme_data.get("citations", [])), 1)
        final_confidence = (llm_confidence + citation_coverage_confidence) / 2 # Simple average

        try:
            themes.append(ThemeExtraction(
                theme_id=f"theme_{uuid4().hex[:8]}",
                theme_name=theme_data["theme_name"],
                description=theme_data["description"],
                keywords=theme_data.get("keywords", []),
                citations=validated_citations,
                confidence=min(max(final_confidence, 0.0), 1.0) # Ensure between 0 and 1
            ))
        except ValidationError as e:
            logger.error("Failed to parse ThemeExtraction from LLM output", data=theme_data, error=str(e))
    
    logger.info("Extracted themes", num_themes=len(themes))
    return themes

# Execution: Extract themes for the current FOMC documents
num_themes_to_extract = 3
extracted_themes = await extract_themes(current_fomc_documents, n_themes=num_themes_to_extract)

print("--- Extracted Themes ---")
for theme in extracted_themes:
    print(f"Theme: {theme.theme_name} (Confidence: {theme.confidence:.2f})")
    print(f"Description: {theme.description}")
    print(f"Keywords: {', '.join(theme.keywords)}")
    if theme.citations:
        print(f"Example Citation: '{theme.citations[0].quote}' from {theme.citations[0].document_id}")
    print("-" * 20)
```

### 4.2. Quantifying Hawkish/Dovish Tone

Alex needs a quantitative score for the Fed's tone, broken down into key economic components (inflation, employment, growth, policy bias, uncertainty). This multi-dimensional assessment, calibrated against prior meetings, offers a more objective measure than a subjective reading.

The **Tone Score** ranges from -1.0 (extremely dovish) to +1.0 (extremely hawkish).

$$ S_{tone} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot s_i $$

Where $S_{tone}$ is the overall tone score, $N$ is the number of components, $w_i$ is the weighting of component $i$ (assumed equal for simplicity in LLM-based scoring), and $s_i$ is the sentiment score for component $i$.

```python
TONE_ANALYSIS_PROMPT = """
You are an expert financial analyst tasked with assessing the hawkish/dovish tone of FOMC communications.
Analyze the provided documents (Statement, Minutes, Press Conference) and assign a tone score from -1.0 (extremely dovish) to +1.0 (extremely hawkish).
Also provide a breakdown for key components, a confidence score, and citations.

Consider the following interpretation scale:
-1.0: Extremely Dovish (e.g., "Support recovery", "Accommodative stance")
-0.5: Moderately Dovish (e.g., "Slowing warranted", "Monitor conditions")
0.0: Neutral/Balanced (e.g., "Risks broadly balanced", "Data dependent")
+0.5: Moderately Hawkish (e.g., "Vigilant on inflation", "Elevated uncertainty")
+1.0: Extremely Hawkish (e.g., "Inflation unacceptably high", "Further tightening")

Historical tone scores for calibration are provided. Use them to ensure consistency in your scoring.

For each component (inflation_stance, employment_stance, growth_outlook, policy_bias, uncertainty_level), assign a score from -1.0 to +1.0 based on the text.
The overall `score` should be a weighted average of these components, reflecting the dominant sentiment.
The `confidence` should reflect how clearly and consistently the tone is conveyed (0.0 to 1.0).

ALL claims made in your `explanation` MUST include a citation in the specified format.

Respond in JSON format with keys: `overall_score`, `confidence`, `components` (with inflation_stance, employment_stance, growth_outlook, policy_bias, uncertainty_level), `citations`, and `explanation`.
Example:
{
  "overall_score": 0.3,
  "confidence": 0.85,
  "components": {
    "inflation_stance": 0.6,
    "employment_stance": 0.1,
    "growth_outlook": 0.2,
    "policy_bias": 0.4,
    "uncertainty_level": -0.1
  },
  "citations": [
    {"citation_id": "c2", "document_id": "statement_2024_01_31", "section_id": "main", "paragraph_number": 1, "quote": "Inflation has eased over the past year but remains elevated.", "quote_start": 100, "quote_end": 156}
  ],
  "explanation": "The Committee noted that while inflation has eased, it remains elevated [CITE: doc_id=statement_2024_01_31, section_id=main, para=1, quote='Inflation has eased over the past year but remains elevated.']. Employment gains have moderated, suggesting a more balanced labor market."
}

Historical tone scores for calibration:
{historical_context}

Documents to analyze:
{combined_text}
"""

async def compute_tone_score(
    documents: List[FOMCDocument],
    prior_scores: Optional[List[ToneScore]] = None
) -> ToneScore:
    """Computes hawkish/dovish tone score for FOMC documents using an LLM."""
    historical_context = ""
    if prior_scores:
        # Provide the last 3 historical scores for calibration
        historical_context = json.dumps([s.model_dump() for s in prior_scores[-3:]], indent=2)

    combined_text = "\n\n".join([
        f"[{doc.document_type.value.upper()}]\n{doc.raw_text}"
        for doc in documents
    ])

    messages = [
        {"role": "system", "content": TONE_ANALYSIS_PROMPT.format(
            historical_context=historical_context,
            combined_text=combined_text
        )}
    ]
    
    response = await llm_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.3
    )

    result = json.loads(response.choices[0].message.content)

    validated_citations = []
    for cite_dict in result.get("citations", []):
        if validate_citation_exists(cite_dict, documents):
            try:
                validated_citations.append(Citation(**cite_dict))
            except ValidationError as e:
                logger.warning("Invalid citation format in LLM tone score output", citation=cite_dict, error=str(e))
        else:
            logger.warning("LLM generated unverified citation in tone score", citation=cite_dict)

    try:
        tone_score_obj = ToneScore(
            score=result["overall_score"],
            confidence=result["confidence"],
            components=ToneComponents(**result["components"]),
            citations=validated_citations,
            explanation=result["explanation"]
        )
    except ValidationError as e:
        logger.error("Failed to parse ToneScore from LLM output", data=result, error=str(e))
        raise # Re-raise to indicate a critical parsing failure

    logger.info("Computed tone score", score=tone_score_obj.score, confidence=tone_score_obj.confidence)
    return tone_score_obj

# Mock historical tone scores for calibration
mock_prior_tone_scores = [
    ToneScore(
        score=0.6, confidence=0.8,
        components=ToneComponents(inflation_stance=0.7, employment_stance=0.5, growth_outlook=0.4, policy_bias=0.8, uncertainty_level=0.2),
        citations=[Citation(citation_id="mock_oct_cite", document_id="statement_2023_10_31", section_id="main_2", paragraph_number=2, quote="Further tightening actions may be warranted", quote_start=120, quote_end=160)],
        explanation="October 2023: Strongly hawkish due to persistent inflation concerns."
    ),
    ToneScore(
        score=0.3, confidence=0.75,
        components=ToneComponents(inflation_stance=0.5, employment_stance=0.3, growth_outlook=0.1, policy_bias=0.4, uncertainty_level=0.3),
        citations=[Citation(citation_id="mock_jul_cite", document_id="minutes_2023_07_26", section_id="overview_1", paragraph_number=1, quote="persistent elevated inflation", quote_start=30, quote_end=58)],
        explanation="July 2023: Moderately hawkish, noting elevated inflation but signs of softening."
    ),
    ToneScore(
        score=-0.1, confidence=0.85,
        components=ToneComponents(inflation_stance=0.2, employment_stance=-0.1, growth_outlook=-0.3, policy_bias=0.0, uncertainty_level=0.5),
        citations=[Citation(citation_id="mock_apr_cite", document_id="statement_2023_04_27", section_id="main_1", paragraph_number=1, quote="growth has shown signs of slowing", quote_start=15, quote_end=45)],
        explanation="April 2023: Neutral-to-slightly dovish, acknowledging slowing growth and moderating inflation."
    )
]

# Execution: Compute tone score for the current FOMC documents
current_tone_score = await compute_tone_score(current_fomc_documents, mock_prior_tone_scores)

print("\n--- Current Meeting Tone Analysis ---")
print(f"Overall Tone Score: {current_tone_score.score:+.2f} (Confidence: {current_tone_score.confidence:.0%})")
print("Component Breakdown:")
print(f"  Inflation Stance: {current_tone_score.components.inflation_stance:+.2f}")
print(f"  Employment Stance: {current_tone_score.components.employment_stance:+.2f}")
print(f"  Growth Outlook: {current_tone_score.components.growth_outlook:+.2f}")
print(f"  Policy Bias: {current_tone_score.components.policy_bias:+.2f}")
print(f"  Uncertainty Level: {current_tone_score.components.uncertainty_level:+.2f}")
print(f"Explanation: {current_tone_score.explanation}")
if current_tone_score.citations:
    print(f"Example Citation: '{current_tone_score.citations[0].quote}' from {current_tone_score.citations[0].document_id}")
```

### 4.3. Explanation of Themes and Tone Analysis Output

Alex now has a clear, structured view of the January 2024 FOMC meeting:
*   **Key Themes:** A list of `ThemeExtraction` objects, each highlighting a significant area of discussion (e.g., "Inflation Moderation," "Labor Market Conditions"). The embedded citations provide direct textual evidence, fulfilling the **Citation-First** design principle and aiding **Auditability**.
*   **Quantitative Tone Score:** A `ToneScore` object providing an aggregate hawkish/dovish score ($S_{tone}$) and a breakdown by economic factors (inflation, employment, growth, policy bias, uncertainty). The score's confidence level gives Alex an idea of the clarity of the Fed's message.

This multi-dimensional assessment helps Alex interpret the Fed's communication with greater objectivity. For instance, if the overall score is neutral but "inflation_stance" is highly hawkish, it indicates a nuanced policy message that warrants deeper investigation. The calibration against historical scores (a form of **Memory** usage) ensures consistency over time.

---

## 5. Detecting Policy Surprises and Shifts

Identifying divergences from expected norms is crucial for a quantitative analyst. Alex needs to know if the Fed's tone has significantly changed from previous meetings and if there are any truly unexpected elements in the communication that could trigger market reactions.

### 5.1. Quantifying the Tone Delta

Alex calculates the **Tone Delta** by comparing the current meeting's aggregate tone score to a benchmark (e.g., the previous meeting's score). This quantitative measure highlights the magnitude and direction of the shift in sentiment.

The Tone Delta ($\Delta S_{tone}$) is simply the difference between the current tone score ($S_{current}$) and the prior tone score ($S_{prior}$):

$$ \Delta S_{tone} = S_{current} - S_{prior} $$

Its significance can be categorized based on its absolute value, helping Alex quickly assess the impact. For example:
*   $|\Delta S_{tone}| < 0.1$: minimal
*   $0.1 \le |\Delta S_{tone}| < 0.3$: notable
*   $0.3 \le |\Delta S_{tone}| < 0.6$: significant
*   $|\Delta S_{tone}| \ge 0.6$: major

```python
def compute_tone_delta(current_score: ToneScore, prior_score: ToneScore) -> ToneDelta:
    """
    Calculates the change in tone between the current and prior meetings.
    """
    delta = current_score.score - prior_score.score
    
    # Determine significance based on delta magnitude
    if abs(delta) < 0.1:
        delta_significance = "minimal"
    elif 0.1 <= abs(delta) < 0.3:
        delta_significance = "notable"
    elif 0.3 <= abs(delta) < 0.6:
        delta_significance = "significant"
    else:
        delta_significance = "major"
        
    # For simplicity, key drivers are just explanations for now. In reality, LLM would extract this.
    key_drivers = [current_score.explanation, prior_score.explanation]
    
    tone_delta_obj = ToneDelta(
        current_meeting=current_score.citations[0].document_id.split('_')[2], # Extract date from doc_id for simplicity
        prior_meeting=prior_score.citations[0].document_id.split('_')[2],
        current_score=current_score,
        prior_score=prior_score,
        delta=delta,
        delta_significance=delta_significance,
        key_drivers=key_drivers,
        citations=current_score.citations + prior_score.citations # Combine relevant citations
    )
    logger.info("Computed tone delta", delta=delta, significance=delta_significance)
    return tone_delta_obj

# Execution: Compute Tone Delta using current and previous meeting's tone scores
# We use the most recent prior meeting's tone score from our mock data (index 0 for mock_prior_tone_scores)
prior_meeting_tone_score_real = mock_prior_tone_scores[0] # October 2023 tone score

tone_delta = compute_tone_delta(current_tone_score, prior_meeting_tone_score_real)

print("\n--- Tone Delta Analysis ---")
print(f"Current Meeting ({tone_delta.current_meeting}): {tone_delta.current_score.score:+.2f}")
print(f"Prior Meeting ({tone_delta.prior_meeting}): {tone_delta.prior_score.score:+.2f}")
print(f"Delta: {tone_delta.delta:+.2f}")
print(f"Significance: {tone_delta.delta_significance}")
print(f"Key Drivers: {tone_delta.key_drivers[0]}")
```

### 5.2. Detecting Unexpected Policy Surprises

Beyond a general tone shift, Alex also looks for specific, unexpected communication elements—be it a sudden change in language, a revised economic forecast, or a new concern raised. This helps him identify specific catalysts for market movements.

```python
SURPRISE_DETECTION_PROMPT = """
You are an expert financial analyst. Your task is to identify 'surprises' or unexpected elements
in the provided current FOMC documents, compared to the established historical context and prior analysis.
A surprise is defined as a significant divergence from expected norms or previous communication.

Categorize each surprise into one of the following:
- "policy_change": A new or significantly altered policy stance.
- "language_shift": Unforeseen changes in key terminology or emphasis.
- "forecast_revision": Unexpected updates to economic projections (e.g., inflation, GDP, unemployment).
- "dissent": Notable disagreement among committee members.
- "new_concern": Introduction of a previously unhighlighted economic risk or factor.
- "omission": The absence of an expected discussion point or phrase.

For each identified surprise, provide:
1. A concise `description`.
2. Its `market_relevance` (low, medium, high).
3. AT LEAST ONE `citation` directly linking to the source text supporting the surprise.
4. A `confidence` score (0.0 to 1.0) on the certainty of it being a surprise.

Historical Context and Prior Analysis:
{historical_context_summary}

Current FOMC Documents:
{current_documents_text}

Prior Meeting Tone Score and Explanation:
{prior_tone_explanation}

Respond in JSON format with a key "surprises" which is a list of Surprise objects.
Example:
{
  "surprises": [
    {
      "surprise_id": "pol_change_1",
      "category": "policy_change",
      "description": "The Committee explicitly stated that future rate cuts are 'not yet in sight', a more direct statement than previously communicated.",
      "market_relevance": "high",
      "citations": [
        {"citation_id": "c3", "document_id": "presser_2024_01_31", "section_id": "powell_opening", "paragraph_number": 1, "quote": "future rate cuts are 'not yet in sight'", "quote_start": 200, "quote_end": 235}
      ],
      "confidence": 0.98
    }
  ]
}
"""

async def detect_surprises(
    current_documents: List[FOMCDocument],
    historical_documents_context: List[Dict], # List of dicts from search_historical_context
    prior_tone_analysis: ToneScore
) -> List[Surprise]:
    """
    Detects unexpected communication elements or shifts using an LLM.
    """
    current_documents_text = "\n\n".join([
        f"[{doc.document_type.value.upper()}]\n{doc.raw_text}"
        for doc in current_documents
    ])
    
    historical_context_summary = "\n".join([
        f"Meeting {h['meeting_date']} ({h['document_type']}, Section {h['section_id']}): {h['text'][:200]}..."
        for h in historical_documents_context
    ])

    prior_tone_explanation = f"Prior Meeting Tone ({prior_tone_analysis.citations[0].document_id.split('_')[2]}): Score {prior_tone_analysis.score:+.2f}. {prior_tone_analysis.explanation}"

    messages = [
        {"role": "system", "content": SURPRISE_DETECTION_PROMPT.format(
            historical_context_summary=historical_context_summary,
            current_documents_text=current_documents_text,
            prior_tone_explanation=prior_tone_explanation
        )}
    ]

    response = await llm_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.4 # Slightly higher temperature for more exploratory analysis
    )
    
    raw_surprises_output = json.loads(response.choices[0].message.content)
    
    surprises = []
    for surprise_data in raw_surprises_output.get("surprises", []):
        validated_citations = []
        for cite_dict in surprise_data.get("citations", []):
            if validate_citation_exists(cite_dict, current_documents):
                try:
                    validated_citations.append(Citation(**cite_dict))
                except ValidationError as e:
                    logger.warning("Invalid citation format in LLM surprise output", citation=cite_dict, error=str(e))
            else:
                logger.warning("LLM generated unverified citation in surprise detection", citation=cite_dict)

        # Use LLM's confidence if available, otherwise default to 0.7
        llm_confidence = surprise_data.get("confidence", 0.7)
        citation_coverage_confidence = len(validated_citations) / max(len(surprise_data.get("citations", [])), 1)
        final_confidence = (llm_confidence + citation_coverage_confidence) / 2
        
        try:
            surprises.append(Surprise(
                surprise_id=f"surprise_{uuid4().hex[:8]}",
                category=surprise_data["category"],
                description=surprise_data["description"],
                market_relevance=surprise_data["market_relevance"],
                citations=validated_citations,
                confidence=min(max(final_confidence, 0.0), 1.0)
            ))
        except ValidationError as e:
            logger.error("Failed to parse Surprise from LLM output", data=surprise_data, error=str(e))
    
    logger.info("Detected surprises", num_surprises=len(surprises))
    return surprises

# Execution: Detect surprises for the current FOMC meeting
# We pass historical_inflation_context to help the LLM identify divergences.
detected_surprises = await detect_surprises(
    current_documents=current_fomc_documents,
    historical_documents_context=historical_inflation_context,
    prior_tone_analysis=prior_meeting_tone_score_real
)

print("\n--- Detected Policy Surprises ---")
if detected_surprises:
    for s in detected_surprises:
        print(f"Surprise Category: {s.category} (Relevance: {s.market_relevance}, Confidence: {s.confidence:.2f})")
        print(f"Description: {s.description}")
        if s.citations:
            print(f"Example Citation: '{s.citations[0].quote}' from {s.citations[0].document_id}")
        print("-" * 20)
else:
    print("No notable surprises detected for this meeting.")
```

### 5.3. Explanation of Surprises and Shifts Output

Alex now has critical intelligence on potential market-moving events:
*   **Tone Delta:** The `ToneDelta` object quantifies the change in Fed sentiment, indicating whether the Fed has become more hawkish or dovish and by how much. This helps Alex adjust his directional biases.
*   **Detected Surprises:** The list of `Surprise` objects highlights specific unexpected elements in the communication, categorized by type (e.g., `policy_change`, `language_shift`) and assigned a market relevance. These are high-signal events that Alex can quickly investigate, potentially leading to immediate trading opportunities or risk adjustments.

This step embodies **Tool Use (LLM for classification)** and leverages **Memory (historical context)** to make informed judgments about the novelty and impact of the Fed's communication. It directly informs strategic decision-making by identifying points of divergence from expectations.

---

## 6. Ensuring Reliability: Validation and Audit Trail

In financial analysis, trust and compliance are non-negotiable. Alex must rigorously validate all AI-generated insights to prevent misinformation (hallucinations) and ensure every claim is supported by auditable evidence. A comprehensive audit trail is also maintained for regulatory and internal governance.

### 6.1. Automated Validation Checks

Alex implements a pipeline of automated validation checks to scrutinize the generated themes, tone scores, and surprises. This forms the **Reflection** capability of the agent, allowing for self-assessment and error detection before human review.

```python
# --- Validation Tools ---

def validate_citation(citation: Citation, documents: List[FOMCDocument]) -> ValidationCheck:
    """
    Validates that a citation references real source text.
    This is a hard check, not a probabilistic assessment.
    """
    doc_found = next((d for d in documents if d.document_id == citation.document_id), None)
    if doc_found is None:
        return ValidationCheck(check_name="citation_validity", passed=False,
                               details=f"Document '{citation.document_id}' not found.", severity="error")

    section_found = next((s for s in doc_found.sections if s.section_id == citation.section_id), None)
    if section_found is None:
        return ValidationCheck(check_name="citation_validity", passed=False,
                               details=f"Section '{citation.section_id}' not found in document '{citation.document_id}'.", severity="error")

    normalized_quote = " ".join(citation.quote.split()).lower()
    normalized_section_text = " ".join(section_found.text.split()).lower()

    if normalized_quote not in normalized_section_text:
        return ValidationCheck(check_name="citation_validity", passed=False,
                               details=f"Quote '{citation.quote[:50]}...' not found in referenced section '{citation.section_id}'.", severity="error")

    if citation.paragraph_number not in section_found.paragraph_numbers:
        return ValidationCheck(check_name="citation_validity", passed=False,
                               details=f"Paragraph {citation.paragraph_number} not in section '{citation.section_id}'.", severity="warning")

    return ValidationCheck(check_name="citation_validity", passed=True, details="Citation verified.", severity="info")

async def check_hallucination(claim: str, documents: List[FOMCDocument]) -> ValidationCheck:
    """
    Uses LLM to check if a claim is supported by source documents.
    """
    combined_text = "\n\n".join([doc.raw_text for doc in documents])

    messages = [
        {"role": "system", "content": """
            You are a fact-checker. Determine if the given claim is supported by the source documents.
            Respond with JSON:
            {
              "supported": true/false,
              "confidence": 0.0-1.0,
              "supporting_evidence": "quote from source if supported",
              "reason": "explanation of support or lack thereof"
            }
        """},
        {"role": "user", "content": f"""
            CLAIM: {claim}
            SOURCE DOCUMENTS:
            {combined_text}
        """}
    ]
    response = await llm_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0 # Deterministic for factual checks
    )
    result = json.loads(response.choices[0].message.content)

    is_supported = result.get("supported", False)
    confidence = result.get("confidence", 0.0)
    reason = result.get("reason", "No reason provided.")
    
    if not is_supported:
        return ValidationCheck(check_name="hallucination_detection", passed=False,
                               details=f"Claim: '{claim[:100]}...'. Reason: {reason}", severity="error")
    return ValidationCheck(check_name="hallucination_detection", passed=True, details=reason, severity="info")

def verify_tone_bounds(tone_score: ToneScore) -> ValidationCheck:
    """Checks if tone score and confidence are within expected bounds."""
    if not (-1.0 <= tone_score.score <= 1.0):
        return ValidationCheck(check_name="tone_bounds_check", passed=False,
                               details=f"Overall tone score {tone_score.score} is out of bounds [-1.0, 1.0].", severity="error")
    if not (0.0 <= tone_score.confidence <= 1.0):
        return ValidationCheck(check_name="tone_bounds_check", passed=False,
                               details=f"Overall tone confidence {tone_score.confidence} is out of bounds [0.0, 1.0].", severity="error")
    
    # Check component bounds as well
    components = tone_score.components
    for comp_name, comp_score in components.model_dump().items():
        if not (-1.0 <= comp_score <= 1.0):
            return ValidationCheck(check_name="tone_bounds_check", passed=False,
                                   details=f"Component '{comp_name}' score {comp_score} is out of bounds [-1.0, 1.0].", severity="error")
    
    return ValidationCheck(check_name="tone_bounds_check", passed=True, details="Tone scores within bounds.", severity="info")

def no_recommendation_check(memo_content: str) -> ValidationCheck:
    """Ensures no trading recommendations are present in the memo."""
    prohibited_patterns = [
        r"\b(buy|sell|hold)\b",
        r"price target",
        r"position size",
        r"trade recommendation",
        r"investment advice",
        r"\$\d+", # Matches dollar amounts, potentially indicating price targets
    ]
    violations = []
    for pattern in prohibited_patterns:
        import re
        matches = re.findall(pattern, memo_content, re.IGNORECASE)
        if matches:
            violations.append({"pattern": pattern, "matches": matches[:3]}) # Limit examples
    
    if violations:
        return ValidationCheck(check_name="no_recommendation_check", passed=False,
                               details=f"Prohibited content found: {json.dumps(violations)}", severity="error")
    return ValidationCheck(check_name="no_recommendation_check", passed=True, details="No trading recommendations found.", severity="info")

class ValidationPipeline:
    """Multi-stage validation for FOMC memos."""
    def __init__(self):
        pass # Checks are called directly for simplicity in this notebook
        
    async def validate(self, memo: 'FOMCMemo', documents: List[FOMCDocument]) -> ValidationReport:
        all_checks = []
        
        # 1. Validate all citations in themes, tone, surprises
        all_citations = []
        for theme in memo.themes:
            all_citations.extend(theme.citations)
        all_citations.extend(memo.tone_analysis.citations)
        if memo.tone_delta:
            all_citations.extend(memo.tone_delta.citations)
        for surprise in memo.surprises:
            all_citations.extend(surprise.citations)
        
        for citation in all_citations:
            all_checks.append(validate_citation(citation, documents))
        
        # 2. Hallucination check on key summary sections
        # For simplicity, let's check the explanation of the tone score and one theme description
        claims_to_check = [memo.tone_analysis.explanation]
        if memo.themes:
            claims_to_check.append(memo.themes[0].description)
        if memo.surprises:
             claims_to_check.append(memo.surprises[0].description)
        
        for claim in claims_to_check:
            all_checks.append(await check_hallucination(claim, documents))
        
        # 3. Tone Bounds Check
        all_checks.append(verify_tone_bounds(memo.tone_analysis))
        
        # 4. No-Recommendation Check
        memo_full_text = memo.executive_summary + " ".join([t.description for t in memo.themes]) + memo.tone_analysis.explanation + " ".join([s.description for s in memo.surprises]) + memo.market_implications
        all_checks.append(no_recommendation_check(memo_full_text))

        all_passed = all(c.passed for c in all_checks)
        validation_report = ValidationReport(
            validation_timestamp=datetime.now(),
            all_checks_passed=all_passed,
            checks=all_checks
        )
        logger.info("Validation pipeline completed", all_checks_passed=all_passed, num_checks=len(all_checks))
        return validation_report

# Execution (assuming a draft memo will be created in the next step, for now we will simulate a partial memo)
# For demo purposes, we will construct a mock memo to pass to the validation pipeline.
# This mock memo will contain the actual results from previous steps.
mock_memo_for_validation = FOMCMemo(
    memo_id="mock_memo_2024_01_31",
    meeting_date=fomc_date,
    generated_at=datetime.now(),
    version=1,
    status="draft",
    executive_summary="Preliminary analysis suggests a moderately hawkish stance from the FOMC. Inflation concerns remain a primary driver.",
    themes=extracted_themes,
    tone_analysis=current_tone_score,
    tone_delta=tone_delta,
    surprises=detected_surprises,
    historical_context_summary="Historical context shows recent inflation language has been consistent, indicating the Fed's long-standing concern.",
    market_implications="Potential for sustained higher rates could impact long-duration assets. Markets might need to adjust expectations for rate cut timing.",
    all_citations=current_tone_score.citations + (extracted_themes[0].citations if extracted_themes else []) + (detected_surprises[0].citations if detected_surprises else []),
    confidence_assessment=ConfidenceAssessment(overall_confidence=0.8, evidence_strength=0.9, citation_coverage=0.95, model_agreement=1.0, flags=[]),
    validation_report=ValidationReport(validation_timestamp=datetime.now(), all_checks_passed=False, checks=[]), # Will be updated by pipeline
    review_history=[]
)

validation_pipeline = ValidationPipeline()
final_validation_report = await validation_pipeline.validate(mock_memo_for_validation, current_fomc_documents)

print("\n--- Automated Validation Report ---")
print(f"Overall Validation Passed: {final_validation_report.all_checks_passed}")
for check in final_validation_report.checks:
    icon = "✅" if check.passed else ("⚠️" if check.severity == "warning" else "❌")
    print(f"{icon} {check.check_name}: {check.details} (Severity: {check.severity.upper()})")

```

### 6.2. Maintaining an Audit Trail

Every action taken by the AI agent, from fetching documents to computing scores and performing validation, is meticulously logged. This structured audit trail (using `structlog`) provides complete transparency and traceability, which is indispensable for regulatory compliance and debugging in a financial context.

```python
# --- Audit Trail Manager ---
class AuditTrail:
    """Immutable audit trail for all agent actions."""
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.entries: List[AuditEntry] = []

    def log_action(
        self,
        agent: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        duration_ms: int
    ) -> None:
        """Logs an agent action."""
        entry = AuditEntry(
            entry_id=f"audit_{uuid4().hex[:8]}",
            request_id=self.request_id,
            timestamp=datetime.now(),
            agent=agent,
            action=action,
            inputs=inputs,
            outputs=outputs,
            duration_ms=duration_ms
        )
        self.entries.append(entry)
        logger.info(
            "Agent action logged",
            entry_id=entry.entry_id,
            request_id=self.request_id,
            agent=agent,
            action=action,
            duration_ms=duration_ms
        )

    def log_human_decision(
        self,
        decision: str,
        comments: str,
        reviewer_id: str,
        memo_version: int
    ) -> None:
        """Logs human review decision - special entry type."""
        entry = AuditEntry(
            entry_id=f"audit_{uuid4().hex[:8]}",
            request_id=self.request_id,
            timestamp=datetime.now(),
            agent="human",
            action=f"review_decision_{decision}",
            inputs={"memo_version": memo_version},
            outputs={"decision": decision, "comments": comments},
            duration_ms=0,
            reviewer_id=reviewer_id
        )
        self.entries.append(entry)
        logger.info(
            "Human decision logged",
            entry_id=entry.entry_id,
            request_id=self.request_id,
            decision=decision,
            reviewer_id=reviewer_id
        )

    def export(self) -> Dict[str, Any]:
        """Exports audit trail for archival."""
        return {
            "request_id": self.request_id,
            "export_timestamp": datetime.now().isoformat(),
            "entry_count": len(self.entries),
            "entries": [e.model_dump() for e in self.entries]
        }

# Execution: Simulate logging key actions
request_id = "fomc_analysis_2024_01_31_req_001"
audit_trail_manager = AuditTrail(request_id=request_id)

start_time = datetime.now()
# Log document fetching (simplified inputs/outputs for demo)
audit_trail_manager.log_action(
    agent="Ingestor Agent",
    action="fetch_documents",
    inputs={"meeting_date": str(fomc_date)},
    outputs={"num_documents": len(current_fomc_documents)},
    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000)
)

start_time = datetime.now()
# Log theme extraction
audit_trail_manager.log_action(
    agent="Analyst Agent",
    action="extract_themes",
    inputs={"num_documents": len(current_fomc_documents), "n_themes": num_themes_to_extract},
    outputs={"num_extracted_themes": len(extracted_themes)},
    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000)
)

start_time = datetime.now()
# Log tone score computation
audit_trail_manager.log_action(
    agent="Analyst Agent",
    action="compute_tone_score",
    inputs={"num_documents": len(current_fomc_documents)},
    outputs={"tone_score": current_tone_score.score, "confidence": current_tone_score.confidence},
    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000)
)

start_time = datetime.now()
# Log surprise detection
audit_trail_manager.log_action(
    agent="Analyst Agent",
    action="detect_surprises",
    inputs={"num_documents": len(current_fomc_documents)},
    outputs={"num_surprises": len(detected_surprises)},
    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000)
)

start_time = datetime.now()
# Log validation report (using the actual report generated above)
audit_trail_manager.log_action(
    agent="Validator Agent",
    action="run_validation_pipeline",
    inputs={"memo_id": mock_memo_for_validation.memo_id, "num_documents": len(current_fomc_documents)},
    outputs={"all_checks_passed": final_validation_report.all_checks_passed, "num_checks": len(final_validation_report.checks)},
    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000)
)

print("\n--- Audit Trail (Last 3 Entries) ---")
for entry in audit_trail_manager.entries[-3:]:
    print(f"Timestamp: {entry.timestamp.isoformat()} | Agent: {entry.agent} | Action: {entry.action} | Outputs: {entry.outputs}")
```

### 6.3. Explanation of Validation and Audit Output

Alex reviews the `ValidationReport` to quickly grasp the quality of the AI-generated output:
*   **Passed Checks:** He sees which checks (e.g., citation validity, tone bounds) have passed, confirming adherence to established quality criteria.
*   **Failed Checks/Warnings:** Any failures or warnings are explicitly flagged, indicating areas that require human attention (e.g., potential hallucinations or uncited claims). This directly facilitates **Human-in-the-Loop approval** and **Collaboration**.

The **Audit Trail** provides a granular record of every step the agent took, along with its inputs and outputs. This `LogEntry` stream demonstrates:
*   **Transparency:** All processing steps are visible and traceable.
*   **Reproducibility:** If an issue arises, Alex can trace back the exact data and actions that led to a particular output.
*   **Compliance:** In a regulated financial environment, this comprehensive log is crucial for demonstrating that due diligence was performed and that the AI system operated within defined parameters.

This combination of automated validation and a detailed audit trail significantly enhances the trustworthiness and utility of the AI agent for Alex's role.

---

## 7. Synthesizing Insights into a Research Memo

The final step in Alex's workflow is to consolidate all the generated insights—themes, tone, surprises, and validation results—into a comprehensive, structured research memo. This `FOMCMemo` object is the ultimate deliverable for his team, providing a holistic view of the FOMC meeting's implications for macro strategies.

### 7.1. Generating the Structured FOMC Research Memo

Alex integrates all the previously generated Pydantic objects into a single `FOMCMemo`. He also adds an executive summary, potentially drafted by an LLM, and explicitly includes the validation report and a confidence assessment.

```python
MEMO_GENERATION_PROMPT = """
You are a senior financial research analyst. Synthesize the following structured analysis
of an FOMC meeting into a concise, professional executive summary (2-3 paragraphs)
and market implications (1-2 paragraphs).

Ensure the summary highlights the overall tone, key themes, and any significant surprises.
The market implications should discuss potential impacts on asset classes or investment strategies
based on the Fed's stance, but **MUST NOT** include specific trading recommendations, price targets, or "buy/sell/hold" language.

Current Meeting Date: {meeting_date}
Extracted Themes: {themes_summary}
Tone Analysis: Score {tone_score:+.2f}, Confidence {tone_confidence:.0%}. Explanation: {tone_explanation}
Tone Delta vs. Prior: {delta_summary}
Detected Surprises: {surprises_summary}
Historical Context: {historical_context_summary}
Validation Report: All checks passed: {validation_passed}. {validation_details}

Respond in JSON format with keys: `executive_summary` and `market_implications`.
Example:
{{
  "executive_summary": "The FOMC meeting revealed a moderately hawkish stance, driven primarily by persistent inflation concerns. Key themes included continued vigilance on price stability and a cautious outlook on labor market deceleration. The tone shift from the prior meeting was notable...",
  "market_implications": "The Fed's sustained hawkish bias suggests that interest rates may remain elevated for longer than anticipated, potentially putting upward pressure on short-term bond yields and posing challenges for growth-sensitive equity sectors. Investors should monitor upcoming inflation data closely..."
}}
"""

async def generate_fomc_memo(
    meeting_date: date,
    themes: List[ThemeExtraction],
    tone_analysis: ToneScore,
    tone_delta: ToneDelta,
    surprises: List[Surprise],
    historical_context_summary: str,
    validation_report: ValidationReport,
    market_implications_input: Optional[str] = None # Allow manual input or LLM generation
) -> FOMCMemo:
    """
    Generates the final FOMC Research Memo by synthesizing all analysis components.
    """
    themes_summary = "; ".join([f"{t.theme_name} (Confidence: {t.confidence:.0%})" for t in themes]) if themes else "No themes extracted."
    surprises_summary = "; ".join([f"{s.category}: {s.description[:50]}... (Relevance: {s.market_relevance})" for s in surprises]) if surprises else "No notable surprises."
    delta_summary = f"{tone_delta.delta_significance} shift ({tone_delta.delta:+.2f})" if tone_delta else "No delta computed."
    
    validation_details = "; ".join([f"{c.check_name}: {'Passed' if c.passed else 'Failed'}" for c in validation_report.checks])

    # Generate executive summary and market implications using LLM
    llm_generation_messages = [
        {"role": "system", "content": MEMO_GENERATION_PROMPT.format(
            meeting_date=meeting_date.isoformat(),
            themes_summary=themes_summary,
            tone_score=tone_analysis.score,
            tone_confidence=tone_analysis.confidence,
            tone_explanation=tone_analysis.explanation,
            delta_summary=delta_summary,
            surprises_summary=surprises_summary,
            historical_context_summary=historical_context_summary,
            validation_passed=validation_report.all_checks_passed,
            validation_details=validation_details
        )}
    ]
    llm_response = await llm_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=llm_generation_messages,
        response_format={"type": "json_object"},
        temperature=0.4
    )
    llm_generated_content = json.loads(llm_response.choices[0].message.content)
    
    executive_summary = llm_generated_content["executive_summary"]
    market_implications = llm_generated_content["market_implications"]
    if market_implications_input: # Override if manual input is provided
        market_implications = market_implications_input

    # Aggregate all citations for the memo
    memo_all_citations = []
    for theme in themes:
        memo_all_citations.extend(theme.citations)
    memo_all_citations.extend(tone_analysis.citations)
    if tone_delta:
        memo_all_citations.extend(tone_delta.citations)
    for surprise in surprises:
        memo_all_citations.extend(surprise.citations)

    # Calculate overall confidence assessment (simplified for demo)
    total_confidence = 0.0
    count_confidence_sources = 0
    if tone_analysis.confidence is not None:
        total_confidence += tone_analysis.confidence
        count_confidence_sources += 1
    if themes:
        theme_avg_confidence = sum(t.confidence for t in themes) / len(themes)
        total_confidence += theme_avg_confidence
        count_confidence_sources += 1
    if surprises:
        surprise_avg_confidence = sum(s.confidence for s in surprises) / len(surprises)
        total_confidence += surprise_avg_confidence
        count_confidence_sources += 1

    overall_confidence_calculated = total_confidence / count_confidence_sources if count_confidence_sources > 0 else 0.7 # Default if no components
    
    citation_coverage = len(memo_all_citations) / max(len(memo_all_citations) + 5, 1) # Mock calculation, assumes some uncited facts are possible
    
    confidence_assessment = ConfidenceAssessment(
        overall_confidence=min(max(overall_confidence_calculated, 0.0), 1.0),
        evidence_strength=min(max(overall_confidence_calculated * 0.9, 0.0), 1.0), # Example
        citation_coverage=min(max(citation_coverage, 0.0), 1.0),
        model_agreement=1.0, # Assumed 1.0 if only one model is used for core analysis
        flags=[] # Would populate with validation warning/error flags from validation_report
    )

    fomc_memo = FOMCMemo(
        memo_id=f"fomc_memo_{meeting_date.isoformat()}",
        meeting_date=meeting_date,
        generated_at=datetime.now(),
        version=1,
        status="pending_review", # Ready for human review
        executive_summary=executive_summary,
        themes=themes,
        tone_analysis=tone_analysis,
        tone_delta=tone_delta,
        surprises=surprises,
        historical_context_summary="Summarized historical context: " + " ".join([h['text'][:100] for h in historical_inflation_context]) + "...",
        market_implications=market_implications,
        all_citations=memo_all_citations,
        confidence_assessment=confidence_assessment,
        validation_report=validation_report,
        review_history=[]
    )
    logger.info("FOMC Memo generated", memo_id=fomc_memo.memo_id, status=fomc_memo.status)
    return fomc_memo

# Execution: Generate the final FOMC Memo
final_fomc_memo = await generate_fomc_memo(
    meeting_date=fomc_date,
    themes=extracted_themes,
    tone_analysis=current_tone_score,
    tone_delta=tone_delta,
    surprises=detected_surprises,
    historical_context_summary="Recent historical context focused on inflation persistence and labor market tightness.",
    validation_report=final_validation_report,
    market_implications_input="The Fed's continued focus on inflation, despite easing, signals a cautious approach. This could imply a longer period of restrictive policy, potentially supporting the dollar and favoring value stocks over high-growth equities in the near term."
)

print("\n--- Final FOMC Research Memo Summary ---")
print(f"Memo ID: {final_fomc_memo.memo_id}")
print(f"Meeting Date: {final_fomc_memo.meeting_date}")
print(f"Status: {final_fomc_memo.status}")
print("\n**Executive Summary:**")
print(final_fomc_memo.executive_summary)
print("\n**Key Themes:**")
for theme in final_fomc_memo.themes:
    print(f"- {theme.theme_name}: {theme.description}")
print("\n**Overall Tone:**")
print(f"Score: {final_fomc_memo.tone_analysis.score:+.2f} (Confidence: {final_fomc_memo.tone_analysis.confidence:.0%})")
print(f"Explanation: {final_fomc_memo.tone_analysis.explanation}")
if final_fomc_memo.tone_delta:
    print(f"\n**Tone Delta vs. Prior Meeting:**")
    print(f"Delta: {final_fomc_memo.tone_delta.delta:+.2f} ({final_fomc_memo.tone_delta.delta_significance})")
print("\n**Detected Surprises:**")
if final_fomc_memo.surprises:
    for s in final_fomc_memo.surprises:
        print(f"- {s.category} (Relevance: {s.market_relevance}): {s.description}")
else:
    print("No notable surprises identified.")
print("\n**Market Implications:**")
print(final_fomc_memo.market_implications)
print(f"\n**Overall Confidence Assessment (from agent):** {final_fomc_memo.confidence_assessment.overall_confidence:.0%}")
print(f"**Validation Status:** {'Passed' if final_fomc_memo.validation_report.all_checks_passed else 'Failed'}")
```

### 7.2. Explanation of Research Memo Output

Alex has successfully generated a comprehensive `FOMCMemo` object, which encapsulates all the AI agent's findings for the January 2024 meeting. This final structured output is the culmination of the entire workflow and demonstrates:
*   **Structured Deliverable:** All key insights (themes, tone, surprises, historical context) are organized into a single, well-defined Pydantic model, making it easy for portfolio managers to consume and analyze.
*   **Actionable Intelligence:** The executive summary and market implications provide direct, actionable intelligence for strategic decision-making, without overstepping into specific trading recommendations.
*   **End-to-End Auditability:** The memo includes explicit references to all underlying citations, a detailed validation report, and an overall confidence assessment. This robust design supports the **AUDITABLE** design principle, providing a clear lineage from source documents to conclusions.

This workflow empowers Alex to consistently produce high-quality, objective, and auditable research, significantly enhancing GMAF's ability to navigate the complexities of Federal Reserve policy and capitalize on market opportunities. The entire process, from ingestion to final memo, demonstrates a pragmatic application of agentic AI concepts in a regulated financial services context.
```