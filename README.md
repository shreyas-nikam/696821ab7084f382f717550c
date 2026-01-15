Here's a comprehensive `README.md` file for your Streamlit application lab project, designed to be professional and informative.

---

# QuLab: FOMC Research Agent

![Streamlit App Screenshot](https://via.placeholder.com/1200x600?text=QuLab+FOMC+Research+Agent+Screenshot)
*(Replace this with an actual screenshot of your running application)*

## Project Title and Description

**QuLab: FOMC Research Agent** is a sophisticated Streamlit application designed to automate and streamline the analysis of Federal Open Market Committee (FOMC) communications. Leveraging advanced AI agents and natural language processing, this tool transforms raw FOMC statements, minutes, and press conference transcripts into structured, auditable investment research memos.

The application aims to provide quantitative analysts and financial researchers with deep insights into the Federal Reserve's stance, policy changes, and potential market surprises. It facilitates an end-to-end workflow from document ingestion and AI-powered analysis to human review and historical tracking, ensuring that every insight is backed by verifiable citations.

## Features

This application offers a robust set of features to empower researchers:

*   **AI-Powered Document Ingestion & Analysis**: Automatically ingests and processes FOMC statements, minutes, and press conference transcripts for selected meeting dates.
*   **Intelligent Theme Extraction**: Identifies and summarizes key themes from FOMC communications, complete with confidence scores and direct citations to source documents.
*   **Quantitative Tone Analysis**: Calculates an overall tone score (Hawkish/Dovish/Neutral) and breaks it down into components (e.g., inflation stance, employment outlook). It also computes the tone delta against previous meetings.
*   **Policy Surprise Detection**: Highlights potential policy surprises or significant shifts from prior expectations, along with their market relevance and supporting evidence.
*   **Automated Validation & Confidence Scoring**: Each generated memo undergoes automated validation checks for citation accuracy, coherence, and consistency. A comprehensive confidence assessment is provided for all outputs.
*   **Auditable Workflow & Citation Tracking**: Every piece of information, from themes to tone, is linked back to specific paragraphs in the source documents, ensuring transparency and auditability. A detailed audit trail of the AI workflow execution is maintained.
*   **Human-in-the-Loop Review**: Provides a dedicated interface for human reviewers to approve, request revisions for, or reject AI-generated memos, ensuring quality and accuracy before finalization.
*   **Interactive Visualizations**:
    *   **Tone Trajectory**: Visualizes the historical evolution of FOMC tone over time.
    *   **Tone Components Breakdown**: Bar charts illustrating the contribution of different factors to the overall tone.
    *   **Citation-Theme Network (Conceptual)**: A network graph (if fully implemented) to show relationships between themes and their supporting citations.
*   **Comprehensive Analysis History**: Stores and allows browsing of all past analyses, including approved memos, revision requests, and rejections.
*   **Responsive User Interface**: Built with Streamlit, featuring a clean dark theme and an intuitive navigation sidebar for an optimal user experience.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.9+
*   `pip` (Python package installer)
*   An OpenAI API key (or equivalent for other LLM providers if configured in `source.py`)

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/quolab-fomc-research-agent.git
    cd quolab-fomc-research-agent
    ```
    *(Replace `your-username/quolab-fomc-research-agent.git` with your actual repository URL)*

2.  **Create a Virtual Environment:**
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit>=1.30.0
    plotly>=5.18.0
    python-dotenv>=1.0.0 # If using .env for API keys
    langchain>=0.1.0
    langchain-openai>=0.0.0 # Or langchain-anthropic etc.
    pydantic>=2.0.0
    # Add any other specific libraries used in source.py
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Key:**
    The application requires an API key for the Large Language Models (LLMs) used by the agents.
    Create a `.env` file in the project root directory and add your API key:
    ```
    OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY"
    # Or ANTHROPIC_API_KEY="sk-YOUR_ANTHROPIC_API_KEY" if using Anthropic
    ```
    *(Note: Ensure your `source.py` is configured to load these environment variables, e.g., using `dotenv` library).*

## Usage

1.  **Run the Streamlit Application:**
    Navigate to the project root directory in your terminal and run:
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

2.  **Navigate the Application:**
    *   **Sidebar Navigation**: Use the left sidebar to switch between `Home`, `New Analysis`, `Pending Reviews`, `Analysis History`, and `Settings`.
    *   **Home Page**: Provides an overview and quick links to core functionalities.
    *   **New Analysis**:
        *   Select an FOMC meeting date.
        *   Configure analysis options (historical context, number of themes, auto-validation, confidence thresholds).
        *   Click "🚀 Start Analysis" to initiate the AI workflow. A progress bar and log will track the analysis.
        *   Upon completion, a preview of the generated memo will be displayed.
    *   **Pending Reviews**:
        *   View a list of AI-generated memos awaiting human review.
        *   Select a memo to dive into its detailed summary, themes, tone analysis, surprises, validation report, and audit trail.
        *   Make a decision (`Approve`, `Request Revision`, `Reject`) and provide comments.
    *   **Analysis History**:
        *   Browse previously processed and reviewed analyses.
        *   View historical tone trajectories, full memo details, and complete audit logs for any past request.
    *   **Settings**: (Conceptual for this lab project) This page would allow configuration of API keys, model parameters, and other system settings.

## Project Structure

```
quolab-fomc-research-agent/
├── app.py                     # Main Streamlit application entry point
├── source.py                  # Core logic, AI agent definitions, data models,
│                              # data handling (e.g., FOMC document ingestion,
│                              # memo storage, workflow orchestration).
├── requirements.txt           # Python dependencies for the project
├── .env                       # Environment variables (e.g., API keys)
├── README.md                  # Project documentation (this file)
└── data/                      # (Optional) Directory for raw/processed documents,
    └── fomc_docs/             # or persistent memo storage (if not in-memory)
        └── 20231213_statement.txt
        └── ...
```

*   `app.py`: Handles the Streamlit UI components, session state management, and routes user interactions to the backend logic.
*   `source.py`: Contains all the "business logic" including:
    *   Pydantic data models (e.g., `FOMCMemo`, `Theme`, `ToneAnalysis`, `ValidationReport`, `AgentState`, `ReviewAction`, `Citation`).
    *   The `SessionStateManager` class for managing application-wide session data and request history.
    *   Functions for interacting with LLMs and orchestrating the LangChain-based agents (`create_fomc_workflow`).
    *   Functions for data ingestion (`main_ingestion`), document availability (`check_document_availability`), and data persistence (`load_memo`, `update_memo_status`, `get_pending_reviews`).
    *   Helper functions for plotting (`render_tone_trajectory`, `render_citation_network`).
    *   (Assumed) `mock_prior_tone_scores` for historical data visualization.

## Technology Stack

*   **Application Framework**: [Streamlit](https://streamlit.io/)
*   **Programming Language**: Python 3.9+
*   **AI/LLM Orchestration**: [LangChain](https://www.langchain.com/) (Implied by agentic workflow and `AgentState`)
*   **Large Language Models (LLMs)**: OpenAI GPT models (or equivalent, e.g., Anthropic Claude, as configured by API keys)
*   **Data Models**: [Pydantic](https://docs.pydantic.dev/latest/)
*   **Asynchronous Operations**: `asyncio`
*   **Data Visualization**: [Plotly](https://plotly.com/python/)
*   **Environment Variables**: `python-dotenv` (for secure API key management)
*   **Data Storage (Lab Project)**: Primarily in-memory session state, potentially simple JSON or file-based persistence for memos as implemented in `source.py`.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  **Fork** the repository.
2.  **Clone** your forked repository.
3.  Create a new **branch** (`git checkout -b feature/your-feature-name` or `bugfix/issue-description`).
4.  Make your changes and **commit** them (`git commit -m "Add: brief description of changes"`).
5.  **Push** to your branch (`git push origin feature/your-feature-name`).
6.  Open a **Pull Request** to the `main` branch of the original repository.

Please ensure your code adheres to standard Python best practices and includes appropriate docstrings and comments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Create a `LICENSE` file in your root directory if you haven't already).*

## Contact

For questions, feedback, or collaborations, please reach out to:

*   **Project Maintainer**: [Alex Chen] (or "QuantUniversity Team")
*   **Email**: [alex.chen@quantuniversity.com] (or your contact email)
*   **GitHub Profile**: [https://github.com/your-github-username](https://github.com/your-github-username)

---