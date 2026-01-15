id: 696821ab7084f382f717550c_user_guide
summary: FOMC Research Agent User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Navigating the FOMC Research Agent

## 1. Introduction to QuLab: The FOMC Research Agent
Duration: 05:00

Welcome to **QuLab: The FOMC Research Agent**, a sophisticated application designed to transform complex Federal Open Market Committee (FOMC) communications into structured, auditable investment research. In the dynamic world of quantitative finance, understanding the Federal Reserve's stance and future policy signals is crucial. Traditionally, this involves sifting through vast amounts of unstructured text from statements, meeting minutes, and press conferences.

This application leverages advanced AI agents to automate the extraction of key insights, allowing financial professionals to quickly grasp the nuances of FOMC meetings. It focuses on:
*   **Theme Extraction:** Identifying the core topics and discussions.
*   **Tone Analysis:** Quantifying the hawkish or dovish sentiment.
*   **Policy Surprise Detection:** Highlighting unexpected shifts in policy.
*   **Auditability:** Providing direct citations to source documents for every piece of extracted information, ensuring transparency and trustworthiness.

This codelab will guide you step-by-step through the functionalities of the QuLab application. By the end, you'll understand how to initiate an analysis, interpret its results, review generated memos, and explore historical trends, all without needing to delve into the underlying code. The goal is to empower you to utilize this tool effectively for informed decision-making.

## 2. Navigating the Application Interface
Duration: 03:00

Upon launching QuLab, you'll be greeted by a clean, intuitive interface. The primary navigation is handled through the **sidebar** on the left.

The sidebar contains several key sections:

*   **🏠 Home:** This is your landing page, providing an overview and quick access to core features.
*   **📈 New Analysis:** Where you initiate a fresh analysis of an FOMC meeting.
*   **📝 Pending Reviews:** A crucial section where draft analyses await human verification and approval. The number in parentheses indicates how many analyses are currently pending review.
*   **📚 Analysis History:** A repository of all past completed or reviewed analyses, allowing you to track trends over time.
*   **⚙️ Settings:** A conceptual page for configuring application parameters (though not fully implemented in this demo).

Below the navigation, you'll find the **System Status** dashboard, which provides quick metrics:
*   **Documents:** Shows the total number of FOMC documents loaded into the system.
*   **Analyses:** Displays the total number of analyses initiated.

<aside class="positive">
<b>Tip:</b> You can always return to the Home page or any other section by clicking the corresponding button in the sidebar.
</aside>

## 3. Starting a New FOMC Analysis
Duration: 10:00

To generate insights from an FOMC meeting, you'll start a new analysis.

1.  Click on the **📈 New Analysis** button in the sidebar.

2.  **Select FOMC Meeting:**
    *   In the "Meeting Date" dropdown, choose the specific FOMC meeting you wish to analyze. The application provides a list of available meeting dates.
    *   To the right, the "Available Documents" section will confirm if the essential documents (Statement, Minutes, Press Conference) for the selected meeting are present. A green checkmark (✅) indicates availability, while a red cross (❌) means it's missing.

3.  **Analysis Options:**
    This section allows you to tailor the analysis to your needs:
    *   **Include Historical Context:** (Default: On) If checked, the agent will compare the current meeting's tone and policy shifts against prior meetings, providing a richer perspective.
    *   **Number of Themes to Extract:** Use the slider to specify how many key themes you want the AI to identify from the meeting materials, ranging from 3 to 10.
    *   **Auto-Validate Citations:** (Default: On) This crucial feature ensures that every piece of information, like a theme description or a tone explanation, is directly traceable to a specific quote from the source FOMC documents. If a citation cannot be verified, it will be flagged in the review process.
    *   **Minimum Confidence Threshold:** Adjust this slider to set the minimum confidence level for themes, tone, and surprises to be displayed. Lowering this value might reveal more subtle insights, but with potentially higher uncertainty.

4.  **Execute Analysis:**
    *   Once you've configured your options, click the **🚀 Start Analysis** button.
    *   The application will then initiate an AI-driven workflow. You'll see a **progress bar** and **status messages** updating in real-time, indicating which step of the analysis the agent is currently performing (e.g., "Running: Document Ingestion", "Running: Theme Extraction").
    *   The workflow involves several steps, from processing the raw documents to extracting themes, analyzing tone, detecting surprises, and generating a draft memo. This process can take a few minutes, depending on the complexity and the current load.

<aside class="positive">
<b>Remember:</b> The progress bar and status messages keep you informed. Avoid navigating away while an analysis is actively running to prevent interruption.
</aside>

## 4. Understanding the Analysis Output (Memo Preview)
Duration: 07:00

Once the analysis workflow completes, the application will display a "Last Generated Memo Preview" at the bottom of the "New FOMC Analysis" page. This preview gives you a quick summary of the findings.

The memo preview includes:

*   **Meeting Date:** The date of the FOMC meeting that was analyzed.
*   **Status:** Indicates the current state of the memo (e.g., `pending_review`, `completed`).
*   **Overall Confidence:** A percentage reflecting the AI's confidence in the analysis results.
*   **Validation Status:** Shows whether all automated checks passed (✅) or if there were warnings/failures (❌).

You'll then see a summary of the core insights:

*   **Executive Summary:** A concise overview of the meeting's key takeaways.
*   **Key Themes:** A list of the identified themes, each with its confidence score and a brief description. This gives you a high-level understanding of the main topics discussed.

<aside class="positive">
<b>Next Step:</b> This preview is just a glimpse. For a detailed, auditable breakdown, you'll need to proceed to the "Review Pending Analysis" section by clicking the <b>📝 Go to Review for full details</b> button.
</aside>

## 5. Reviewing Pending Analyses
Duration: 15:00

The review page is where human expertise meets AI-generated insights. This is a critical step to verify the accuracy and validity of the analysis before it's approved for use.

1.  Click on the **📝 Pending Reviews** button in the sidebar.

2.  **Select Analysis to Review:**
    *   If there are pending analyses, a dropdown will appear, allowing you to select an analysis by its meeting date and current status.
    *   Once selected, the full memo details will load below.

3.  **FOMC Memo Overview:**
    *   You'll see the memo's meeting date, its current status, and when it was generated. Pay attention to the status, indicated by a colored tag (e.g., <span class='status-pending'>Pending Review</span>, <span class='status-approved'>Approved</span>).

4.  **Detailed Review Tabs:**
    The memo is broken down into several tabs for comprehensive review:

    *   **Summary:**
        *   **Executive Summary:** The main points of the memo.
        *   **Market Implications:** Potential impacts on financial markets.
        *   **Historical Context Summary:** How the current meeting compares to previous ones.

    *   **Themes:**
        *   Lists all extracted themes with their confidence scores.
        *   Click on each theme to expand it and view its full description, associated keywords, and most importantly, the **supporting citations**.
        *   **Citations:** Each citation links a specific piece of text from the memo back to the exact quote and location (Document ID, Section, Paragraph) in the original FOMC document. This feature ensures **full auditability** and allows you to verify the AI's interpretation against the source material.

    *   **Tone Analysis:**
        *   **Overall Tone:** Shows a numerical score (e.g., `+0.25` for hawkish, `-0.15` for dovish) along with a qualitative label (Hawkish, Dovish, Neutral).
        *   **Confidence:** The AI's confidence in its tone assessment.
        *   **Delta vs Prior:** If historical context was included, this indicates the change in tone compared to the previous meeting, along with its significance (e.g., `+0.10 Significant Hawkish Shift`).
        *   **Tone Components Breakdown:** A bar chart visually represents the sentiment scores for key economic components like inflation, employment, growth outlook, policy bias, and uncertainty. This provides a granular view of what factors contribute to the overall tone.
        *   **Explanation:** A textual explanation for the tone assessment.
        *   **Supporting Citations:** Just like themes, the tone assessment is backed by direct quotes from the FOMC documents.

        <aside class="positive">
        <b>Concept: Tone Score Formula</b>
        The overall tone score ($S_{tone}$) is a composite measure, often calculated as a weighted average of scores from various components:
        $$ S_{tone} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot s_i $$
        where $N$ is the number of components (e.g., inflation stance, employment outlook), $w_i$ is the weighting of component $i$, and $s_i$ is the sentiment score (e.g., from -1 for dovish to +1 for hawkish) for component $i$.
        </aside>

    *   **Surprises:**
        *   Highlights any detected policy surprises, categorizing them (e.g., `Rate_Path_Shift`, `Economic_Outlook_Change`).
        *   For each surprise, it provides its market relevance, confidence, description, and supporting citations.

    *   **Validation:**
        *   **Automated Validation Report:** Details if the AI's internal checks (e.g., citation verification, consistency) passed or failed. This gives you an indication of the memo's internal integrity.
        *   **Confidence Assessment:** Provides a breakdown of confidence metrics, including `Overall Confidence`, `Evidence Strength`, and `Citation Coverage`. It also lists any `Flags/Concerns` identified during the validation process.

    *   **Audit Trail:**
        *   This tab provides a **partial log (last 5 entries)** of the steps the AI agent took during the analysis process. This offers transparency into the workflow and can be useful for debugging or understanding specific decisions made by the agent.
        *   It typically shows events like `on_chain_start`, `on_tool_start`, `on_agent_action`, and `on_chain_end`, detailing the execution flow.

## 6. Making Review Decisions
Duration: 05:00

After thoroughly reviewing the memo across all tabs, you must make a decision about its quality and readiness.

1.  **Review Decision Buttons:**
    At the bottom of the review page, you'll find three buttons:
    *   **✅ Approve:** Use this if you are satisfied with the memo's accuracy and completeness. An approved memo is considered final and ready for use.
    *   **✏️ Request Revision:** Select this if the memo requires changes or further analysis. This indicates that while the analysis is useful, it needs improvements.
    *   **❌ Reject:** Choose this if the memo is fundamentally flawed or inaccurate and cannot be used.

2.  **Add Comments:**
    *   Regardless of your decision, a text area for "Comments (required)" will appear. **It is mandatory to provide comments** explaining your decision.
    *   For "Request Revision" or "Reject," detailed comments are crucial for providing actionable feedback to improve future analyses or understand why an analysis was deemed unsuitable. For "Approve," comments can validate specific strengths or nuances.

3.  **Submit Decision:**
    *   Once your comments are entered, click the **Submit Decision** button.
    *   The application will update the memo's status, record your decision and comments in its review history, and then navigate you back to the "Pending Reviews" page (or clear the current selection if the memo is no longer pending).

<aside class="negative">
<b>Important:</b> Always provide clear and constructive comments. This feedback loop is essential for improving the AI agent's performance and ensuring the quality of the research output.
</aside>

## 7. Exploring Analysis History
Duration: 10:00

The "Analysis History" page serves as a repository for all past analyses, whether approved, rejected, or revised. This section allows you to track trends, review historical context, and revisit any completed research.

1.  Click on the **📚 Analysis History** button in the sidebar.

2.  **Select Past Analysis:**
    *   A dropdown will list all completed analyses, sorted by meeting date, allowing you to select any prior memo for review.

3.  **FOMC Memo History Overview:**
    *   Similar to the review page, you'll see the memo's meeting date, status, generation timestamp, and crucially, its **Last Reviewed** information (timestamp, reviewer, action).
    *   The **Overall Confidence** and **Validation Status** provide quick summaries of its quality.

4.  **Historical Analysis Tabs:**
    The history view offers specialized tabs for historical context:

    *   **Tone Trajectory:**
        *   This tab visualizes the evolution of the FOMC's overall tone over time.
        *   It plots the tone scores of multiple meetings, including the currently selected one and prior meetings, allowing you to observe shifts and trends (e.g., a gradual hawkish pivot, or a sudden dovish turn).
        *   Understanding these trajectories is vital for quantitative analysts to model market reactions to Fed policy.
        *   The formula for the tone score, $ S_{tone} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot s_i $, is also presented to reinforce the conceptual basis of the scoring.

    *   **Citation-Theme Network:**
        *   This conceptual visualization (if fully implemented) would show the relationships between different themes and the citations that support them.
        *   It helps in understanding how evidence from the source documents connects various identified themes, providing a deeper structural understanding of the FOMC communication.

    *   **Full Memo View:**
        *   This tab presents the entire generated memo, including the Executive Summary, Key Themes (with descriptions and citations), Tone Analysis, Detected Surprises, Market Implications, and the Automated Validation Report, all in a single consolidated view. This is useful for a quick, comprehensive re-read.

    *   **Audit Log:**
        *   Unlike the limited view in "Pending Reviews," this tab provides the **full audit log** for the selected analysis request.
        *   It records every step and event in the AI workflow, offering complete transparency and the ability to trace the agent's reasoning and actions from start to finish. This can be invaluable for forensic analysis or understanding complex AI behaviors.

## 8. Understanding Settings (Conceptual)
Duration: 02:00

The **⚙️ Settings** page in QuLab is designed to be the central hub for configuring the application's underlying parameters.

While this page in the demo is conceptual, it highlights important aspects of an AI agent system:

*   **API Keys:** Fields to securely input and manage API keys for Large Language Models (LLMs) from providers like OpenAI or Anthropic.
*   **Model Configuration:** Options to select default LLM models (e.g., `gpt-4-turbo`), embedding models (for understanding text context), and adjust parameters like `temperature` which influences the creativity/determinism of the AI.
*   **Memory Persistence:** Settings for how the AI's "memory" (e.g., document embeddings, workflow states) is stored, typically in a vector database like ChromaDB.
*   **Validation Thresholds:** Parameters for the minimum confidence required for an output to be considered valid, and retry limits for agent tasks.
*   **Logging:** Controls for the verbosity and format of internal logs, useful for developers and advanced users.

<aside class="info">
This page is for demonstration purposes to illustrate the types of configurations a robust AI agent application would offer. Its functionalities are not active in this codelab.
</aside>

## 9. Conclusion and Next Steps
Duration: 02:00

Congratulations! You have successfully navigated the core functionalities of QuLab: The FOMC Research Agent. You've learned how to:

*   Initiate new analyses of FOMC meetings.
*   Understand the key concepts of theme extraction, tone analysis, and policy surprise detection.
*   Interpret and review the detailed insights provided in the generated memos, emphasizing the critical role of citations for auditability.
*   Make informed review decisions (Approve, Request Revision, Reject).
*   Explore historical analyses and track trends over time.

QuLab empowers financial professionals to efficiently process complex, unstructured text from Federal Reserve communications, transforming it into actionable, auditable research. By embracing tools like QuLab, you can enhance your analytical capabilities and make more data-driven investment decisions.

Feel free to experiment with different meeting dates and analysis options to further explore the application's capabilities. Your feedback as a reviewer is invaluable in refining the AI agent's performance and ensuring the quality of the insights it generates.
