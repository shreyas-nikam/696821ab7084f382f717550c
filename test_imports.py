#!/usr/bin/env python3
"""Test script to verify all imports and functions are available."""

import sys
print("Testing imports...")

try:
    from source import (
        SessionStateManager,
        get_pending_review_count,
        get_fomc_meeting_dates,
        check_document_availability,
        create_fomc_workflow,
        AgentState,
        get_pending_reviews,
        update_memo_status,
        load_memo,
        mock_prior_tone_scores,
        render_tone_trajectory,
        render_citation_network,
        load_source_documents,
    )
    print("✓ All required imports successful!")

    # Test basic functionality
    print("\nTesting basic functionality...")

    # Test SessionStateManager
    print("  - SessionStateManager: ", end="")
    manager = SessionStateManager()
    print("✓")

    # Test get_fomc_meeting_dates
    print("  - get_fomc_meeting_dates: ", end="")
    from datetime import date
    dates = get_fomc_meeting_dates(date(2023, 1, 1), date(2024, 12, 31))
    print(f"✓ (found {len(dates)} meetings)")

    # Test check_document_availability
    print("  - check_document_availability: ", end="")
    availability = check_document_availability(date(2024, 1, 31))
    print(f"✓ (checked {len(availability)} document types)")

    # Test create_fomc_workflow
    print("  - create_fomc_workflow: ", end="")
    workflow = create_fomc_workflow()
    print("✓")

    # Test mock_prior_tone_scores
    print("  - mock_prior_tone_scores: ", end="")
    print(f"✓ (found {len(mock_prior_tone_scores)} historical scores)")

    print("\n✅ All tests passed!")
    sys.exit(0)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
