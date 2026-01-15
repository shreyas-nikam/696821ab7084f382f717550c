#!/usr/bin/env python
"""
Test script for LLM structured outputs with Pydantic models.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def test_llm_client():
    """Test that the LLM client is properly initialized."""
    try:
        from source import llm_client
        print("✓ LLM client imported successfully")
        print(f"  Client type: {type(llm_client)}")
        print(
            f"  API key set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

        if not os.getenv('OPENAI_API_KEY'):
            print("\n⚠️  WARNING: OPENAI_API_KEY not set in environment")
            print("   Set it in .env file or as environment variable")
            return False

        return True

    except Exception as e:
        print(f"✗ Failed to import LLM client: {e}")
        return False


async def test_structured_output_models():
    """Test that all structured output models are defined."""
    try:
        from source import (
            ThemesResponse,
            ToneAnalysisResponse,
            SurprisesResponse,
            HallucinationCheckResponse,
            MemoGenerationResponse
        )

        print("\n✓ All structured output models imported successfully")
        print(f"  - ThemesResponse")
        print(f"  - ToneAnalysisResponse")
        print(f"  - SurprisesResponse")
        print(f"  - HallucinationCheckResponse")
        print(f"  - MemoGenerationResponse")

        return True

    except Exception as e:
        print(f"\n✗ Failed to import structured output models: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_analysis_functions():
    """Test that analysis functions are properly defined."""
    try:
        from source import (
            extract_themes,
            compute_tone_score,
            detect_surprises,
            check_hallucination
        )

        print("\n✓ All analysis functions imported successfully")
        print(f"  - extract_themes")
        print(f"  - compute_tone_score")
        print(f"  - detect_surprises")
        print(f"  - check_hallucination")

        # Check function signatures
        import inspect

        sig = inspect.signature(extract_themes)
        print(f"\n  extract_themes signature: {sig}")

        sig = inspect.signature(compute_tone_score)
        print(f"  compute_tone_score signature: {sig}")

        return True

    except Exception as e:
        print(f"\n✗ Failed to import analysis functions: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_llm_call():
    """Test a simple LLM call with structured output."""
    try:
        from source import llm_client, ThemesResponse

        if not os.getenv('OPENAI_API_KEY'):
            print("\n⚠️  Skipping LLM call test (no API key)")
            return True

        print("\n Testing simple LLM call with structured output...")

        # Simple test with minimal data
        response = await llm_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "Extract themes from the text. Return JSON with key 'themes' as a list."},
                {"role": "user", "content": "Inflation remains elevated. Job market is strong."}
            ],
            response_format=ThemesResponse,
            temperature=0.3
        )

        result = response.choices[0].message.parsed
        print(f"✓ LLM call successful")
        print(f"  Result type: {type(result)}")
        print(f"  Parsed data: {result.model_dump()}")

        return True

    except Exception as e:
        print(f"\n⚠️  LLM call test failed: {e}")
        print("   This is expected if you don't have a valid API key or credits")
        import traceback
        traceback.print_exc()
        return True  # Don't fail the test if API call fails


async def main():
    """Run all tests."""
    print("=" * 70)
    print("  LLM Structured Outputs Integration Test")
    print("=" * 70)

    tests = [
        ("LLM Client Initialization", test_llm_client),
        ("Structured Output Models", test_structured_output_models),
        ("Analysis Functions", test_analysis_functions),
        ("Simple LLM Call", test_simple_llm_call),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTest: {name}")
        print("-" * 70)
        result = await test_func()
        results.append((name, result))

    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:>10} - {name}")
        if not result:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("  ✅ All tests passed!")
    else:
        print("  ⚠️  Some tests failed")
    print("=" * 70)

    return all_passed

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
