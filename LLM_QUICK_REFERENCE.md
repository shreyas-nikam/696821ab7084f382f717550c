# Quick Reference: Using LLM Structured Outputs

## Environment Setup

1. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="sk-..."
   # Or add to .env file:
   echo "OPENAI_API_KEY=sk-..." >> .env
   ```

2. **Verify setup:**
   ```bash
   python test_llm_structured_outputs.py
   ```

## Usage Examples

### Extract Themes from Documents

```python
import asyncio
from source import extract_themes, main_ingestion

async def analyze_themes():
    # Load documents
    documents = await main_ingestion(limit=2, recent_only=True)
    
    # Extract themes (uses real LLM)
    themes = await extract_themes(documents, n_themes=3)
    
    # Display results
    for theme in themes:
        print(f"\n{theme.theme_name} (confidence: {theme.confidence:.0%})")
        print(f"  {theme.description}")
        print(f"  Keywords: {', '.join(theme.keywords)}")
        print(f"  Citations: {len(theme.citations)}")

asyncio.run(analyze_themes())
```

### Compute Tone Score

```python
from source import compute_tone_score

async def analyze_tone():
    documents = await main_ingestion(limit=2, recent_only=True)
    
    # Compute tone (uses real LLM)
    tone = await compute_tone_score(documents)
    
    print(f"Overall Tone: {tone.score:+.2f}")
    print(f"Confidence: {tone.confidence:.0%}")
    print(f"\nComponents:")
    print(f"  Inflation: {tone.components.inflation_stance:+.2f}")
    print(f"  Employment: {tone.components.employment_stance:+.2f}")
    print(f"  Growth: {tone.components.growth_outlook:+.2f}")
    print(f"  Policy: {tone.components.policy_bias:+.2f}")

asyncio.run(analyze_tone())
```

### Detect Surprises

```python
from source import detect_surprises, search_historical_context

async def find_surprises():
    documents = await main_ingestion(limit=2, recent_only=True)
    
    # Get historical context
    historical = await search_historical_context(
        query="inflation policy",
        meeting_date=documents[0].meeting_date,
        n_results=5
    )
    
    # Mock prior tone for example
    from source import ToneScore, ToneComponents, Citation
    prior_tone = ToneScore(
        score=0.5,
        confidence=0.8,
        components=ToneComponents(
            inflation_stance=0.6,
            employment_stance=0.4,
            growth_outlook=0.3,
            policy_bias=0.7,
            uncertainty_level=0.2
        ),
        citations=[],
        explanation="Prior meeting was moderately hawkish"
    )
    
    # Detect surprises (uses real LLM)
    surprises = await detect_surprises(documents, historical, prior_tone)
    
    for surprise in surprises:
        print(f"\n{surprise.category.upper()}")
        print(f"  {surprise.description}")
        print(f"  Relevance: {surprise.market_relevance}")
        print(f"  Confidence: {surprise.confidence:.0%}")

asyncio.run(find_surprises())
```

## Response Models

### ThemesResponse
```python
class ThemeData(BaseModel):
    theme_name: str           # "Inflation Moderation"
    description: str          # Detailed description
    keywords: List[str]       # ["inflation", "prices", ...]
    citations: List[...]      # Supporting quotes
    confidence: float         # 0.0 to 1.0
```

### ToneAnalysisResponse
```python
class ToneAnalysisResponse(BaseModel):
    overall_score: float      # -1.0 (dovish) to +1.0 (hawkish)
    confidence: float         # 0.0 to 1.0
    components: ToneComponentsData
    citations: List[...]
    explanation: str
```

### SurprisesResponse
```python
class SurpriseData(BaseModel):
    surprise_id: str
    category: Literal[        # Type of surprise
        "policy_change",
        "language_shift",
        "forecast_revision",
        "dissent",
        "new_concern",
        "omission"
    ]
    description: str
    market_relevance: Literal["low", "medium", "high"]
    citations: List[...]
    confidence: float
```

## All Available Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `extract_themes()` | Extract key themes | `List[ThemeExtraction]` |
| `compute_tone_score()` | Hawkish/dovish analysis | `ToneScore` |
| `detect_surprises()` | Find unexpected elements | `List[Surprise]` |
| `check_hallucination()` | Validate claims | `ValidationCheck` |
| `generate_fomc_memo()` | Create full memo | `FOMCMemo` |

## Common Patterns

### Parallel Processing
```python
# Process multiple analyses in parallel
themes_task = extract_themes(documents)
tone_task = compute_tone_score(documents)

themes, tone = await asyncio.gather(themes_task, tone_task)
```

### Error Handling
```python
try:
    themes = await extract_themes(documents)
except Exception as e:
    logger.error(f"Theme extraction failed: {e}")
    themes = []  # Use empty list as fallback
```

### Custom Parameters
```python
# Extract more themes
themes = await extract_themes(documents, n_themes=10)

# With historical context for tone
tone = await compute_tone_score(documents, prior_scores=[...])
```

## Testing

### Quick Test
```bash
python test_llm_structured_outputs.py
```

### Manual Test
```python
from source import llm_client, ThemesResponse

async def test():
    response = await llm_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract themes."},
            {"role": "user", "content": "Inflation is high."}
        ],
        response_format=ThemesResponse,
        temperature=0.3
    )
    print(response.choices[0].message.parsed)

asyncio.run(test())
```

## Cost Estimates

Approximate costs for typical operations:

| Operation | Tokens | Cost |
|-----------|--------|------|
| Extract themes (3 themes) | ~2,000 | $0.02 |
| Compute tone score | ~1,500 | $0.015 |
| Detect surprises | ~2,500 | $0.025 |
| Full analysis | ~8,000 | $0.08 |

**Note:** Actual costs vary based on document length and complexity.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "API key not set" | Set `OPENAI_API_KEY` in environment |
| "Invalid schema" | Check Pydantic model definitions |
| "Rate limit" | Add delays between calls |
| "Timeout" | Increase timeout or reduce document size |
| "Model not found" | Verify access to gpt-4o-2024-08-06 |

## Best Practices

1. **Always handle exceptions** - API calls can fail
2. **Use reasonable limits** - Don't process too many documents at once
3. **Cache results** - LLM calls are expensive, cache when possible
4. **Monitor costs** - Track token usage in production
5. **Test with small datasets** - Validate before scaling up
6. **Use type hints** - Full benefit of structured outputs

## Additional Resources

- [LLM_STRUCTURED_OUTPUTS_SUMMARY.md](LLM_STRUCTURED_OUTPUTS_SUMMARY.md) - Detailed implementation docs
- [test_llm_structured_outputs.py](test_llm_structured_outputs.py) - Test suite
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)

---

*For questions or issues, refer to the full documentation in LLM_STRUCTURED_OUTPUTS_SUMMARY.md*
