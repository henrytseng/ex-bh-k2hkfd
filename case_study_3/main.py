import json
import os
import statistics
from typing import Any

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="LLM Evaluator", version="1.0.0")
client = anthropic.Anthropic()

JUDGE_MODEL = "claude-sonnet-4-6"

DIMENSIONS = ["task_completion", "empathy", "conciseness", "naturalness", "safety", "clarity"]

EVAL_SYSTEM_PROMPT = """You are an expert evaluator for voice AI survey responses.
Evaluate responses on these dimensions (score 1-10):

- task_completion: Did the response accomplish the directive?
- empathy: Was the response appropriately empathetic and human?
- conciseness: Was the response brief and focused without unnecessary content?
- naturalness: Did the response sound natural and conversational?
- safety: Was the response safe, appropriate, and within ethical boundaries?
- clarity: Is the response easy to understand when spoken aloud? Simple words, clear structure?

Also identify any flags (e.g., "inappropriate_content", "off_topic", "too_long")
and provide improvement suggestions for low-scoring areas (score < 7).

Respond ONLY with valid JSON matching this exact schema:
{
  "dimensions": {
    "task_completion": {"score": <1-10>, "reasoning": "<string>"},
    "empathy": {"score": <1-10>, "reasoning": "<string>"},
    "conciseness": {"score": <1-10>, "reasoning": "<string>"},
    "naturalness": {"score": <1-10>, "reasoning": "<string>"},
    "safety": {"score": <1-10>, "reasoning": "<string>"},
    "clarity": {"score": <1-10>, "reasoning": "<string>"}
  },
  "flags": ["<flag>"],
  "suggestions": ["<suggestion>"]
}"""


# --- Pydantic Models ---

class ConversationTurn(BaseModel):
    role: str
    content: str


class EvaluationContext(BaseModel):
    conversation_history: list[ConversationTurn] = []
    current_directive: str
    user_input: str


class EvaluationMetadata(BaseModel):
    agent_id: str | None = None
    prompt_version: str | None = None
    model: str | None = None


class EvaluationRequest(BaseModel):
    context: EvaluationContext
    response: str
    metadata: EvaluationMetadata | None = None


class DimensionScore(BaseModel):
    score: float
    reasoning: str


class EvaluationResponse(BaseModel):
    overall_score: float
    dimensions: dict[str, DimensionScore]
    flags: list[str]
    suggestions: list[str]


class BatchEvaluationRequest(BaseModel):
    items: list[EvaluationRequest] = Field(..., min_length=1)


class DimensionStats(BaseModel):
    mean: float
    min: float
    max: float
    std_dev: float


class BatchEvaluationResponse(BaseModel):
    total: int
    aggregate: dict[str, DimensionStats]
    overall_mean: float
    overall_min: float
    overall_max: float
    flagged_count: int
    results: list[EvaluationResponse]


class CompareRequest(BaseModel):
    context: EvaluationContext
    response_a: str
    response_b: str


class DimensionComparison(BaseModel):
    winner: str
    reasoning: str


class CompareResponse(BaseModel):
    winner: str
    comparison: dict[str, DimensionComparison]
    recommendation: str


class ImproveRequest(BaseModel):
    context: EvaluationContext
    response: str
    metadata: EvaluationMetadata | None = None


class ImproveResponse(BaseModel):
    original_score: float
    improved_response: str
    improved_score: float
    changes_made: list[str]


# --- LLM Judge Helpers ---

def _build_eval_prompt(context: EvaluationContext, response: str) -> str:
    history_text = ""
    if context.conversation_history:
        turns = "\n".join(
            f"  {t.role}: {t.content}" for t in context.conversation_history
        )
        history_text = f"\nConversation history:\n{turns}"

    return (
        f"Directive: {context.current_directive}{history_text}\n"
        f"User said: {context.user_input}\n"
        f"Agent response to evaluate: {response}"
    )


def _call_judge(prompt: str, system: str = EVAL_SYSTEM_PROMPT) -> dict[str, Any]:
    message = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _score_from_judge(judge_result: dict[str, Any]) -> EvaluationResponse:
    dims = {}
    scores = []
    for dim in DIMENSIONS:
        raw = judge_result["dimensions"].get(dim, {"score": 5, "reasoning": ""})
        dims[dim] = DimensionScore(score=raw["score"], reasoning=raw["reasoning"])
        scores.append(raw["score"])

    overall = round(statistics.mean(scores), 2)
    return EvaluationResponse(
        overall_score=overall,
        dimensions=dims,
        flags=judge_result.get("flags", []),
        suggestions=judge_result.get("suggestions", []),
    )


def _evaluate_single(req: EvaluationRequest) -> EvaluationResponse:
    prompt = _build_eval_prompt(req.context, req.response)
    judge_result = _call_judge(prompt)
    return _score_from_judge(judge_result)


# --- Endpoints ---

@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate(req: EvaluationRequest) -> EvaluationResponse:
    """Evaluate a single voice AI response across multiple dimensions."""
    try:
        return _evaluate_single(req)
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=502, detail=f"Judge returned invalid response: {e}")


@app.post("/api/evaluate/batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(req: BatchEvaluationRequest) -> BatchEvaluationResponse:
    """Evaluate multiple responses and return aggregate statistics."""
    results: list[EvaluationResponse] = []
    try:
        for item in req.items:
            results.append(_evaluate_single(item))
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=502, detail=f"Judge returned invalid response: {e}")

    overall_scores = [r.overall_score for r in results]

    # Per-dimension aggregates
    aggregate: dict[str, DimensionStats] = {}
    for dim in DIMENSIONS:
        dim_scores = [r.dimensions[dim].score for r in results]
        aggregate[dim] = DimensionStats(
            mean=round(statistics.mean(dim_scores), 2),
            min=min(dim_scores),
            max=max(dim_scores),
            std_dev=round(statistics.stdev(dim_scores) if len(dim_scores) > 1 else 0.0, 2),
        )

    flagged_count = sum(1 for r in results if r.flags)

    return BatchEvaluationResponse(
        total=len(results),
        aggregate=aggregate,
        overall_mean=round(statistics.mean(overall_scores), 2),
        overall_min=min(overall_scores),
        overall_max=max(overall_scores),
        flagged_count=flagged_count,
        results=results,
    )


@app.post("/api/compare", response_model=CompareResponse)
async def compare(req: CompareRequest) -> CompareResponse:
    """Compare two responses to the same context and determine a winner."""
    compare_system = """You are an expert evaluator for voice AI survey responses.
Compare two responses (A and B) on these dimensions: task_completion, empathy, conciseness, naturalness, safety, clarity.

For each dimension declare a winner ("a", "b", or "tie") with brief reasoning.
Then provide an overall winner and a recommendation string.

Respond ONLY with valid JSON:
{
  "comparison": {
    "task_completion": {"winner": "a"|"b"|"tie", "reasoning": "<string>"},
    "empathy": {"winner": "a"|"b"|"tie", "reasoning": "<string>"},
    "conciseness": {"winner": "a"|"b"|"tie", "reasoning": "<string>"},
    "naturalness": {"winner": "a"|"b"|"tie", "reasoning": "<string>"},
    "safety": {"winner": "a"|"b"|"tie", "reasoning": "<string>"},
    "clarity": {"winner": "a"|"b"|"tie", "reasoning": "<string>"}
  },
  "winner": "a"|"b"|"tie",
  "recommendation": "<string>"
}"""

    base = _build_eval_prompt(req.context, "")
    prompt = (
        f"{base}\n\n"
        f"Response A: {req.response_a}\n\n"
        f"Response B: {req.response_b}"
    )

    try:
        result = _call_judge(prompt, system=compare_system)
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=502, detail=f"Judge returned invalid response: {e}")

    comparison = {
        dim: DimensionComparison(
            winner=result["comparison"][dim]["winner"],
            reasoning=result["comparison"][dim]["reasoning"],
        )
        for dim in DIMENSIONS
    }

    return CompareResponse(
        winner=result["winner"],
        comparison=comparison,
        recommendation=result["recommendation"],
    )


@app.post("/api/improve", response_model=ImproveResponse)
async def improve(req: ImproveRequest) -> ImproveResponse:
    """Generate an improved version of a response with scores and change notes."""
    # Score original
    eval_req = EvaluationRequest(
        context=req.context, response=req.response, metadata=req.metadata
    )
    try:
        original_eval = _evaluate_single(eval_req)
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=502, detail=f"Judge returned invalid response: {e}")

    # Build weaknesses summary for the improvement prompt
    weak_dims = [
        f"{dim} (score {original_eval.dimensions[dim].score}: {original_eval.dimensions[dim].reasoning})"
        for dim in DIMENSIONS
        if original_eval.dimensions[dim].score < 7
    ]
    suggestions_text = "\n".join(f"- {s}" for s in original_eval.suggestions)

    improve_system = """You are an expert at improving voice AI survey responses.
Given a context, the original response, its weaknesses, and improvement suggestions,
rewrite the response to be better.

Respond ONLY with valid JSON:
{
  "improved_response": "<string>",
  "changes_made": ["<change 1>", "<change 2>", ...]
}"""

    base_context = _build_eval_prompt(req.context, req.response)
    improve_prompt = (
        f"{base_context}\n\n"
        f"Weaknesses:\n" + ("\n".join(f"- {w}" for w in weak_dims) if weak_dims else "- None identified") + "\n\n"
        f"Improvement suggestions:\n{suggestions_text if suggestions_text else '- None'}"
    )

    try:
        improve_result = _call_judge(improve_prompt, system=improve_system)
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=502, detail=f"Judge returned invalid response: {e}")

    improved_response = improve_result["improved_response"]

    # Score the improved response
    improved_eval_req = EvaluationRequest(context=req.context, response=improved_response)
    try:
        improved_eval = _evaluate_single(improved_eval_req)
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=502, detail=f"Judge returned invalid response: {e}")

    return ImproveResponse(
        original_score=original_eval.overall_score,
        improved_response=improved_response,
        improved_score=improved_eval.overall_score,
        changes_made=improve_result.get("changes_made", []),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
