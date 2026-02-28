Build an LLM evaluator.  

Automated evaluation is needed for:

1. Score responses on multiple dimensions (accuracy, empathy, conciseness) 
2. Compare prompt variants for A/B testing 
3. Flag problematic responses for human review 
4. Generate improvement suggestions for low-scoring responses 

Evaluator using an LLM as a judge pattern:

1. Design evaluation criteria for voice AI responses 
2. Implement multi-dimensional scoring using an LLM judge 
3. Build comparison tools for A/B prompt testing 
4. Create a feedback loop with improvement suggestions 

For a request with the schema:

```json
{
    "context": {
        "conversation_history": [],
        "current_directive": "Ask the user about their food security",
        "user_input": "Well, we sometimes run out of food before the end of the month"
    },
    "response": "I understand that can be really challenging. Just to clarify for our records - would you say that happens often, sometimes, o",
    "metadata": {
        "agent_id": "survey_agent",
        "prompt_version": "v2.1",
        "model": "gpt-4o-mini"
    }
}
```

Build a service with the following endpoints:

# POST /api/evaluate

```
curl -X POST http://localhost:8000/api/evaluate \
    -H "Content-Type: application/json" \
    -d @data/chat1.json
```

# POST /api/evaluate/batch

```
curl -X POST http://localhost:8000/api/evaluate/batch \
    -H "Content-Type: application/json" \
    -d @data/batch2.json
```

# POST /api/compare

```
curl -X POST http://localhost:8000/api/compare \
    -H "Content-Type: application/json" \
    -d @data/compare3.json
```

# POST /api/improve

```
curl -X POST http://localhost:8000/api/improve \
    -H "Content-Type: application/json" \
    -d @data/improve4.json
```
