# Case Study 1

The following is a platform designed as a prompt management system.  Since the problem is generally familiar, building a semantic search by calculating cosine similarity between text documents.  I started by building a quick FastAPI server and leveraged Pandas to build a in-memory columnar storage then utilized coding agents to fill in some of the gaps such as duplication detection by running search over the combinations within the dataset.  

The `hkunlp/instructor-base` embedding vectors perform well over instruction sets and the prompts provided are partial prompts.  The model performs well enough over this dataset that using more sophisticated chunking strategies did not seem necessary.  Otherwise, recommended strategies would be to use a hybrid approach combining concatenation, addition, or multiplication of the embeddings to improve performance.  Since prompts can solicit varying differences in behavior across different models, additional analysis would be necessary to determine the best approach.  

Concatenation in combination with paragraph level chunking could improve accuracy with prompts with similar structure but different content.  Addition of vectors with normalization could improve accuracy where similar sentences and phrases are present.  Multiplication of vectors could improve accuracy where prompts are very similar in structure and content.


## Usage

Setup requires `uv` for Python dependency management.  

```
uv sync
```

Running server

```
bin/serve
```

## API

Generates embeddings for all prompts; this must be to initialize the system

```
curl -X POST http://localhost:8000/api/embeddings/generate
```

Searches for prompts similar to a specific `prompt_id`

```
curl http://localhost:8000/api/prompts/survey.question.base/similar\?limit\=2 | jq
```

Retrieves similar prompts

```
curl -X POST http://localhost:8000/api/search/semantic -d "query=When"
```

Searches for prompts with semantic search

```
curl -X POST http://localhost:8000/api/search/semantic -H "Content-Type: application/json" -d '{"query":"When"}'
```

Retrieves a list of duplicate prompts with threshold

```
curl http://localhost:8000/api/analysis/duplicates
```
