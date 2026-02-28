# Case Study 3

## Usage

Install

```
uv sync
```

Run

```bash
uv run uvicorn main:app --reload
```

## API

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
