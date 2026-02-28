# Case Study 2

First, setup generate synthetic data script.  Began noticing consistency issues with data schema and decided to stick with the technical requirements schema.  Agent utilized a normalization script to ensure supported format.  

Reviewed feature engineering as starting point.  Features were broken out into timing, speech patterns, and engagement.  Removed org_id as feature since it was not a useful predictor and would lead to overfitting.  

Added unit testing for feature engineering to catch regressions and support better iteration.  

Model section utilized RandomForestClassifier with sklearn which appears to be a solid baseline for an initial implementation.  

Reviewed model training and prediction.  


## Usage

Install

```
uv sync
```

Run

```bash
uv run uvicorn main:app --reload
```

Generate synthetic data

```bash
uv run python generate_synthetic_data.py
```

## API

Train a model

```bash
curl -X POST http://localhost:8000/api/model/train -H "Content-Type: application/json" -d "{\"training_data_path\": \"data/calls.json\"}"
```

Predict call outcome

```bash
curl -X POST -H "Content-Type: application/json" --json @data/input.json http://localhost:8000/api/predict 
```

Get model feature importance

```bash
curl http://localhost:8000/api/model/model_eb82e7cd/importance
```

## Tests

Run tests

```bash
uv run pytest -v
```
