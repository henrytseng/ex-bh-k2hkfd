import json
import logging
import pandas as pd
import numpy as np
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Constants
PREVIEW_LENGTH = 20

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup embeddings model
model = SentenceTransformer("hkunlp/instructor-base")

# HTTP Request parameters
class RequestSemanticSearch(BaseModel):
    query: str
    limit: int = 10

def load_dataset():
    """Load dataset from JSON file and return as pandas DataFrame."""
    with open("data/prompts.json", "r") as f:
        dataset = json.load(f)
        df = pd.DataFrame(dataset.get('prompts'))
        
        df["uid"] = [str(uuid.uuid4()) for _ in range(len(df))]
        logger.info(f"Loaded {len(df)} prompts into the vector database {df.columns}")
    return df

# In-memory column store
knowledgebase_df = load_dataset()

# Initialize FastAPI
app = FastAPI()

def search_dataset(query: str, limit: int = 10, filtered_df: pd.DataFrame = None):
    """Search the dataset for similar prompts."""
    limit = max(0, limit)

    # Use entire dataset by default
    filtered_df = knowledgebase_df.copy() if filtered_df is None else filtered_df.copy()

    # Calculate embedding for query
    query_embedding = model.encode(query)

    # Calculate distance between query and all embeddings
    distances = filtered_df['embedding'].apply(lambda x: model.similarity(query_embedding, x))
    sorted_distances = distances.sort_values(ascending=False)
    
    results_df = filtered_df.loc[sorted_distances.index][:limit]
    results_df['distance'] = sorted_distances.astype(float).values[:limit]
    
    return results_df
    

@app.get("/")
async def root():
    return {"message": "Hello"}

@app.post('/api/embeddings/generate')
async def generate_embeddings():
    # Generate embeddings
    embedding = model.encode(knowledgebase_df['content'].tolist())
    knowledgebase_df['embedding'] = embedding.tolist()
    logger.info("Generated embeddings successfully. Storing in vector database...")

    return {"message": "Embeddings generated and stored successfully"}

@app.get('/api/prompts/{prompt_id}/similar')
async def get_similar_prompts(prompt_id: str, limit: int = 5, threshold: float = 0.8):
    try:
        selected_prompt = knowledgebase_df[knowledgebase_df['prompt_id'] == prompt_id]

        # If prompt is not found, return empty list
        if selected_prompt.empty:
            raise HTTPException(status_code=404, detail="Not found")

        # Search for similar
        results_df = search_dataset(
            query=selected_prompt['content'].iloc[0],
            limit=limit,
            filtered_df=knowledgebase_df[knowledgebase_df['prompt_id'] != prompt_id]
        )
        
        # Return preview of contents
        results_df = results_df[['prompt_id', 'content', 'distance']]
        results_df['content'] = results_df['content'].str[:PREVIEW_LENGTH]
        
        return results_df.to_dict(orient='records')

    except KeyError:
        raise HTTPException(status_code=400, detail="No embeddings found. Please generate embeddings first.")

@app.post('/api/search/semantic')
async def search_semantic(request: RequestSemanticSearch):
    try:
        results_df = search_dataset(
            query=request.query,
            limit=request.limit
        )
        return results_df['content'].tolist()

    except KeyError:
        raise HTTPException(status_code=400, detail="No embeddings found. Please generate embeddings first.")

@app.get('/api/analysis/duplicates')
async def get_duplicates(threshold: float = 0.9):
    try:
        num_i, _ = knowledgebase_df.shape

        # Calculate pairwise similarity between all prompts
        pairwise_similarity = [model.similarity(x, knowledgebase_df['embedding']) for x in knowledgebase_df['embedding'].tolist()]

        # Use numpy to access array
        pairwise_similarity = np.array(pairwise_similarity).reshape(num_i, num_i)
        
        # Find duplicates
        duplicates = []
        for i in range(num_i):
            cluster_id = knowledgebase_df['prompt_id'].iloc[i]
            prompts = []
            for j in range(i + 1, num_i):
                if pairwise_similarity[i][j] > threshold:
                    prompts.append({
                        'prompt_id': knowledgebase_df['prompt_id'].iloc[i],
                        'content': knowledgebase_df['content'].iloc[i],
                        'distance': float(pairwise_similarity[i][j])
                    })
            if len(prompts) > 0:
                duplicates.append({
                    'cluster_id': cluster_id,
                    'prompts': prompts
                })
        return duplicates

    except KeyError:
        raise HTTPException(status_code=400, detail="No embeddings found. Please generate embeddings first.")