import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import torch
import nltk
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
import spacy
from cachetools import TTLCache

# Carga de dependencias comunes
def load_wordnet():
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

load_wordnet()

def load_common_models():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_model = SentenceTransformer('all-mpnet-base-v2', device=device)
    roberta_classifier_text = pipeline(
        "text-classification",
        model="roberta-large-mnli",
        tokenizer=AutoTokenizer.from_pretrained("roberta-large-mnli", use_fast=True),
        device=0 if device == "cuda" else -1
    )
    ner_model = pipeline("ner", model="FacebookAI/xlm-roberta-large-finetuned-conll03-english", aggregation_strategy="simple", device=0 if device == "cuda" else -1)
    nlp = spacy.load("en_core_web_sm")
    return embeddings_model, roberta_classifier_text, nlp, ner_model

embeddings_model, roberta_classifier_text, nlp, ner_model = load_common_models()
cache = TTLCache(maxsize=200000, ttl=3600)

# Importar la lógica de inferencia desde el archivo externo
from logic_inference import (
    adjust_tags_proximities_by_context_inference_logic,
    adjust_descs_proximities_by_context_inference_logic,
    get_embeddings_logic
)

# Importar funciones de segmentación de query desde query_segment.py
from query_segment import query_segment

app = FastAPI()

@app.post("/adjust_tags_proximities_by_context_inference")
async def adjust_tags_endpoint(request: Request):
    data = await request.json()
    results = adjust_tags_proximities_by_context_inference_logic(data)
    return JSONResponse(content=results)

@app.post("/adjust_descs_proximities_by_context_inference")
async def adjust_descs_endpoint(request: Request):
    data = await request.json()
    results = adjust_descs_proximities_by_context_inference_logic(data)
    return JSONResponse(content=results)

@app.post("/get_embeddings")
async def get_embeddings_endpoint(request: Request):
    data = await request.json()
    results = get_embeddings_logic(data)
    return JSONResponse(content=results)

@app.post("/query_segment")
async def query_segment_endpoint(request: Request):
    data = await request.json()
    result = query_segment(data["query"])
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
