import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import torch
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
import uvicorn
import nltk
from nltk.stem import WordNetLemmatizer
import inflect
import time
import spacy
from spacy.matcher import Matcher
from cachetools import TTLCache
import re

app = FastAPI()
cache = TTLCache(maxsize=200000, ttl=3600)

# Funciones de inicialización y helpers (se mantienen igual)
def load_wordnet():
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

def load_embeddings_model():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    device = 0 if torch.cuda.is_available() else -1
    embeddings_model = SentenceTransformer('all-mpnet-base-v2', device=device)
    roberta_classifier_text = pipeline(
        "text-classification", 
        model="roberta-large-mnli", 
        tokenizer=AutoTokenizer.from_pretrained("roberta-large-mnli", use_fast=True), 
        device=device
    )
    ner_model = pipeline("ner", model="FacebookAI/xlm-roberta-large-finetuned-conll03-english", aggregation_strategy="simple")
    nlp = spacy.load("en_core_web_sm")
    return embeddings_model, roberta_classifier_text, nlp, ner_model

def preprocess_text(text, to_singular=False):
    lemmatizer = WordNetLemmatizer()
    p = inflect.engine()
    normalized_text = text.lower().replace('_', ' ')
    words = normalized_text.split()
    
    if len(words) == 1:
        lemmatized_word = lemmatizer.lemmatize(normalized_text)
        return p.singular_noun(lemmatized_word) or lemmatized_word if to_singular else lemmatized_word
    
    if to_singular:
        words = [p.singular_noun(word) or word for word in words]
    
    return ' '.join(words)

def cached_inference(batch_queries, batch_size):
    cached_results = []
    queries_to_infer = []
    indexes_to_infer = []

    for i, query in enumerate(batch_queries):
        if query in cache:
            cached_results.append(cache[query])
        else:
            queries_to_infer.append(query)
            indexes_to_infer.append(i)

    if queries_to_infer:
        dataset = Dataset.from_dict({"query": queries_to_infer})
        batch_results = roberta_classifier_text(dataset["query"], batch_size=batch_size)
        for i, result in zip(indexes_to_infer, batch_results):
            cache[batch_queries[i]] = result
            cached_results.insert(i, result)
    return cached_results

# Función auxiliar movida fuera del endpoint
def combine_tag_name_with_group(tag):
    if tag.get("group") == "symbols":
        return f"{tag['name']} (symbol or sign)"
    if tag.get("group") == "culture":
        return f"{tag['name']} culture"
    if tag.get("group") == "location":
        return f"{tag['name']} (place)"
    if tag.get("group") == "generic":
        return f"{tag['name']} (as general topic)"
    if tag.get("group") == "objects":
        return f"{tag['name']} (physical thing)"
    return tag["name"]

# --- Funciones de lógica para cada operación ---

def adjust_tags_proximities_by_context_inference_logic(data: dict):
    start_time = time.perf_counter()
    BATCH_SIZE = 128
    THRESHOLD = 0.82

    term = preprocess_text(data.get("term", ""), True)
    tag_list = data.get("tag_list", [])
    premise_wrapper = data.get("premise_wrapper", "The photo featured {term}")
    hypothesis_wrapper = data.get("hypothesis_wrapper", "The photo featured {term}")

    if not term or not tag_list:
        raise ValueError("Missing required fields (term, tag_list)")

    batch_queries = [
        f"{premise_wrapper.format(term=preprocess_text(combine_tag_name_with_group(tag)))} [SEP] {hypothesis_wrapper.format(term=term)}"
        for tag in tag_list
    ]
    tag_names = [tag['name'] for tag in tag_list]
    batch_results = cached_inference(batch_queries, BATCH_SIZE)

    results = {}
    for tag_name, result in zip(tag_names, batch_results):
        label = result["label"].lower()
        score = result["score"]
        adjusted_score = score if label == "entailment" and score >= THRESHOLD else -score if label == "contradiction" else 0
        results[tag_name] = {"adjusted_proximity": adjusted_score, "label": label, "score": score}
        if label == "entailment" and score >= THRESHOLD:
            print(f"✅ [TAG MATCH] {tag_name} -> {term}: {label.upper()} con score {score:.4f}")

    print(f"⏳ Tiempo de ejecución: {time.perf_counter() - start_time:.4f} segundos")
    return results

def adjust_descs_proximities_by_context_inference_logic(data: dict):
    BATCH_SIZE = 128
    THRESHOLD = 0.55

    term = preprocess_text(data.get("term", ""), True)
    chunk_list = data.get("tag_list", [])
    premise_wrapper = data.get("premise_wrapper", "the photo has the following fragment in its description: '{term}'")
    hypothesis_wrapper = data.get("hypothesis_wrapper", "the photo features {term}")

    if not term or not chunk_list:
        raise ValueError("Missing required fields (term, tag_list)")

    batch_queries = [
        f"{premise_wrapper.format(term=chunk['name'])} [SEP] {hypothesis_wrapper.format(term=term)}"
        for chunk in chunk_list
    ]
    chunk_names = [chunk['name'] for chunk in chunk_list]
    batch_results = cached_inference(batch_queries, BATCH_SIZE)

    results = {}
    for chunk_name, result in zip(chunk_names, batch_results):
        label = result["label"].lower()
        score = result["score"]
        adjusted_score = score if label == "entailment" and score >= THRESHOLD else -score if label == "contradiction" else 0
        results[chunk_name] = {"adjusted_proximity": adjusted_score, "label": label, "score": score}
        if label == "entailment" and score >= THRESHOLD:
            print(f"✅ [DESC MATCH] {chunk_name} -> {term}: {label.upper()} con score {score:.4f}")

    return results

def get_embeddings_logic(data: dict):
    tags = data.get("tags", [])
    if not tags or not isinstance(tags, list):
        raise ValueError("Field 'tags' must be a list.")
    embeddings = embeddings_model.encode(tags, convert_to_tensor=False)
    return {"tags": tags, "embeddings": [emb.tolist() for emb in embeddings]}

def structure_query_logic(data: dict):
    query = data.get("query", "").strip()
    if not query:
        raise ValueError("Missing 'query' field")
    print(f"📥 Received query: {query}")
    structured_query_str, types, positive_segments, negative_segments, query_no_prefix = segment_query(query)
    print(f"📤 Generated response: {structured_query_str}")
    return {
        "clear": structured_query_str,
        "no_prefix": query_no_prefix,
        "types": types,
        "positive_segments": positive_segments,
        "negative_segments": negative_segments
    }

# --- Endpoints de FastAPI ---

@app.post("/adjust_tags_proximities_by_context_inference")
async def adjust_tags_proximities_by_context_inference(request: Request):
    data = await request.json()
    results = adjust_tags_proximities_by_context_inference_logic(data)
    return JSONResponse(content=results)

@app.post("/adjust_descs_proximities_by_context_inference")
async def adjust_descs_proximities_by_context_inference(request: Request):
    data = await request.json()
    results = adjust_descs_proximities_by_context_inference_logic(data)
    return JSONResponse(content=results)

@app.post("/get_embeddings")
async def get_embeddings(request: Request):
    data = await request.json()
    results = get_embeddings_logic(data)
    return JSONResponse(content=results)

@app.post("/structure_query")
async def structure_query(request: Request):
    data = await request.json()
    results = structure_query_logic(data)
    return JSONResponse(content=results)

# Inicializar recursos
load_wordnet()
embeddings_model, roberta_classifier_text, nlp, ner_model = load_embeddings_model()

# --- Configuración para RunPod Serverless ---
if os.getenv("RUNPOD_SERVERLESS", "false").lower() == "true":
    import runpod

    def handler(job):
        input_data = job.get("input", {})
        operation = input_data.get("operation")
        if not operation:
            return {"error": "Missing 'operation' in input"}
        try:
            if operation == "adjust_tags_proximities_by_context_inference":
                data = input_data.get("data", {})
                result = adjust_tags_proximities_by_context_inference_logic(data)
            elif operation == "adjust_descs_proximities_by_context_inference":
                data = input_data.get("data", {})
                result = adjust_descs_proximities_by_context_inference_logic(data)
            elif operation == "get_embeddings":
                data = input_data.get("data", {})
                result = get_embeddings_logic(data)
            elif operation == "structure_query":
                data = input_data.get("data", {})
                result = structure_query_logic(data)
            else:
                result = {"error": f"Operation '{operation}' not supported"}
        except Exception as e:
            result = {"error": str(e)}
        return result

    runpod.serverless.start({"handler": handler})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
