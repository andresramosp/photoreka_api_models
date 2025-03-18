from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncio
import uvicorn
from models import MODELS
from logic_inference import (
    adjust_tags_proximities_by_context_inference_logic,
    adjust_descs_proximities_by_context_inference_logic,
    get_embeddings_logic,
    clean_texts,
    generate_groups_for_tags,
    extract_tags_ntlk,
    extract_tags_spacy
)
from query_segment import query_segment, remove_photo_prefix

app = FastAPI()

@app.post("/adjust_tags_proximities_by_context_inference")
async def adjust_tags_endpoint(request: Request):
    data = await request.json()
    results = await asyncio.to_thread(adjust_tags_proximities_by_context_inference_logic, data)
    return JSONResponse(content=results)

@app.post("/adjust_descs_proximities_by_context_inference")
async def adjust_descs_endpoint(request: Request):
    data = await request.json()
    results = await asyncio.to_thread(adjust_descs_proximities_by_context_inference_logic, data)
    return JSONResponse(content=results)

@app.post("/get_embeddings")
async def get_embeddings_endpoint(request: Request):
    data = await request.json()
    results = await asyncio.to_thread(get_embeddings_logic, data)
    return JSONResponse(content=results)

@app.post("/query_segment")
async def query_segment_endpoint(request: Request):
    data = await request.json()
    result = query_segment(data["query"])
    return JSONResponse(content=result)

@app.post("/query_no_prefix")
async def query_no_prefix_endpoint(request: Request):
    data = await request.json()
    result = remove_photo_prefix(data["query"])
    return JSONResponse(content=result)

@app.post("/clean_texts")
async def clean_texts_endpoint(request: Request):
    data = await request.json()
    result = clean_texts(data)
    return JSONResponse(content=result)

@app.post("/generate_groups_for_tags")
async def generate_groups_for_tags_endpoint(request: Request):
    data = await request.json()
    result = generate_groups_for_tags(data)
    return JSONResponse(content=result)

@app.post("/extract_tags")
async def extract_tags_endpoint(request: Request):
    data = await request.json()
    if data.get("method") == "spacy":
        result = extract_tags_spacy(data.get("text"), data.get("allowed_groups"))
    elif data.get("method") == "ntlk":
        result = extract_tags_ntlk(data.get("text"))
    else:
        result = {"error": "Método no soportado"}
    return JSONResponse(content=result)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=5000, reload=True, access_log=False)
