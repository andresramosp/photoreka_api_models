from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
import asyncio
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, Form
from logic_inference import (
    adjust_tags_proximities_by_context_inference_logic,
    adjust_descs_proximities_by_context_inference_logic,
    get_embeddings_logic,
    clean_texts,
    generate_groups_for_tags,
    extract_tags_ntlk,
    extract_tags_spacy
)
from image_analyzer import (
    get_image_embeddings_from_base64, 
    process_grounding_dino_detections_batched, 
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

@app.post("/get_embeddings_image")
async def get_embeddings_image_endpoint(request: Request):
    data = await request.json()  # Espera {"images": [{id, base64}, ...]}
    items = data["images"]
    results = await asyncio.to_thread(get_image_embeddings_from_base64, items)
    return JSONResponse(content=results)

@app.post("/detect_objects_raw")
async def detect_objects_raw(
    files: list[UploadFile] = File(...),
    categories: list[str] = Form(...),
    min_box_size: int = Form(...)
):
    """
    Endpoint que recibe archivos raw (multipart/form-data) junto con:
      - "categories": [lista de strings]
      - "min_box_size": entero
    """
    items = []
    for file in files:
        id_ = file.filename
        content = await file.read()
        img = Image.open(BytesIO(content)).convert("RGB")
        items.append({"id": id_, "image": img})
    results = await asyncio.to_thread(process_grounding_dino_detections_batched, items, categories, min_box_size)
    return JSONResponse(content=results)

@app.post("/detect_objects_base64")
async def detect_objects_base64(request: Request):
    data = await request.json()  # Espera {"images": [{id, base64}, ...], "categories": [{string, min_box_size, max_box_area_ratio, merge}, ...]}

    images = data.get("images", [])
    categories = data.get("categories", [])

    if not categories:
        return JSONResponse(status_code=400, content={"error": "Missing 'categories'"})

    items = []
    for img_obj in images:
        try:
            img_data = base64.b64decode(img_obj["base64"])
            img = Image.open(BytesIO(img_data)).convert("RGB")
            items.append({"id": img_obj["id"], "image": img})
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Error decoding image {img_obj.get('id', '')}: {str(e)}"}
            )

    results = await asyncio.to_thread(
        process_grounding_dino_detections_batched,
        items,
        categories
    )
    return JSONResponse(content=results)

# @app.post("/generate_line_maps")
# async def generate_line_maps_endpoint(request: Request):
#     data = await request.json()  # Espera {"images": [{id, base64}, ...]}
#     items = data["images"]

#     def process_all(items):
#         results = []
#         for item in items:
#             presence_map, path, segments = generate_line_map(item["base64"], image_id=item["id"])
#             segments = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in segments]
#             results.append({
#                 "id": item["id"],
#                 "presence_map": presence_map.tolist(),
#                 "image_path": path,
#                 "segments": segments
#             })
#         return results

#     results = await asyncio.to_thread(process_all, items)
#     return JSONResponse(content={"results": results})

# @app.post("/generate_presence_maps")
# async def generate_presence_maps_endpoint(request: Request):
#     data = await request.json()  # Espera {"images": [{id, base64}, ...]}
#     items = data["images"]

#     def process_all(items):
#         results = []
#         for item in items:
#             presence_map, path = generate_presence_map(item["base64"], image_id=item["id"], method=3)
#             results.append({
#                 "id": item["id"],
#                 "presence_map": presence_map.tolist(),
#                 "image_path": path
#             })
#         return results

#     results = await asyncio.to_thread(process_all, items)
#     return JSONResponse(content={"results": results})

# @app.post("/find_similar_presence_maps")
# async def find_similar_presence_maps_endpoint(request: Request):
#     data = await request.json()
#     image_id = data["id"]
#     top_k = data.get("top_k", 100)
    
#     results = await asyncio.to_thread(
#         find_similar_line_maps_by_id,
#         image_id=image_id,
#         top_k=top_k
#     )
#     return JSONResponse(content={"results": results})

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
