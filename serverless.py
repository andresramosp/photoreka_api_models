import runpod
import asyncio
import base64
from io import BytesIO
from PIL import Image

from models import get_models

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


async def handler(job):
    input_data = job.get("input", {})
    operation = input_data.get("operation")
    data = input_data.get("data", {})

    if not operation:
        return {"error": "Missing 'operation' in input"}

    try:
        # Los modelos solo se cargan la primera vez
        get_models()

        if operation == "ping":
            return {"status": "warmed up"}

        if operation == "adjust_tags_proximities_by_context_inference":
            result = await asyncio.to_thread(
                adjust_tags_proximities_by_context_inference_logic, data
            )
        elif operation == "adjust_descs_proximities_by_context_inference":
            result = await asyncio.to_thread(
                adjust_descs_proximities_by_context_inference_logic, data
            )
        elif operation == "get_embeddings":
            result = await asyncio.to_thread(get_embeddings_logic, data)
        elif operation == "get_embeddings_image":
            result = await asyncio.to_thread(
                get_image_embeddings_from_base64, data.get("images", [])
            )
        elif operation == "detect_objects_base64":
            items = []
            for img_obj in data.get("images", []):
                img_data = base64.b64decode(img_obj["base64"])
                img = Image.open(BytesIO(img_data)).convert("RGB")
                items.append({"id": img_obj["id"], "image": img})

            result = await asyncio.to_thread(
                process_grounding_dino_detections_batched,
                items,
                data.get("categories", []),
            )
        elif operation == "query_segment":
            result = await asyncio.to_thread(
                query_segment, data.get("query", "")
            )
        elif operation == "query_no_prefix":
            result = await asyncio.to_thread(
                remove_photo_prefix, data.get("query", "")
            )
        elif operation == "clean_texts":
            result = await asyncio.to_thread(clean_texts, data)
        else:
            result = {"error": f"Unsupported operation: {operation}"}
    except Exception as e:
        result = {"error": str(e)}

    return result


#  Permite hasta 4 peticiones simultáneas por worker.
runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": (lambda current: 4),
    }
)
