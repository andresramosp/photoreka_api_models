import runpod
import asyncio
import base64
from io import BytesIO
from PIL import Image

from models import get_models
from image_analyzer import (
    get_image_embeddings_from_base64,
    process_grounding_dino_detections_batched,
)

# Inicializa solo los modelos necesarios
get_models(
    only=["clip_model", "clip_preprocess", "gdino_model", "gdino_processor"],
    load_nltk=False
)

async def handler(job):
    input_data = job.get("input", {})
    operation = input_data.get("operation")
    data = input_data.get("data", {})

    if not operation:
        return {"error": "Missing 'operation' in input"}

    try:
        if operation == "ping":
            return {"status": "warmed up"}

        if operation == "get_embeddings_image":
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
        else:
            result = {"error": f"Unsupported operation: {operation}"}
    except Exception as e:
        result = {"error": str(e)}

    return result


# Permite hasta 4 peticiones simultáneas por worker.
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": (lambda current: 4),
})
