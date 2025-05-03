import runpod
import asyncio

from models import get_models

from logic_inference import (
    get_embeddings_logic
)
from query_segment import query_segment, remove_photo_prefix

# Cargar modelos necesarios para NLP y proximidad, excluyendo CLIP y Grounding DINO
get_models(
    only=[
        "embeddings_model"
    ],
    load_nltk=True
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

        elif operation == "get_emebddings":
            result = await asyncio.to_thread(
                get_embeddings_logic, data
            )

    except Exception as e:
        result = {"error": str(e)}

    return result


runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": (lambda current: 4),
})
