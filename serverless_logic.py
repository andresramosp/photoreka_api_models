import runpod
import asyncio

from models import get_models

from logic_inference import (
    adjust_tags_proximities_by_context_inference_logic,
    adjust_descs_proximities_by_context_inference_logic,
    adjust_tags_proximities_by_context_inference_logic_modern,
    adjust_descs_proximities_by_context_inference_logic_modern,
    clean_texts
)
from query_segment import query_segment, remove_photo_prefix

# Cargar modelos necesarios para NLP y proximidad, excluyendo CLIP y Grounding DINO
get_models(
    only=[
        "roberta_classifier_text",
        "modern_ce_classifier",
        # "bart_classifier",
        # "nlp",
        # "ner_model",
        # "embeddings_model"
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

        elif operation == "adjust_tags_proximities_by_context_inference":
            result = await asyncio.to_thread(
                adjust_tags_proximities_by_context_inference_logic, data
            )

        elif operation == "adjust_descs_proximities_by_context_inference":
            result = await asyncio.to_thread(
                adjust_descs_proximities_by_context_inference_logic, data
            )

        elif operation == "adjust_tags_proximities_by_context_inference_modern":
            result = await asyncio.to_thread(
                adjust_tags_proximities_by_context_inference_logic_modern, data
            )

        elif operation == "adjust_descs_proximities_by_context_inference_modern":
            result = await asyncio.to_thread(
                adjust_descs_proximities_by_context_inference_logic_modern, data
            )

        # elif operation == "query_segment":
        #     result = await asyncio.to_thread(
        #         query_segment, data.get("query", "")
        #     )

        # elif operation == "query_no_prefix":
        #     result = await asyncio.to_thread(
        #         remove_photo_prefix, data.get("query", "")
        #     )

        elif operation == "clean_texts":
            result = await asyncio.to_thread(clean_texts, data)

        else:
            result = {"error": f"Unsupported operation: {operation}"}

    except Exception as e:
        result = {"error": str(e)}

    return result


runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": (lambda current: 4),
})
