import runpod
import base64
from io import BytesIO
from PIL import Image

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

# Handler principal
async def handler(job):
    input_data = job.get("input", {})
    operation = input_data.get("operation")

    if not operation:
        return {"error": "Missing 'operation' in input"}

    try:
        if operation == "adjust_tags_proximities_by_context_inference":
            result = adjust_tags_proximities_by_context_inference_logic(input_data.get("data", {}))
        elif operation == "adjust_descs_proximities_by_context_inference":
            result = adjust_descs_proximities_by_context_inference_logic(input_data.get("data", {}))
        elif operation == "get_embeddings":
            result = get_embeddings_logic(input_data.get("data", {}))
        elif operation == "get_embeddings_image":
            result = get_image_embeddings_from_base64(input_data.get("images", []))
        elif operation == "detect_objects_base64":
            images = input_data.get("images", [])
            categories = input_data.get("categories", [])
            items = []
            for img_obj in images:
                img_data = base64.b64decode(img_obj["base64"])
                img = Image.open(BytesIO(img_data)).convert("RGB")
                items.append({"id": img_obj["id"], "image": img})
            result = process_grounding_dino_detections_batched(items, categories)
        elif operation == "query_segment":
            result = query_segment(input_data.get("query"))
        elif operation == "query_no_prefix":
            result = remove_photo_prefix(input_data.get("query"))
        elif operation == "clean_texts":
            result = clean_texts(input_data.get("data", []))
        elif operation == "generate_groups_for_tags":
            result = generate_groups_for_tags(input_data.get("data", []))
        elif operation == "extract_tags":
            text = input_data.get("text", "")
            method = input_data.get("method", "spacy")
            if method == "spacy":
                result = extract_tags_spacy(text, input_data.get("allowed_groups", []))
            elif method == "ntlk":
                result = extract_tags_ntlk(text)
            else:
                result = {"error": "Unsupported method"}
        else:
            result = {"error": f"Unsupported operation: {operation}"}
    except Exception as e:
        result = {"error": str(e)}

    return result

runpod.serverless.start({"handler": handler})
