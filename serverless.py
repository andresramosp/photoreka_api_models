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


async def handler(job):
    input_data = job.get("input", {})
    operation = input_data.get("operation")
    data = input_data.get("data", {})

    if not operation:
        return {"error": "Missing 'operation' in input"}

    try:
        if operation == "adjust_tags_proximities_by_context_inference":
            result = adjust_tags_proximities_by_context_inference_logic(data)
        elif operation == "adjust_descs_proximities_by_context_inference":
            result = adjust_descs_proximities_by_context_inference_logic(data)
        elif operation == "get_embeddings":
            result = get_embeddings_logic(data)
        elif operation == "get_embeddings_image":
            result = get_image_embeddings_from_base64(data.get("images", []))
        elif operation == "detect_objects_base64":
            items = []
            for img_obj in data.get("images", []):
                img_data = base64.b64decode(img_obj["base64"])
                img = Image.open(BytesIO(img_data)).convert("RGB")
                items.append({"id": img_obj["id"], "image": img})
            result = process_grounding_dino_detections_batched(items, data.get("categories", []))
        elif operation == "query_segment":
            result = query_segment(data.get("query", ""))
        elif operation == "query_no_prefix":
            result = remove_photo_prefix(data.get("query", ""))
        elif operation == "clean_texts":
            result = clean_texts(data)
        # elif operation == "generate_groups_for_tags":
        #     result = generate_groups_for_tags(data)
        # elif operation == "extract_tags":
        #     method = data.get("method", "spacy")
        #     if method == "spacy":
        #         result = extract_tags_spacy(data.get("text", ""), data.get("allowed_groups", []))
        #     elif method == "ntlk":
        #         result = extract_tags_ntlk(data.get("text", ""))
        #     else:
        #         result = {"error": "Unsupported method"}
        else:
            result = {"error": f"Unsupported operation: {operation}"}
    except Exception as e:
        result = {"error": str(e)}

    return result
