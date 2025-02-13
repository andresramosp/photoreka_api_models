import runpod
from api import (
    adjust_tags_proximities_by_context_inference_logic,
    adjust_descs_proximities_by_context_inference_logic,
    get_embeddings_logic
)

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
        else:
            result = {"error": f"Operation '{operation}' not supported"}
    except Exception as e:
        result = {"error": str(e)}

    return result

# Iniciar el servidor RunPod Serverless
runpod.serverless.start({"handler": handler})
