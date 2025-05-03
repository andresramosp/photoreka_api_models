import runpod

from models import get_models
from logic_inference import get_embeddings_logic

# ====== Initialization ======
print("[INIT] Starting model loading...", flush=True)
get_models(only=["embeddings_model"], load_nltk=False)
print("[INIT] Embeddings model loaded.", flush=True)


def handler(job):
    # Entry
    print("[HANDLER] Invoked handler.", flush=True)
    print(f"[HANDLER] Received job payload: {job}", flush=True)

    input_data = job.get("input", {})
    print(f"[HANDLER] Parsed input_data: {input_data}", flush=True)

    operation = input_data.get("operation")
    data = input_data.get("data", {})
    print(f"[HANDLER] Operation: {operation}", flush=True)
    print(f"[HANDLER] Data: {data}", flush=True)

    # Validate operation
    if not operation:
        print("[ERROR] Missing 'operation' in input.", flush=True)
        return {"error": "Missing 'operation' in input"}

    # Ping check
    if operation == "ping":
        print("[HANDLER] Ping operation detected. Responding with warmed up.", flush=True)
        return {"status": "warmed up"}

    # Main logic
    try:
        print("[HANDLER] Calling get_embeddings_logic...", flush=True)
        result = get_embeddings_logic(data)
        print(f"[HANDLER] get_embeddings_logic returned: {result}", flush=True)
    except Exception as e:
        print(f"[ERROR] Exception in get_embeddings_logic: {e}", flush=True)
        result = {"error": str(e)}

    print(f"[HANDLER] Returning result: {result}", flush=True)
    return result


# Start the serverless handler without extra concurrency
if __name__ == "__main__":
    print("[SERVER] Starting RunPod serverless...")
    runpod.serverless.start({
        "handler": handler
    })
