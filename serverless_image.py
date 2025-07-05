import runpod
import logging

from models import get_models
from image_analyzer import get_image_embeddings_from_base64

# Configurar logs
logging.basicConfig(level=logging.INFO)

# Inicializa solo los modelos necesarios
get_models(
    only=["clip_model", "clip_preprocess"],
    load_nltk=False
)

async def handler(job):
    input_data = job.get("input", {})
    operation = input_data.get("operation")
    data = input_data.get("data", {})

    logging.info(f"Job recibido: {job}")
    logging.info(f"Operación solicitada: {operation}")

    if not operation:
        logging.warning("Falta 'operation' en el input.")
        return {"error": "Missing 'operation' in input"}

    try:
        if operation == "ping":
            logging.info("Ping recibido.")
            return {"status": "warmed up"}

        if operation == "get_embeddings_image":
            images = data.get("images", [])
            logging.info(f"Imágenes recibidas: {len(images)}")

            if not images:
                logging.warning("No se recibieron imágenes.")
                return {"error": "No images provided."}

            result = get_image_embeddings_from_base64(images)
            logging.info("Embeddings procesados correctamente.")

        # elif operation == "detect_objects_base64":
        #     items = []
        #     for img_obj in data.get("images", []):
        #         img_data = base64.b64decode(img_obj["base64"])
        #         img = Image.open(BytesIO(img_data)).convert("RGB")
        #         items.append({"id": img_obj["id"], "image": img})

        #     result = await asyncio.to_thread(
        #         process_grounding_dino_detections_batched,
        #         items,
        #         data.get("categories", []),
        #     )
        else:
            logging.warning(f"Operación no soportada: {operation}")
            result = {"error": f"Unsupported operation: {operation}"}

    except Exception as e:
        logging.error(f"Error durante el procesamiento: {str(e)}")
        result = {"error": str(e)}

    return result


# Permite hasta 4 peticiones simultáneas por worker.
runpod.serverless.start({
    "handler": handler
    # "concurrency_modifier": (lambda current: 4),
})
