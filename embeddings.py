import time
import torch
import models

def get_embeddings_logic(data: dict):
    print("ENTRA EN EL PUTO METODO")
    start_time = time.perf_counter()
    tags = data.get("tags", [])
    if not tags or not isinstance(tags, list):
        raise ValueError("Field 'tags' must be a list.")
    embeddings_model = models.MODELS["embeddings_model"]
    with torch.inference_mode():
        embeddings_tensor = embeddings_model.encode(tags, batch_size=16, convert_to_tensor=True)
    embeddings = embeddings_tensor.cpu().tolist()
    print(f"⏳ [Get Embeddings] Tiempo de ejecución: {time.perf_counter() - start_time:.4f} segundos")

    return {"tags": tags, "embeddings": embeddings}
