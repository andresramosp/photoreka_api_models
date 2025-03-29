import torch
import open_clip
from PIL import Image
from io import BytesIO
import base64

# Usa GPU si está disponible, si no, usa CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carga el modelo y el preprocesador
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# Función principal
def get_image_embeddings_from_base64(items: list[dict]) -> list[dict]:
    images = []
    ids = []

    for item in items:
        ids.append(item["id"])
        image_data = base64.b64decode(item["base64"])
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img = preprocess(img)
        images.append(img)

    # Stack y pasar en batch
    image_batch = torch.stack(images).to(device)

    with torch.no_grad():
        embeddings = model.encode_image(image_batch)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # normalización opcional

    # Asociar cada embedding con su ID
    results = []
    for idx, embedding in zip(ids, embeddings):
        results.append({"id": idx, "embedding": embedding.cpu().tolist()})

    return results
