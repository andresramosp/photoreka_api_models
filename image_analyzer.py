import torch
import open_clip
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import euclidean
from uuid import uuid4
import random
from models import MODELS
from torch.amp import autocast
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Configuración del dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar modelo CLIP
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

def setup_grounding_dino(device="cuda"):
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to(device)

    # Convertir a FP16 para optimizar rendimiento/memoria
    model = model.half()
    model.eval()

    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    return model, processor

gdino_model, gdino_processor = setup_grounding_dino(device)

def resize_image_max_width(image: Image.Image, max_width: int = 1500) -> Image.Image:
    width, height = image.size
    if width <= max_width:
        return image
    new_height = int((max_width / width) * height)
    return image.resize((max_width, new_height), Image.LANCZOS)

def process_grounding_dino_detections(
    items: list[dict],
    categories: list[str],
    min_box_size: int,
    max_width: int = 1500,
    max_box_area_ratio: float = 0.8  # 80% del área de la imagen
) -> list[dict]:
    results = []
    debug_folder = "debug_groundingdino_hf"
    os.makedirs(debug_folder, exist_ok=True)

    prompt = ". ".join(categories).strip() + "."

    for item in items:
        id_ = item["id"]
        image = resize_image_max_width(item["image"], max_width)
        img_width, img_height = image.size
        max_box_area = img_width * img_height * max_box_area_ratio

        inputs = gdino_processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            with autocast("cuda", dtype=torch.float16):
                outputs = gdino_model(**inputs)

        target_size = torch.tensor([image.size[::-1]], device=device)
        detections = gdino_processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            target_sizes=target_size,
            box_threshold=0.3,
            text_threshold=0.25
        )[0]

        category_boxes = {cat: [] for cat in categories}
        debug_img = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for box, label in zip(detections["boxes"], detections["labels"]):
            label_lower = label.lower()
            matched_cat = next((cat for cat in categories if cat.lower() == label_lower), None)
            if not matched_cat:
                continue

            x1, y1, x2, y2 = box.tolist()
            width, height = x2 - x1, y2 - y1
            area = width * height

            if (
                width < min_box_size or height < min_box_size or
                area > max_box_area
            ):
                continue

            category_boxes[matched_cat].append([x1, y1, x2, y2])
            fill_color = tuple(random.randint(0, 255) for _ in range(3)) + (90,)
            draw.rectangle([x1, y1, x2, y2], outline=fill_color[:3], width=2, fill=fill_color)

        debug_img = Image.alpha_composite(debug_img, overlay).convert("RGB")
        debug_img.save(os.path.join(debug_folder, f"{id_}.jpg"))
        results.append({"id": id_, "detections": {k: v for k, v in category_boxes.items() if v}})

    return results

def process_yolo_detections(items: list[dict], categories: list[str], min_box_size: int) -> list[dict]:
    """
    Función única de lógica de negocio:
      - Recibe una lista de diccionarios con "id" e "image" (PIL.Image).
      - Ejecuta detección con YOLO, filtra por categorías y tamaño mínimo,
        dibuja cajas en una imagen de debug y las guarda localmente.
      - Retorna una lista con los boxes detectados para cada imagen.
    """
    results = []
    debug_folder = "debug_yolo"
    os.makedirs(debug_folder, exist_ok=True)

    yolo_model = MODELS["yolo_model"]
    
    for item in items:
        id_ = item["id"]
        img = item["image"]
        img_np = np.array(img)
        detections = yolo_model(img_np)
        boxes = []
        debug_img = img.copy()
        draw = ImageDraw.Draw(debug_img)

        # Se asume detections[0].boxes.data con formato [x1, y1, x2, y2, conf, cls]
        for det in detections[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = det
            label = yolo_model.model.names[int(cls_id)]
            width, height = x2 - x1, y2 - y1
            if label in categories and width >= min_box_size and height >= min_box_size:
                boxes.append([x1, y1, x2, y2])
                color = tuple(random.randint(0, 255) for _ in range(3))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
                
        # Guardar imagen de debug
        debug_img.save(os.path.join(debug_folder, f"{id_}.jpg"))
        results.append({"id": id_, "boxes": boxes})
    
    return results


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

def generate_presence_map1(
    base64_image: str,
    image_id: str = None,
    target_width=200,
    output_dir="presence_maps"
) -> tuple[np.ndarray, str]:
    # Decodifica y convierte a escala de grises
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data)).convert("L")
    image_np = np.array(image)

    # Detección de bordes
    edges = cv2.Canny(image_np, 100, 200)

    # Reescalado proporcional
    h, w = edges.shape
    scale = target_width / w
    target_size = (target_width, int(h * scale))
    small = cv2.resize(edges, target_size, interpolation=cv2.INTER_AREA)

    # Mapa binario
    presence_map = (small > 0).astype(int)

    # Guarda el archivo
    os.makedirs(output_dir, exist_ok=True)
    image_id = image_id or uuid4().hex[:8]
    output_path = os.path.join(output_dir, f"presence_map_{image_id}.png")
    cv2.imwrite(output_path, small)

    return presence_map, output_path


def generate_presence_map2(
    base64_image: str,
    image_id: str = None,
    target_width=200,
    output_dir="presence_maps",
    blur_kernel_size=(5, 5),
    adaptive_block_size=11,
    adaptive_C=2
) -> tuple[np.ndarray, str]:
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data)).convert("L")
    image_np = np.array(image)

    h, w = image_np.shape
    scale = target_width / w
    target_size = (target_width, int(h * scale))
    resized = cv2.resize(image_np, target_size, interpolation=cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(resized, blur_kernel_size, 0)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        adaptive_block_size, adaptive_C
    )

    presence_map = (adaptive > 0).astype(int)

    os.makedirs(output_dir, exist_ok=True)
    image_id = image_id or uuid4().hex[:8]
    output_path = os.path.join(output_dir, f"presence_map_{image_id}.png")
    cv2.imwrite(output_path, adaptive)

    return presence_map, output_path

# contornos
def generate_presence_map3(
    base64_image: str,
    image_id: str = None,
    target_width=200,
    output_dir="presence_maps",
    canny_threshold1=50,
    canny_threshold2=150,
    min_contour_area=10
) -> tuple[np.ndarray, str]:
    # Decodifica y convierte a escala de grises
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data)).convert("L")
    image_np = np.array(image)

    # Reescalado proporcional
    h, w = image_np.shape
    scale = target_width / w
    target_size = (target_width, int(h * scale))
    resized = cv2.resize(image_np, target_size, interpolation=cv2.INTER_AREA)

    # Canny para detección de bordes
    edges = cv2.Canny(resized, canny_threshold1, canny_threshold2)

    # Encuentra contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mapa en blanco y dibuja solo los contornos con cierta área
    contour_map = np.zeros_like(edges)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_contour_area:
            cv2.drawContours(contour_map, [cnt], -1, color=255, thickness=1)

    presence_map = (contour_map > 0).astype(int)

    os.makedirs(output_dir, exist_ok=True)
    image_id = image_id or uuid4().hex[:8]
    output_path = os.path.join(output_dir, f"presence_map_{image_id}.png")
    cv2.imwrite(output_path, contour_map)

    return presence_map, output_path


def generate_presence_map(
    base64_image: str,
    image_id: str = None,
    method: int = 1,
    **kwargs
) -> tuple[np.ndarray, str]:
    if method == 1:
        return generate_presence_map1(base64_image, image_id=image_id, **kwargs)
    elif method == 2:
        return generate_presence_map2(base64_image, image_id=image_id, **kwargs)
    elif method == 3:
        return generate_presence_map3(base64_image, image_id=image_id, **kwargs)
    else:
        raise ValueError(f"Método de generación {method} no está implementado.")


def find_similar_presence_maps_by_id(
    image_id: str,
    maps_folder: str = "presence_maps",
    top_k: int = 5
) -> list[dict]:
    query_path = os.path.join(maps_folder, f"presence_map_{image_id}.png")
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"Presence map for ID '{image_id}' not found.")

    img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    query_map = (img > 0).astype(int)
    query_vec = query_map.flatten()

    results = []

    for filename in os.listdir(maps_folder):
        if not filename.endswith(".png") or not filename.startswith("presence_map_"):
            continue

        comp_id = filename.replace("presence_map_", "").replace(".png", "")

        # Excluir imagen de consulta y archivos temporales
        if comp_id == image_id or comp_id.startswith("temp"):
            continue

        path = os.path.join(maps_folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        comp_map = (img > 0).astype(int).flatten()

        if comp_map.shape != query_vec.shape:
            continue

        distance = np.linalg.norm(query_vec - comp_map)
        results.append({"id": comp_id, "distance": distance})

    results.sort(key=lambda x: x["distance"])
    return results[:top_k]

