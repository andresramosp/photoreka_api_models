import torch
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import euclidean
from uuid import uuid4
import random
import models
from torch.amp import autocast
from PIL import ImageColor
import base64
from io import BytesIO
from colorthief import ColorThief
from PIL import Image

# Configuración del dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"


def resize_image_max_width(image: Image.Image, max_width: int = 1500) -> Image.Image:
    width, height = image.size
    if width <= max_width:
        return image
    new_height = int((max_width / width) * height)
    return image.resize((max_width, new_height), Image.LANCZOS)

def process_grounding_dino_detections_batched(
    items: list[dict],
    categories: list[dict],
    max_width: int = 1500,
    batch_size: int = 2,
) -> list[dict]:
    results = []
    debug_folder = "debug_groundingdino_hf"
    os.makedirs(debug_folder, exist_ok=True)

    category_config = {cat["name"]: cat for cat in categories}
    all_category_names = list(category_config.keys())
    category_batches = [all_category_names[i:i + batch_size] for i in range(0, len(all_category_names), batch_size)]

    for item in items:
        id_ = item["id"]
        image = resize_image_max_width(item["image"], max_width)
        img_width, img_height = image.size

        print(f"\nProcesando imagen ID: {id_} ({img_width}x{img_height})")
        final_category_boxes = {cat_name: [] for cat_name in category_config}

        for batch_index, batch in enumerate(category_batches):
            print(f"  → Batch {batch_index + 1}/{len(category_batches)}: {', '.join(batch)}")
            prompt = ". ".join(batch).strip() + "."
            base_box_threshold = 0.3
            base_text_threshold = 0.25

            category_boxes_batch = {cat_name: [] for cat_name in batch}
            for attempt in range(3):
                box_threshold = base_box_threshold + 0.1 * attempt
                text_threshold = base_text_threshold + 0.1 * attempt

                inputs = models.MODELS["gdino_processor"](images=image, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    with autocast("cuda", dtype=torch.float16):
                        outputs = models.MODELS["gdino_model"](**inputs)

                target_size = torch.tensor([image.size[::-1]], device=device)
                detections = models.MODELS["gdino_processor"].post_process_grounded_object_detection(
                    outputs=outputs,
                    input_ids=inputs["input_ids"],
                    target_sizes=target_size,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )[0]

                temp_boxes = {cat_name: [] for cat_name in batch}
                for box, label in zip(detections["boxes"], detections["labels"]):
                    label = label.lower()
                    matched_cat = next((cat for cat in batch if cat.lower() == label), None)
                    if not matched_cat:
                        continue

                    config = category_config[matched_cat]
                    x1, y1, x2, y2 = box.tolist()
                    width, height = x2 - x1, y2 - y1
                    area = width * height
                    max_area = img_width * img_height * config["max_box_area_ratio"]

                    if width < config["min_box_size"] or height < config["min_box_size"] or area > max_area:
                        continue

                    temp_boxes[matched_cat].append([x1, y1, x2, y2])

                total_boxes = sum(len(b) for b in temp_boxes.values())
                if total_boxes > 0 or attempt == 2:
                    category_boxes_batch = temp_boxes
                    if attempt > 0:
                        print(f"    ↪ Reintento #{attempt}: box_threshold={box_threshold:.2f}, text_threshold={text_threshold:.2f}")
                    print(f"    ↪ Detectadas {total_boxes} cajas en batch")
                    break

            for cat in batch:
                final_category_boxes[cat].extend(category_boxes_batch.get(cat, []))

        # Dibujo final
        debug_img = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for cat_name, boxes in final_category_boxes.items():
            if not boxes:
                continue
            config = category_config[cat_name]
            rgb_color = ImageColor.getrgb(config["color"])
            border_color = rgb_color + (255,)
            fill_color = rgb_color + (70,)
            for box in boxes:
                draw.rectangle(box, outline=border_color, width=5, fill=fill_color)

        debug_img = Image.alpha_composite(debug_img, overlay).convert("RGB")
        debug_img.save(os.path.join(debug_folder, f"{id_}.jpg"))
        results.append({"id": id_, "detections": {k: v for k, v in final_category_boxes.items() if v}})

    return results


def process_grounding_dino_detections(
    items: list[dict],
    categories: list[dict],
    max_width: int = 1500,
) -> list[dict]:
    results = []
    debug_folder = "debug_groundingdino_hf"
    os.makedirs(debug_folder, exist_ok=True)

    category_config = {cat["name"]: cat for cat in categories}
    prompt = ". ".join(category_config.keys()).strip() + "."

    for item in items:
        id_ = item["id"]
        image = resize_image_max_width(item["image"], max_width)
        img_width, img_height = image.size

        category_boxes = None
        base_box_threshold = 0.3
        base_text_threshold = 0.25

        for attempt in range(3):  # intento 0 (original), 1, y 2
            box_threshold = base_box_threshold + 0.1 * attempt
            text_threshold = base_text_threshold + 0.1 * attempt

            inputs = models.MODELS["gdino_processor"](images=image, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                with autocast("cuda", dtype=torch.float16):
                    outputs =  models.MODELS["gdino_model"](**inputs)

            target_size = torch.tensor([image.size[::-1]], device=device)
            detections =  models.MODELS["gdino_processor"].post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs["input_ids"],
                target_sizes=target_size,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )[0]

            category_boxes = {cat_name: [] for cat_name in category_config}
            temp_boxes = {cat_name: [] for cat_name in category_config}
            for box, label in zip(detections["boxes"], detections["labels"]):
                label = label.lower()
                matched_cat = next((k for k in category_config if k.lower() == label), None)
                if not matched_cat:
                    continue

                config = category_config[matched_cat]
                x1, y1, x2, y2 = box.tolist()
                width, height = x2 - x1, y2 - y1
                area = width * height
                max_area = img_width * img_height * config["max_box_area_ratio"]

                if width < config["min_box_size"] or height < config["min_box_size"] or area > max_area:
                    continue

                temp_boxes[matched_cat].append([x1, y1, x2, y2])

            total_boxes = sum(len(b) for b in temp_boxes.values())
            if total_boxes > 0 or attempt == 2:
                category_boxes = temp_boxes
                if attempt > 0:
                    print(f"Reintento #{attempt} con box={box_threshold:.2f}, text={text_threshold:.2f} para imagen ID: {id_}")
                break

        # Dibujar cajas
        debug_img = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for cat_name, boxes in category_boxes.items():
            config = category_config[cat_name]
            rgb_color = ImageColor.getrgb(config["color"])
            border_color = rgb_color + (255,)
            fill_color = rgb_color + (70,)
            for box in boxes:
                draw.rectangle(box, outline=border_color, width=5, fill=fill_color)

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

    yolo_model = models.MODELS["yolo_model"]
    
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
        img = models.MODELS["clip_preprocess"](img)
        images.append(img)

    # Stack y pasar en batch
    image_batch = torch.stack(images).to(device)

    with torch.no_grad():
        embeddings = models.MODELS["clip_model"].encode_image(image_batch)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # normalización opcional

    # Asociar cada embedding con su ID
    results = []
    for idx, embedding in zip(ids, embeddings):
        results.append({"id": idx, "embedding": embedding.cpu().tolist()})

    return results


def get_color_embeddings_from_base64(items: list[dict]) -> list[dict]:
    images = []
    ids = []

    for item in items:
        ids.append(item["id"])
        image_data = base64.b64decode(item["base64"])
        img = Image.open(BytesIO(image_data)).convert("RGB")
        images.append(img)

    results = []

    for idx, img in zip(ids, images):
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        color_thief = ColorThief(img_bytes)
        palette = color_thief.get_palette(color_count=5)

        # Normalizamos a [0,1] y aplanamos
        flattened_palette = []
        for color in palette:
            flattened_palette.extend([c / 255 for c in color])

        results.append({"id": idx, "embedding": flattened_palette})

    return results
