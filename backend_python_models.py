from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import torch
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Cargar modelo BLIP
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = None, None

# Cargar modelo CLIP
def load_clip_model():
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return clip_processor, clip_model

clip_processor, clip_model = None, None

def generate_caption(image, max_length=20, num_beams=1, num_return_sequences=1):
    inputs = processor(image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences
    )
    return [processor.decode(output, skip_special_tokens=True) for output in outputs]

# YOLO setup (se asume YOLOv5 preentrenado)
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO("yolov5s.pt")

yolo_model = load_yolo_model()

def detect_objects(image, conf=0.25, iou=0.45, classes=None):
    # Ejecutar predicción con YOLO
    results = yolo_model.predict(image, conf=conf, iou=iou, classes=classes)
    detected_objects = []

    # Iterar sobre cada resultado detectado
    for result in results[0].boxes:  # Acceso a las cajas detectadas
        box = result.xyxy[0].cpu().numpy()  # Coordenadas xyxy
        label = yolo_model.names[int(result.cls.cpu().numpy())]  # Clase del objeto
        confidence = result.conf.cpu().numpy()  # Confianza
        detected_objects.append({
            "label": label,
            "confidence": float(confidence),
            "bbox": [float(coord) for coord in box]
        })

    return detected_objects

def extract_tags(descriptions, detections):
    # Combinar descripciones de BLIP y detecciones de YOLO para generar tags
    tags = set()

    # Extraer palabras clave de las descripciones
    for description in descriptions:
        tags.update(description.lower().split())

    # Añadir etiquetas de YOLO
    for detection in detections:
        tags.add(detection["label"].lower())

    return list(tags)

def clip_generate_tags(image, candidate_tags):
    # Procesar imagen y etiquetas para CLIP
    inputs = clip_processor(text=candidate_tags, images=image, return_tensors="pt", padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)  # Convertir a probabilidades

    # Combinar etiquetas con sus probabilidades
    tags_with_scores = sorted(zip(candidate_tags, probs[0].tolist()), key=lambda x: x[1], reverse=True)
    return [tag for tag, score in tags_with_scores if score > 0.1]  # Filtrar etiquetas con relevancia

@app.route("/blip", methods=["POST"])
def blip_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Se requiere un archivo de imagen"}), 400

        # Leer imagen desde el archivo
        file = request.files['image']
        image_data = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # Leer parámetros opcionales
        max_length = int(request.form.get("max_length", 20))
        num_beams = int(request.form.get("num_beams", 1))
        num_return_sequences = int(request.form.get("num_return_sequences", 1))

        # Generar descripciones con BLIP
        descriptions = generate_caption(image, max_length=max_length, num_beams=num_beams, num_return_sequences=num_return_sequences)
        return jsonify({"descriptions": descriptions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/yolo", methods=["POST"])
def yolo_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Se requiere un archivo de imagen"}), 400

        # Leer imagen desde el archivo
        file = request.files['image']
        image_data = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # Leer parámetros opcionales
        conf = float(request.form.get("conf", 0.25))
        iou = float(request.form.get("iou", 0.45))
        classes = request.form.get("classes")  # Lista de clases separadas por comas, e.g., "0,1,2"
        if classes:
            classes = [int(c) for c in classes.split(",")]

        # Detectar objetos con YOLO
        detections = detect_objects(image, conf=conf, iou=iou, classes=classes)
        return jsonify({"tags": detections})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/combined", methods=["POST"])
def combined_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Se requiere un archivo de imagen"}), 400

        # Leer imagen desde el archivo
        file = request.files['image']
        image_data = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # Leer parámetros opcionales
        max_length = int(request.form.get("max_length", 20))
        num_beams = int(request.form.get("num_beams", 1))
        num_return_sequences = int(request.form.get("num_return_sequences", 1))
        conf = float(request.form.get("conf", 0.25))
        iou = float(request.form.get("iou", 0.45))
        classes = request.form.get("classes")
        if classes:
            classes = [int(c) for c in classes.split(",")]

        # Generar descripciones con BLIP
        descriptions = generate_caption(image, max_length=max_length, num_beams=num_beams, num_return_sequences=num_return_sequences)

        # Detectar objetos con YOLO
        detections = detect_objects(image, conf=conf, iou=iou, classes=classes)

        # Extraer tags combinados
        tags = extract_tags(descriptions, detections)

        return jsonify({
            "descriptions": descriptions,
            "detections": detections,
            "tags": tags
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clip", methods=["POST"])
def clip_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Se requiere un archivo de imagen"}), 400

        # Leer imagen desde el archivo
        file = request.files['image']
        image_data = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # Leer etiquetas candidatas
        candidate_tags = request.form.get("candidate_tags")
        if not candidate_tags:
            return jsonify({"error": "Se requieren etiquetas candidatas"}), 400

        candidate_tags = candidate_tags.split(",")

        # Generar tags relevantes con CLIP
        tags = clip_generate_tags(image, candidate_tags)

        return jsonify({"tags": tags})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    processor, model = load_blip_model()
    clip_processor, clip_model = load_clip_model()
    app.run(host="0.0.0.0", port=5000)
