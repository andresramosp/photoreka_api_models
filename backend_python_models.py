from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import torch
import cv2
import numpy as np
import base64
import shutil
import os
import json
import requests
import nltk
import spacy
import hashlib


app = Flask(__name__)

# Cargar variables de entorno desde .env
load_dotenv()

# Limpiar la caché de Hugging Face (opcional)
def clear_cache():
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("Caché de Hugging Face eliminada correctamente.")

def load_embeddings_model():
    nltk.download('wordnet') 
    model = SentenceTransformer('all-MiniLM-L6-v2')
    lemmatizer = WordNetLemmatizer()
    spacy_model = spacy.load("en_core_web_sm")  # Load a small English NLP model
    return model, lemmatizer, spacy_model

def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = None, None

# Cargar modelo BLIP-2
# def load_blip2_model():
#     processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
#     model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
#     return processor, model

# blip2_processor, blip2_model = None, None

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

@app.route("/llava", methods=["POST"])
def llava_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Se requiere un archivo de imagen"}), 400

        # Leer imagen desde el archivo
        file = request.files['image']
        image_path = "temp_image.jpg"
        file.save(image_path)

        # Codificar imagen en base64
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # Leer prompt opcional
        prompt = request.form.get("prompt", "Describe this image in detail.")

        print(os.getenv("HUGGINGFACE_TOKEN"))

        # Configurar cliente de Hugging Face
        llava_client = InferenceClient(model="llava-hf/llava-1.5-7b-hf", token=os.getenv("HUGGINGFACE_TOKEN"))

        # Preparar entrada para la API
        inputs = {
            "image": image_base64,
            "text": prompt
        }

        # Enviar imagen y prompt a la API
        response = llava_client.post(json=inputs)

        # Procesar la respuesta
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/llava_spaces", methods=["POST"])
def llava_spaces_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Se requiere un archivo de imagen"}), 400

        # Leer imagen desde el archivo
        file = request.files['image']
        image_path = "temp_image.jpg"
        file.save(image_path)

        # Codificar imagen en base64
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # Leer prompt opcional
        prompt = request.form.get("prompt", "Describe this image in detail.")

        # Preparar la solicitud para el Space de Hugging Face
        space_url = "https://<your-space-name>.hf.space/api/predict"
        payload = {
            "data": [image_base64, prompt]
        }

        # Realizar la solicitud HTTP
        response = requests.post(space_url, json=payload)
        response.raise_for_status()

        # Procesar la respuesta
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def preprocess_text(text):
    """Normalize text by lemmatizing and converting to lowercase if it's a single word."""
    if len(text.split()) == 1:  # Apply lemmatization only for single words
        return lemmatizer.lemmatize(text.lower())
    return text.lower()

@app.route('/semantic_proximity', methods=['POST'])
def calculate_semantic_proximity():
    try:
        # Parse JSON request
        data = request.get_json()
        tag = preprocess_text(data['tag'])
        tag_list = [preprocess_text(t) for t in data['tag_list']]
        threshold_percent = data.get('threshold', 0)  # Default threshold is 0

        # Convert threshold from percentage to similarity value (0 to 1)
        threshold = threshold_percent / 100.0

        # Generate cache key
        tag_hash = hashlib.md5(tag.encode()).hexdigest()
        tag_list_hash = hashlib.md5(''.join(tag_list).encode()).hexdigest()
        cache_key = f"{tag_hash}:{tag_list_hash}:{threshold_percent}"

        # Check cache
        if cache_key in similarity_cache:
            return jsonify({"similarities": similarity_cache[cache_key]})

        # Encode the input tag and tag list
        tag_embedding = embeddings_model.encode(tag, convert_to_tensor=True)
        tag_list_embeddings = embeddings_model.encode(tag_list, convert_to_tensor=True)

        # Compute cosine similarities
        similarities = util.cos_sim(tag_embedding, tag_list_embeddings)

        # Filter by threshold and convert similarities to percentage
        similarity_percentages = {
            original_tag: round(float(similarity) * 100, 2)
            for original_tag, similarity in zip(data['tag_list'], similarities[0])
            if float(similarity) >= threshold
        }

        # Cache the result
        similarity_cache[cache_key] = similarity_percentages

        return jsonify({"similarities": similarity_percentages})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/semantic_proximity_bulk', methods=['POST'])
def semantic_proximity_bulk():
    data = request.get_json()
    tags1 = data.get('tags1', [])
    tags2 = data.get('tags2', [])

    if not tags1 or not tags2:
        return jsonify({'error': 'Both tags1 and tags2 must be provided'}), 400

    # Obtener embeddings para ambas listas
    embeddings1 = embeddings_model.encode(tags1)
    embeddings2 = embeddings_model.encode(tags2)

    # Calcular similitudes de coseno
    similarities = cosine_similarity(embeddings1, embeddings2)

    # Construir el resultado como un diccionario
    result = {}
    for i, tag1 in enumerate(tags1):
        result[tag1] = {
            tags2[j]: float(similarities[i, j]) for j in range(len(tags2))
        }

    return jsonify({'similarities': result})
    
@app.route('/semantic_proximity_obj', methods=['POST'])
def calculate_semantic_proximity_obj():
    try:
        # Parse JSON request
        data = request.get_json()
        tag = preprocess_text(data['tag'])
        tag_list = [{"id": item["id"], "text": preprocess_text(item["text"])} for item in data['tag_list']]

        # Encode the input tag and tag list texts
        tag_embedding = embeddings_model.encode(tag, convert_to_tensor=True)
        tag_list_embeddings = embeddings_model.encode([item['text'] for item in tag_list], convert_to_tensor=True)

        # Compute cosine similarities
        similarities = util.cos_sim(tag_embedding, tag_list_embeddings)

        # Convert similarities to percentage and map to tag IDs
        similarity_percentages = {item['id']: round(float(similarity) * 100, 2) 
                                   for item, similarity in zip(tag_list, similarities[0])}

        return jsonify({"similarities": similarity_percentages})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/generate_tags', methods=['POST'])
def generate_tags():
    try:
        data = request.get_json()
        description = data.get('description')
        custom_tags = data.get('custom_tags', [])

        if not description:
            return jsonify({"error": "Description is required"}), 400

        # Encode the description
        description_embedding = embeddings_model.encode(description, convert_to_tensor=True)

        # Extract candidate tags from the description
        keywords = extract_keywords(description)

        # Match keywords to custom tags if provided
        matched_tags = {}
        if custom_tags:
            custom_embeddings = embeddings_model.encode(custom_tags, convert_to_tensor=True)
            similarities = util.cos_sim(description_embedding, custom_embeddings)

            for keyword, similarity in zip(custom_tags, similarities[0]):
                if float(similarity) >= 0.7:
                    matched_tags[keyword] = round(float(similarity) * 100, 2)

        return jsonify({
            "generated_tags": keywords,
            "matched_tags": matched_tags
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def extract_keywords(text, relevance_threshold=0.2):
    """
    Extract keywords based on context and importance.
    :param text: The input text.
    :param relevance_threshold: Threshold to filter relevant words (0 to 1).
    :return: A list of relevant keywords.
    """
    doc = spacy_model(text)

    # Extract candidate keywords: nouns, proper nouns, and entities
    keywords = []
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} or token.ent_type_:  # Includes entities
            keywords.append(token.text.lower())

    # Remove duplicates and encode in batch
    unique_keywords = list(set(keywords))
    keyword_embeddings = embeddings_model.encode(unique_keywords, convert_to_tensor=True)

    # Encode the full text and calculate relevance
    text_embedding = embeddings_model.encode(text, convert_to_tensor=True)
    similarities = util.cos_sim(keyword_embeddings, text_embedding)

    # Filter keywords based on relevance threshold
    relevant_keywords = [
        keyword for keyword, similarity in zip(unique_keywords, similarities[:, 0])
        if float(similarity) >= relevance_threshold
    ]

    return relevant_keywords



if __name__ == "__main__":
    # Limpiar caché si es necesario
    clear_cache()

    # Cargar modelos
    embeddings_model, lemmatizer, spacy_model = load_embeddings_model()
    similarity_cache = {}
    processor, model = load_blip_model()
    clip_processor, clip_model = load_clip_model()

    app.run(host="0.0.0.0", port=5000)
