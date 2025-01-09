from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from SPARQLWrapper import SPARQLWrapper, JSON
from asgiref.wsgi import WsgiToAsgi
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
from nltk.corpus import wordnet
from functools import lru_cache

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


def load_wordnet():
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

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
    """Normalize text by lemmatizing, converting to lowercase, and replacing underscores with spaces."""
    normalized_text = text.lower().replace('_', ' ')
    if len(normalized_text.split()) == 1:  # Apply lemmatization only for single words
        return lemmatizer.lemmatize(normalized_text)
    return normalized_text

@app.route('/semantic_proximity_chunks', methods=['POST'])
def semantic_proximity_chunks():
    try:
        # Parse request data
        data = request.get_json()
        text1 = data.get('text1')
        text2 = data.get('text2')
        chunk_size = int(data.get('chunk_size', 100))

        if not text1 or not text2:
            return jsonify({"error": "Both 'text1' and 'text2' are required."}), 400

        # Split text2 into chunks by sentences while respecting chunk_size
        sentences = text2.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (sentence + ". ")
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Encode text1 and chunks
        text1_embedding = embeddings_model.encode(text1, convert_to_tensor=True)
        chunk_embeddings = embeddings_model.encode(chunks, convert_to_tensor=True)

        # Calculate semantic proximity for each chunk
        proximities = util.cos_sim(text1_embedding, chunk_embeddings)[0]

        # Create response with chunk and proximity
        response = [
            {"text_chunk": chunk, "proximity": round(float(proximity), 4)}
            for chunk, proximity in zip(chunks, proximities)
        ]

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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



# @app.route('/get_semantically_contained_words_embeddings', methods=['POST'])
# def get_related_words():
#     try:
#         # Parsear el JSON de la solicitud
#         data = request.json
#         word = preprocess_text(data.get('word'))
#         tag_list = [preprocess_text(tag) for tag in data.get('tag_list', [])]
#         num_results = data.get('num_results', 10)  # Por defecto, 10 resultados
#         threshold_percent = data.get('threshold', 50)  # Umbral por defecto: 50%
#         levels = data.get('levels', 1)  # Por defecto, 1 nivel de profundidad

#         if not word:
#             return jsonify({'error': 'Por favor, proporciona un parámetro word'}), 400

#         # Convertir el umbral de porcentaje a valor de similitud (0 a 1)
#         threshold = threshold_percent / 100.0

#         if not tag_list:
#             # Usar WordNet para encontrar palabras relacionadas
#             synsets = wordnet.synsets(word)
#             related_words = set()

#             for synset in synsets:
#                 # Incluir sinónimos
#                 related_words.update([lemma.name() for lemma in synset.lemmas()])

#                 # Incluir hipónimos hasta el nivel especificado
#                 hyponyms = get_hyponyms_recursively(synset, levels)
#                 related_words.update([lemma.name() for hyponym in hyponyms for lemma in hyponym.lemmas()])

#             # Limitar el número de resultados
#             related_words = list(related_words)[:num_results]

#             return jsonify({'word': word, 'related_words': related_words})

#         # Obtener embeddings para la palabra y la lista de tags
#         word_embedding = embeddings_model.encode(word, convert_to_tensor=True)
#         tag_list_embeddings = embeddings_model.encode(tag_list, convert_to_tensor=True)

#         # Calcular similitudes de coseno
#         similarities = util.cos_sim(word_embedding, tag_list_embeddings)

#         # Filtrar la tag_list por proximidad semántica
#         filtered_tag_list = [
#             tag for tag, similarity in zip(tag_list, similarities[0])
#             if float(similarity) >= threshold
#         ]

#         # Usar WordNet para encontrar palabras relacionadas solo dentro de la lista filtrada
#         synsets = wordnet.synsets(word)
#         related_words = set()

#         for synset in synsets:
#             # Incluir sinónimos
#             related_words.update([lemma.name() for lemma in synset.lemmas()])

#             # Incluir hipónimos hasta el nivel especificado
#             hyponyms = get_hyponyms_recursively(synset, levels)
#             related_words.update([lemma.name() for hyponym in hyponyms for lemma in hyponym.lemmas()])

#         # Filtrar las palabras relacionadas que estén en la lista filtrada
#         related_words = list(set(filtered_tag_list) & related_words)[:num_results]

#         for synset in synsets:
#             print(f"Hipónimos de {synset.name()}: {[lemma.name() for hyponym in synset.hyponyms() for lemma in hyponym.lemmas()]}")
    
#         # Imprimir similitudes calculadas
#         print(f"Similitudes calculadas: {filtered_tag_list}")

#         return jsonify({'word': word, 'related_words': related_words})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

def get_hyponyms_recursively(synset, levels):
    """Get hyponyms up to a specified number of levels."""
    if levels == 0:
        return set()

    hyponyms = set(synset.hyponyms())
    for hyponym in list(hyponyms):
        hyponyms.update(get_hyponyms_recursively(hyponym, levels - 1))

    return hyponyms

def filter_relevant_words(synsets, tag_list, levels, word, threshold=0.5):
    """Filter relevant words based on synsets, conceptual containment, and semantic proximity."""
    related_words = set()

    for synset in synsets:
        print(f"Processing synset: {synset.name()}")
        print(f"Lemmas: {[lemma.name() for lemma in synset.lemmas()]}")

        # Incluir sinónimos
        related_words.update([lemma.name() for lemma in synset.lemmas()])

        # Incluir hipónimos hasta el nivel especificado
        hyponyms = get_hyponyms_recursively(synset, levels)
        for hyponym in hyponyms:
            print(f"Hyponym: {hyponym.name()} Lemmas: {[lemma.name() for lemma in hyponym.lemmas()]}")
            related_words.update([lemma.name() for lemma in hyponym.lemmas()])

    # Filtrar las palabras relacionadas que estén en la lista de tags proporcionada
    if tag_list:
        related_words = set(tag_list) & related_words

    return list(related_words)


@app.route('/get_semantically_contained_words', methods=['POST'])
def get_related_words():
    try:
        # Parsear el JSON de la solicitud
        data = request.json
        word = preprocess_text(data.get('word', ''))
        tag_list = [preprocess_text(tag) for tag in data.get('tag_list', [])]
        num_results = data.get('num_results', 10)  # Por defecto, 10 resultados
        levels = data.get('levels', 1)  # Por defecto, 1 nivel de profundidad
        threshold = data.get('threshold', 0.5)  # Por defecto, umbral 0.5

        if not word:
            return jsonify({'error': 'Por favor, proporciona un parámetro word'}), 400

        # Usar WordNet para encontrar palabras relacionadas
        synsets = wordnet.synsets(word)
        related_words = filter_relevant_words(synsets, tag_list, levels, word, threshold)

        if not related_words:
            return jsonify({'word': word, 'related_words': []})

        # Limitar el número de resultados
        related_words = related_words[:num_results]

        return jsonify({'word': word, 'related_words': related_words})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_semantic_explosion', methods=['POST'])
def get_semantic_explosion():
    try:
        # Parsear el JSON de la solicitud
        data = request.json
        word = preprocess_text(data.get('word'))
        tag_list = [preprocess_text(tag) for tag in data.get('tag_list', [])]
        num_results = data.get('num_results', 10)  # Por defecto, 10 resultados
        threshold_percent = data.get('threshold', 50)  # Por defecto, umbral 50%

        if not word:
            return jsonify({'error': 'Por favor, proporciona un parámetro word'}), 400

        # Convertir el umbral de porcentaje a valor de similitud (0 a 1)
        threshold = threshold_percent / 100.0

        if tag_list:
            # Obtener embeddings para la palabra y la lista de tags
            word_embedding = embeddings_model.encode(word, convert_to_tensor=True)
            tag_list_embeddings = embeddings_model.encode(tag_list, convert_to_tensor=True)

            # Calcular similitudes de coseno
            similarities = util.cos_sim(word_embedding, tag_list_embeddings)

            # Filtrar la tag_list por proximidad semántica
            filtered_tag_list = [
                tag for tag, similarity in zip(tag_list, similarities[0])
                if float(similarity) >= threshold
            ]

            # Limitar el número de resultados
            filtered_tag_list = filtered_tag_list[:num_results]

            return jsonify({'word': word, 'semantic_tags': filtered_tag_list})
        else:
            # Usar WordNet para encontrar palabras relacionadas en todas direcciones
            synsets = wordnet.synsets(word)
            related_words = set()

            for synset in synsets:
                # Incluir sinónimos e hipónimos
                related_words.update([lemma.name() for lemma in synset.lemmas()])
                related_words.update([lemma.name() for hyponym in synset.hyponyms() for lemma in hyponym.lemmas()])

                # Incluir hiperónimos
                related_words.update([lemma.name() for hypernym in synset.hypernyms() for lemma in hypernym.lemmas()])

            # Limitar el número de resultados
            related_words = list(related_words)[:num_results]

            return jsonify({'word': word, 'semantic_explosion': related_words})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
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

@lru_cache(maxsize=10000)  # Cachea hasta 10,000 resultados
def verificar_subclase(subclase_qid, superclase_qid):
    """
    Verifica si un QID es subclase de otro usando una consulta ASK.
    Los resultados se cachean para evitar consultas repetidas.
    """
    consulta = f"""
    ASK {{
      wd:{subclase_qid} wdt:P279* wd:{superclase_qid}.
    }}
    """
    try:
        sparql.setQuery(consulta)
        sparql.setReturnFormat(JSON)
        resultado = sparql.query().convert()
        return resultado.get('boolean', False)
    except Exception as e:
        print(f"Error en la consulta SPARQL: {e}")
        return False

def obtener_qid(tag_name, lang="en"):
    """
    Obtiene el QID de un tag dado su nombre.
    """
    consulta = f"""
    SELECT ?item WHERE {{
      ?item rdfs:label "{tag_name}"@{lang}.
    }}
    LIMIT 1
    """
    try:
        sparql.setQuery(consulta)
        sparql.setReturnFormat(JSON)
        resultados = sparql.query().convert()
        bindings = resultados.get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["item"]["value"].split("/")[-1]
    except Exception as e:
        print(f"Error al obtener QID para '{tag_name}': {e}")
    return None

# Define the preprocessing function at the top level
def preprocess_for_synonym_matching(text):
    text = text.lower().replace('_', ' ')
    tokens = spacy_model(text)
    processed_tokens = [
        lemmatizer.lemmatize(token.text) if token.pos_ in {"NOUN", "VERB", "ADJ"} else token.text
        for token in tokens
    ]
    return " ".join(processed_tokens)

@app.route('/get_synonym_tags', methods=['POST'])
def get_synonym_tags():
    try:
        data = request.get_json()
        tag = preprocess_for_synonym_matching(data.get('tag', ''))
        tag_list = [preprocess_for_synonym_matching(t) for t in data.get('tag_list', [])]
        proximity_threshold = float(data.get('proximity_threshold', 0.9))  # Default semantic proximity threshold
        apply_semantic_proximity = data.get('apply_semantic_proximity', False)  # Whether to apply semantic proximity

        if not tag or not tag_list:
            return jsonify({"error": "Both 'tag' and 'tag_list' are required."}), 400

        processed_tag_list = [preprocess_for_synonym_matching(t) for t in data.get('tag_list', [])]

        # Strict lexical matches: Match only exact lemmatized phrases
        lexical_matches = [
            original_tag for original_tag, processed_tag_item in zip(data.get('tag_list', []), processed_tag_list)
            if tag == processed_tag_item
        ]

        # Apply semantic proximity (optional)
        semantic_matches = []
        if apply_semantic_proximity:
            tag_embedding = embeddings_model.encode(tag, convert_to_tensor=True)
            tag_list_embeddings = embeddings_model.encode(processed_tag_list, convert_to_tensor=True)

            similarities = util.cos_sim(tag_embedding, tag_list_embeddings)

            # Filter by threshold
            semantic_matches = [
                original_tag for original_tag, similarity in zip(data.get('tag_list', []), similarities[0])
                if float(similarity) >= proximity_threshold
            ]

        # Combine results
        combined_matches = lexical_matches if not apply_semantic_proximity else list(set(lexical_matches + semantic_matches))

        # Ensure original tags are returned instead of processed tags
        return jsonify({
            "tag": data.get('tag', ''),
            "matches": combined_matches
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_advanced_synonym_tags', methods=['POST'])
async def get_advanced_synonym_tags():
    try:
        data = request.get_json()
        tag = preprocess_for_synonym_matching(data.get('tag', ''))
        tag_list = data.get('tag_list', [])
        proximity_threshold = float(data.get('proximity_threshold', 0.9))  # Default threshold is 0.9

        if not tag or not tag_list:
            return jsonify({"error": "Both 'tag' and 'tag_list' are required."}), 400

        # # Extract relevant words (nouns, adjectives, verbs) from the main tag
        # tag_doc = spacy_model(tag)
        # tag_words = {
        #     token.text.lower() for token in tag_doc
        #     if token.pos_ in {"NOUN", "VERB", "ADJ"}
        # }

        # # Filter the tag list based on relevance
        # filtered_tags = []
        # for candidate_tag in tag_list:
        #     candidate_doc = spacy_model(candidate_tag)
        #     candidate_words = {
        #         token.text.lower() for token in candidate_doc
        #         if token.pos_ in {"NOUN", "VERB", "ADJ"}
        #     }

        #     # Check if at least half of the main tag's words are in the candidate tag
        #     if len(tag_words & candidate_words) >= max(1, len(tag_words) / 2):
        #         filtered_tags.append(candidate_tag)

        # # If no tags pass the initial filtering, return an empty result
        # if not filtered_tags:
        #     return jsonify({
        #         "tag": tag,
        #         "similar_tags": []
        #     })

        # Apply semantic proximity on filtered tags
        tag_embedding = embeddings_model.encode(tag, convert_to_tensor=True)
        filtered_tag_embeddings = embeddings_model.encode(tag_list, convert_to_tensor=True)

        similarities = util.cos_sim(tag_embedding, filtered_tag_embeddings)

        # Filter tags by proximity threshold
        similar_tags = [
            candidate_tag for candidate_tag, similarity in zip(tag_list, similarities[0])
            if float(similarity) >= proximity_threshold
        ]

        return jsonify({
            "tag": tag,
            "matches": similar_tags
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# @app.route('/check', methods=['POST'])
# def check_relation():
#     data = request.get_json()
#     tag_principal_name = data.get('tag_principal')
#     tags = data.get('tags')

#     if not tag_principal_name or not tags:
#         return jsonify({"error": "Se requieren 'tag_principal' y 'tags'."}), 400

#     # Obtener el QID del tag principal
#     tag_principal_qid = obtener_qid(tag_principal_name)
#     print(f"QID del tag principal '{tag_principal_name}': {tag_principal_qid}")
#     if not tag_principal_qid:
#         return jsonify({"error": f"No se encontró QID para el tag principal '{tag_principal_name}'."}), 404

#     resultados = {}
#     for tag_name in tags:
#         tag_qid = obtener_qid(tag_name)
#         print(f"QID de '{tag_name}': {tag_qid}")
#         if not tag_qid:
#             resultados[tag_name] = "No se encontró QID"
#             continue

#         es_subclase = verificar_subclase(tag_qid, tag_principal_qid)
#         print(f"¿'{tag_name}' es subclase de '{tag_principal_name}'? {es_subclase}")
#         resultados[tag_name] = es_subclase

#     return jsonify(resultados)

@app.route('/get_embeddings', methods=['POST'])
def get_embeddings():
    try:
        # Parsear el JSON de la solicitud
        data = request.get_json()
        tags = data.get('tags', [])

        if not tags or not isinstance(tags, list):
            return jsonify({"error": "Se requiere una lista de palabras en el campo 'tags'."}), 400

        # Generar los embeddings para las palabras
        embeddings = embeddings_model.encode(tags, convert_to_tensor=False)

        # Crear respuesta con las palabras y sus embeddings
        response = {
            "tags": tags,
            "embeddings": [embedding.tolist() for embedding in embeddings]
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Limpiar caché si es necesario
    clear_cache()

    # Cargar modelos
    embeddings_model, lemmatizer, spacy_model = load_embeddings_model()
    load_wordnet()
    similarity_cache = {}
    # processor, model = load_blip_model()
    # clip_processor, clip_model = load_clip_model()

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)

    # app.run(host="0.0.0.0", port=5000)
     # Convert Flask app to ASGI
    asgi_app = WsgiToAsgi(app)

    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=5000)
