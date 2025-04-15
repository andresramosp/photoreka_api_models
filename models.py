import os
import time
import torch
import spacy
from cachetools import TTLCache
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
    AutoProcessor, AutoModelForZeroShotObjectDetection
)
from ultralytics import YOLO
import open_clip
import nltk

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

def init_models():
    print("Inicializando modelos comunes...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = "/runpod-volume/models" if os.path.exists("/runpod-volume") else None

    models = {}

    # SentenceTransformer
    try:
        models["embeddings_model"] = SentenceTransformer(
            'all-mpnet-base-v2',
            device=device,
            cache_folder=cache_dir
        )
    except Exception as e:
        print(f"[ERROR] embeddings_model: {e}")
        models["embeddings_model"] = None

    # Roberta classifier
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large-mnli",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "roberta-large-mnli",
            use_fast=True,
            cache_dir=cache_dir
        )
        models["roberta_classifier_text"] = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer
        )
    except Exception as e:
        print(f"[ERROR] roberta_classifier_text: {e}")
        models["roberta_classifier_text"] = None

    # NER
    try:
        ner_model = AutoModelForTokenClassification.from_pretrained(
            "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
            cache_dir=cache_dir
        )
        ner_tokenizer = AutoTokenizer.from_pretrained(
            "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
            cache_dir=cache_dir
        )
        models["ner_model"] = pipeline(
            "ner",
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple",
            device=0 if device == "cuda" else -1
        )
    except Exception as e:
        print(f"[ERROR] ner_model: {e}")
        models["ner_model"] = None

    # BART classifier
    try:
        models["bart_classifier"] = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            cache_dir=cache_dir
        )
    except Exception as e:
        print(f"[ERROR] bart_classifier: {e}")
        models["bart_classifier"] = None

    # SpaCy
    try:
        models["nlp"] = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"[ERROR] spacy: {e}")
        models["nlp"] = None

    # Cache
    models["cache"] = TTLCache(maxsize=200000, ttl=3600)

    # YOLO
    try:
        models["yolo_model"] = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"[ERROR] yolo_model: {e}")
        models["yolo_model"] = None

    # Grounding DINO
    try:
        gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base",
            cache_dir=cache_dir
        ).to(device)
        gdino_model = gdino_model.half()
        gdino_model.eval()
        gdino_processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base",
            cache_dir=cache_dir
        )
        models["gdino_model"] = gdino_model
        models["gdino_processor"] = gdino_processor
    except Exception as e:
        print(f"[ERROR] grounding_dino: {e}")
        models["gdino_model"] = None
        models["gdino_processor"] = None

    # CLIP
    try:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        clip_model = clip_model.to(device)
        clip_model.eval()
        models["clip_model"] = clip_model
        models["clip_preprocess"] = clip_preprocess
    except Exception as e:
        print(f"[ERROR] clip_model: {e}")
        models["clip_model"] = None
        models["clip_preprocess"] = None

    return models

# Cargar MODELS al momento de la importación
MODELS = None
MAX_RETRIES = 3

def get_models():
    global MODELS  
    if MODELS is None:
        MODELS = init_models()

    # Verificar si alguno de los modelos (excluida la cache) es None y reintentar la inicialización
    retries = 0
    essential_keys = [key for key in MODELS if key != "cache"]
    while retries < MAX_RETRIES and any(MODELS.get(key) is None for key in essential_keys):
        print(f"[WARNING] Algunos modelos no se inicializaron correctamente. Reintentando ({retries + 1}/{MAX_RETRIES})...")
        MODELS = init_models()
        retries += 1
