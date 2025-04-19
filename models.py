import os
import torch
from cachetools import TTLCache

# Global
MODELS = None
MAX_RETRIES = 3

def init_models(only=None, load_nltk=True):
    print("Inicializando modelos comunes...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = "/runpod-volume/models" if os.path.exists("/runpod-volume") else None

    models = {}

    def should_load(name):
        return only is None or name in only

    if load_nltk:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        print("[INFO] Recursos NLTK descargados.")

    if should_load("embeddings_model"):
        try:
            from sentence_transformers import SentenceTransformer
            models["embeddings_model"] = SentenceTransformer(
                'all-mpnet-base-v2',
                device=device,
                cache_folder=cache_dir
            )
            print("[OK] embeddings_model cargado.")
        except Exception as e:
            print(f"[ERROR] embeddings_model: {e}")
            models["embeddings_model"] = None

    if should_load("roberta_classifier_text"):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
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
            print("[OK] roberta_classifier_text cargado.")
        except Exception as e:
            print(f"[ERROR] roberta_classifier_text: {e}")
            models["roberta_classifier_text"] = None

    if should_load("ner_model"):
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
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
            print("[OK] ner_model cargado.")
        except Exception as e:
            print(f"[ERROR] ner_model: {e}")
            models["ner_model"] = None

    if should_load("bart_classifier"):
        try:
            from transformers import pipeline
            models["bart_classifier"] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                cache_dir=cache_dir
            )
            print("[OK] bart_classifier cargado.")
        except Exception as e:
            print(f"[ERROR] bart_classifier: {e}")
            models["bart_classifier"] = None

    if should_load("nlp"):
        try:
            import spacy
            models["nlp"] = spacy.load("en_core_web_sm")
            print("[OK] SpaCy (nlp) cargado.")
        except Exception as e:
            print(f"[ERROR] spacy: {e}")
            models["nlp"] = None

    if should_load("yolo_model"):
        try:
            from ultralytics import YOLO
            models["yolo_model"] = YOLO("yolov8n.pt")
            print("[OK] yolo_model cargado.")
        except Exception as e:
            print(f"[ERROR] yolo_model: {e}")
            models["yolo_model"] = None

    if should_load("gdino_model") or should_load("gdino_processor"):
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
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
            print("[OK] grounding_dino cargado.")
        except Exception as e:
            print(f"[ERROR] grounding_dino: {e}")
            models["gdino_model"] = None
            models["gdino_processor"] = None

    if should_load("clip_model") or should_load("clip_preprocess"):
        try:
            import open_clip
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            clip_model = clip_model.to(device)
            clip_model.eval()
            models["clip_model"] = clip_model
            models["clip_preprocess"] = clip_preprocess
            print("[OK] CLIP cargado.")
        except Exception as e:
            print(f"[ERROR] clip_model: {e}")
            models["clip_model"] = None
            models["clip_preprocess"] = None

    models["cache"] = TTLCache(maxsize=200000, ttl=3600)
    print("[OK] Cache TTL inicializada.")

    return models


def get_models(only=None, load_nltk=True):
    global MODELS

    if MODELS is None:
        MODELS = init_models(only=only, load_nltk=load_nltk)

    retries = 0
    essential_keys = [key for key in MODELS if key != "cache"]
    while retries < MAX_RETRIES and any(MODELS.get(key) is None for key in essential_keys):
        print(f"[WARNING] Algunos modelos no se inicializaron correctamente. Reintentando ({retries + 1}/{MAX_RETRIES})...")
        MODELS = init_models(only=only, load_nltk=load_nltk)
        retries += 1

    return MODELS
