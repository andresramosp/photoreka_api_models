import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
from cachetools import TTLCache
import nltk
from ultralytics import YOLO
from controlnet_aux import MLSDdetector

# Descargas NLTK (se realizan al importar el módulo)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def init_models():
    print("Inicializando modelos comunes...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_model = SentenceTransformer('all-mpnet-base-v2', device=device)

    roberta_classifier_model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large-mnli", torch_dtype=torch.bfloat16, device_map="auto"
)
    roberta_classifier_text = pipeline(
        "text-classification",
        model=roberta_classifier_model,
        tokenizer=AutoTokenizer.from_pretrained("roberta-large-mnli", use_fast=True)
    )
    ner_model = pipeline(
        "ner",
        model="FacebookAI/xlm-roberta-large-finetuned-conll03-english",
        aggregation_strategy="simple",
        device=0 if device == "cuda" else -1
    )
    bart_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    nlp = spacy.load("en_core_web_sm")
    cache = TTLCache(maxsize=200000, ttl=3600)

    yolo_model = YOLO("yolov8n.pt")
    # print(yolo_model.model.names)

    mlsd_detector = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    return {
        "embeddings_model": embeddings_model,
        "roberta_classifier_text": roberta_classifier_text,
        "ner_model": ner_model,
        "bart_classifier": bart_classifier,
        "nlp": nlp,
        "cache": cache,
        "yolo_model": yolo_model,
        "mlsd_model": mlsd_detector,
    }

# Cargar MODELS al momento de la importación
MODELS = init_models()
