import os
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
    # Usa /runpod-volume/models solo si existe (RunPod); si no, usa default
    cache_dir = "/runpod-volume/models" if os.path.exists("/runpod-volume") else None

    embeddings_model = SentenceTransformer(
        'all-mpnet-base-v2',
        device=device,
        cache_folder=cache_dir
    )

    roberta_classifier_model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-large-mnli",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )
    roberta_classifier_tokenizer = AutoTokenizer.from_pretrained(
        "roberta-large-mnli",
        use_fast=True,
        cache_dir=cache_dir
    )
    roberta_classifier_text = pipeline(
        "text-classification",
        model=roberta_classifier_model,
        tokenizer=roberta_classifier_tokenizer
    )

    # NER con carga manual
    ner_model_obj = AutoModelForTokenClassification.from_pretrained(
        "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
        cache_dir=cache_dir
    )
    ner_tokenizer = AutoTokenizer.from_pretrained(
        "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
        cache_dir=cache_dir
    )
    ner_model = pipeline(
        "ner",
        model=ner_model_obj,
        tokenizer=ner_tokenizer,
        aggregation_strategy="simple",
        device=0 if device == "cuda" else -1
    )

    bart_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        cache_dir=cache_dir
    )

    nlp = spacy.load("en_core_web_sm")
    cache = TTLCache(maxsize=200000, ttl=3600)

    yolo_model = YOLO("yolov8n.pt")  # cache automático

    # GroundDino
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

    # CLIP
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device)
    clip_model.eval()

    return {
        "embeddings_model": embeddings_model,
        "roberta_classifier_text": roberta_classifier_text,
        "ner_model": ner_model,
        "bart_classifier": bart_classifier,
        "nlp": nlp,
        "cache": cache,
        "yolo_model": yolo_model,
        "gdino_model": gdino_model,
        "gdino_processor": gdino_processor,
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess
    }

# Cargar MODELS al momento de la importación
MODELS = init_models()
