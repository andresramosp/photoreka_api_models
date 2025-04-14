FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# NLTK
RUN python -m nltk.downloader punkt averaged_perceptron_tagger punkt_tab averaged_perceptron_tagger_eng

# spaCy
RUN python -m spacy download en_core_web_sm

# Sentence Transformers
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-mpnet-base-v2', cache_folder='/workspace/models')"

# roberta-large-mnli
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('roberta-large-mnli', cache_dir='/workspace/models'); \
AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli', cache_dir='/workspace/models')"

# NER - xlm-roberta
RUN python -c "from transformers import pipeline; \
pipeline('ner', model='FacebookAI/xlm-roberta-large-finetuned-conll03-english', aggregation_strategy='simple', cache_dir='/workspace/models')"

# Zero-shot - bart
RUN python -c "from transformers import pipeline; \
pipeline('zero-shot-classification', model='facebook/bart-large-mnli', cache_dir='/workspace/models')"

# Grounding DINO
RUN python -c "from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor; \
AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base', cache_dir='/workspace/models'); \
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base', cache_dir='/workspace/models')"

# YOLOv8
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"  # cache_dir no soportado directamente

CMD ["python", "serverless.py"]
