FROM python:3.10

WORKDIR /app
COPY . /app

# Instalar dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelos de NLTK
RUN python -m nltk.downloader punkt averaged_perceptron_tagger punkt_tab averaged_perceptron_tagger_eng

# Descargar modelo spaCy
RUN python -m spacy download en_core_web_sm

# Pre-descargar Sentence Transformer
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# Pre-descargar modelo roberta-large-mnli para text classification
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('roberta-large-mnli'); \
    AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')"

# Pre-descargar modelo NER (xlm-roberta)
RUN python -c "from transformers import pipeline; pipeline('ner', model='FacebookAI/xlm-roberta-large-finetuned-conll03-english', aggregation_strategy='simple')"

# Pre-descargar modelo zero-shot classification - BART
RUN python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"

# Pre-descargar Grounding DINO (modelo y processor)
RUN python -c "from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor; \
    AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base'); \
    AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')"

# Pre-descargar YOLOv8
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Ejecutar el handler de RunPod Serverless
CMD ["python", "serverless.py"]
