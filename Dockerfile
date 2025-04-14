FROM python:3.10

RUN apt-get update && apt-get install -y libgl1

WORKDIR /app
COPY . /app

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Descargar recursos NLTK
RUN python -m nltk.downloader punkt averaged_perceptron_tagger punkt_tab averaged_perceptron_tagger_eng

# Descargar modelo spaCy
RUN python -m spacy download en_core_web_sm

# === Descargas de modelos comentadas ===

# Sentence Transformers (se descargará en runtime si no está cacheado)
# RUN mkdir -p /runpod-volume/models && \
#     python -c "from sentence_transformers import SentenceTransformer; \
#     SentenceTransformer('all-mpnet-base-v2', cache_folder='/runpod-volume/models')"

# HuggingFace - roberta-large-mnli
# RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
#     AutoTokenizer.from_pretrained('roberta-large-mnli', cache_dir='/runpod-volume/models'); \
#     AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli', cache_dir='/runpod-volume/models')"

# HuggingFace - NER xlm-roberta
# RUN python -c "from transformers import AutoModelForTokenClassification, AutoTokenizer; \
#     AutoModelForTokenClassification.from_pretrained('FacebookAI/xlm-roberta-large-finetuned-conll03-english', cache_dir='/runpod-volume/models'); \
#     AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large-finetuned-conll03-english', cache_dir='/runpod-volume/models')"

# HuggingFace - bart
# RUN python -c "from transformers import pipeline; \
#     pipeline('zero-shot-classification', model='facebook/bart-large-mnli', cache_dir='/runpod-volume/models')"

# Grounding DINO
# RUN python -c "from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor; \
#     AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base', cache_dir='/runpod-volume/models'); \
#     AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base', cache_dir='/runpod-volume/models')"

# OpenCLIP
# ENV TORCH_HOME=/runpod-volume/models
# RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"

# YOLOv8
# RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Comando final: ejecutar handler serverless
CMD ["python", "serverless.py"]
