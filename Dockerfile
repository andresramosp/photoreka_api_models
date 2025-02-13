# Usamos una imagen base optimizada para PyTorch con CUDA
FROM python:3.10

# Definir el directorio de trabajo
WORKDIR /app

# Copiar archivos de la API
COPY . /app

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelos de NLTK y spaCy
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
RUN python -m spacy download en_core_web_sm

# Pre-descargar modelos de Sentence Transformers
RUN python -c "from sentence_transformers import SentenceTransformer; \
               model = SentenceTransformer('all-mpnet-base-v2'); \
               model.save('/app/models/all-mpnet-base-v2')"

# Pre-descargar modelos de Hugging Face (para evitar descargas en cada arranque)
RUN python -c "from transformers import pipeline, AutoTokenizer; \
               pipeline('text-classification', model='roberta-large-mnli', \
               tokenizer=AutoTokenizer.from_pretrained('roberta-large-mnli', use_fast=True))"

RUN python -c "from transformers import pipeline; \
               pipeline('ner', model='FacebookAI/xlm-roberta-large-finetuned-conll03-english', \
               aggregation_strategy='simple')"

# Definir variables de entorno
ENV MODEL_PATH="/app/models/all-mpnet-base-v2"

# Exponer el puerto de la API
EXPOSE 7860

# Ejecutar la API con Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
