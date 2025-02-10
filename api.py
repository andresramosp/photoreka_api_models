from flask import Flask, request, jsonify
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
from asgiref.wsgi import WsgiToAsgi
import uvicorn
import nltk
from nltk.stem import WordNetLemmatizer
import inflect
import time

app = Flask(__name__)

def load_wordnet():
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

def load_embeddings_model():
    device = 0 if torch.cuda.is_available() else -1
    embeddings_model = SentenceTransformer('all-mpnet-base-v2', device=device)
    roberta_classifier_text = pipeline("text-classification", model="roberta-large-mnli", device=device)
    return embeddings_model, roberta_classifier_text

def preprocess_text(text, to_singular=False):
    lemmatizer = WordNetLemmatizer()
    p = inflect.engine()
    normalized_text = text.lower().replace('_', ' ')

    words = normalized_text.split()
    if len(words) == 1:  # Apply lemmatization only for single words
        lemmatized_word = lemmatizer.lemmatize(normalized_text)
        if to_singular:
            return p.singular_noun(lemmatized_word) or lemmatized_word
        return lemmatized_word
    return normalized_text

def combine_tag_name_with_group(tag):
    if tag.get("group") == "symbols":
        return f"{tag['name']} (symbol or sign)"
    if tag.get("group") == "culture":
        return f"{tag['name']} culture"
    if tag.get("group") == "location":
        return f"{tag['name']} (place)"
    if tag.get("group") == "generic":
        return f"{tag['name']} (as general topic)"
    if tag.get("group") == "objects":
        return f"{tag['name']} (physical thing)"
    return tag["name"]

@app.route("/adjust_tags_proximities_by_context_inference", methods=["POST"])
async def adjust_tags_proximities_by_context_inference():
    start_time = time.perf_counter()  # Iniciar medición de tiempo

    BATCH_SIZE = 128
    THRESHOLD = 0.82  # TODO: ajustar umbrales según el tipo de query

    data = request.json
    term = preprocess_text(data.get("term", ""), True)
    tag_list = data.get("tag_list", [])
    premise_wrapper = data.get("premise_wrapper", "The photo featured {term}") 
    hypothesis_wrapper = data.get("hypothesis_wrapper", "The photo featured {term}")

    if not term or not tag_list:
        return jsonify({"error": "Missing required fields (term, tag_list)"}), 400

    # print(f"[INFO] Ejecutando inferencia de proximidad para: '{term}' con {len(tag_list)} etiquetas")

    batch_queries = []
    tag_names = []

    for tag in tag_list:
        premise_text = premise_wrapper.format(term=preprocess_text(combine_tag_name_with_group(tag)))
        hypothesis_text = hypothesis_wrapper.format(term=term)
        query_text = f"{premise_text} [SEP] {hypothesis_text}"

        batch_queries.append(query_text)
        tag_names.append(tag['name'])

        # print(f"[DEBUG] Inferencia: Premisa = '{premise_text}', Hipótesis = '{hypothesis_text}'")

    # Usar Dataset de Hugging Face para optimizar el batch processing
    dataset = Dataset.from_dict({"text": batch_queries})
    batch_results = roberta_classifier_text(dataset["text"], batch_size=BATCH_SIZE)

    results = {}
    for tag_name, result in zip(tag_names, batch_results):
        label = result["label"].lower()
        score = result["score"]

        # Aplicar umbral mínimo
        if score >= THRESHOLD:
            adjusted_score = score if label == "entailment" else -score if label == "contradiction" else 0
        else:
            adjusted_score = 0  # Si el score es menor al umbral, se trata como neutral

        results[tag_name] = {
            "adjusted_proximity": adjusted_score,
            "label": label,
            "score": score
        }

        if label == "entailment" and score >= THRESHOLD:
            print(f"✅ [MATCH] {tag_name}: {label.upper()} con score {score:.4f}")

    end_time = time.perf_counter()  # Fin de medición de tiempo
    elapsed_time = end_time - start_time
    print(f"⏳ Tiempo de ejecución: {elapsed_time:.4f} segundos")

    return jsonify(results)


@app.route("/adjust_descs_proximities_by_context_inference", methods=["POST"])
async def adjust_descs_proximities_by_context_inference():
    BATCH_SIZE = 128
    THRESHOLD = 0.55

    data = request.get_json()
    term = preprocess_text(data.get("term", ""), True)
    tag_list = data.get("tag_list", [])
    premise_wrapper = data.get("premise_wrapper", "the photo has the following fragment in its description: '{term}'")
    hypothesis_wrapper = data.get("hypothesis_wrapper", "the photo features {term}")

    if not term or not tag_list:
        return jsonify({"error": "Missing required fields (term, tag_list)"}), 400

    batch_queries = []
    tag_names = []

    for tag in tag_list:
        premise_text = premise_wrapper.format(term=preprocess_text(combine_tag_name_with_group(tag)))
        hypothesis_text = hypothesis_wrapper.format(term=term)
        query_text = f"{premise_text} [SEP] {hypothesis_text}"

        batch_queries.append(query_text)
        tag_names.append(tag["name"])

        # print(f"[DEBUG] Premise = '{premise_text}', Hypothesis = '{hypothesis_text}'")

    dataset = Dataset.from_dict({"text": batch_queries})
    batch_results = roberta_classifier_text(dataset["text"], batch_size=BATCH_SIZE)

    results = {}
    for tag_name, result in zip(tag_names, batch_results):
        label = result["label"].lower()
        score = result["score"]

        if score >= THRESHOLD:
            adjusted_score = score if label == "entailment" else -score if label == "contradiction" else 0
        else:
            adjusted_score = 0

        results[tag_name] = {
            "adjusted_proximity": adjusted_score,
            "label": label,
            "score": score
        }

        if label == "entailment" and score >= THRESHOLD:
            print(f"✅ [MATCH] {tag_name}: {label.upper()} with score {score:.4f}")

    return jsonify(results)

@app.route("/get_embeddings", methods=["POST"])
async def get_embeddings():
    try:
        data = request.get_json()
        tags = data.get("tags", [])

        if not tags or not isinstance(tags, list):
            return jsonify({"error": "Field 'tags' must be a list."}), 400

        embeddings = embeddings_model.encode(tags, convert_to_tensor=False)
        response = {
            "tags": tags,
            "embeddings": [emb.tolist() for emb in embeddings]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

load_wordnet()
embeddings_model, roberta_classifier_text = load_embeddings_model()


asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    uvicorn.run(asgi_app, host="0.0.0.0", port=5000, reload=True)



