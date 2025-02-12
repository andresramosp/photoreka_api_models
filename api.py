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
import spacy
from spacy.matcher import Matcher
from cachetools import TTLCache

app = Flask(__name__)

# Configurar caché con un tamaño máximo de 1000 elementos y TTL de 1 hora
cache = TTLCache(maxsize=200000, ttl=3600)

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
    ner_model = pipeline("ner", model="FacebookAI/xlm-roberta-large-finetuned-conll03-english", aggregation_strategy="simple")
    nlp = spacy.load("en_core_web_sm")
    return embeddings_model, roberta_classifier_text, nlp, ner_model

def preprocess_text(text, to_singular=False):
    lemmatizer = WordNetLemmatizer()
    p = inflect.engine()
    normalized_text = text.lower().replace('_', ' ')

    words = normalized_text.split()
    if len(words) == 1:
        lemmatized_word = lemmatizer.lemmatize(normalized_text)
        if to_singular:
            return p.singular_noun(lemmatized_word) or lemmatized_word
        return lemmatized_word
    return normalized_text

def cached_inference(batch_queries, batch_size):
    cached_results = []
    queries_to_infer = []
    indexes_to_infer = []

    for i, query in enumerate(batch_queries):
        if query in cache:
            cached_results.append(cache[query])
        else:
            queries_to_infer.append(query)
            indexes_to_infer.append(i)

    if queries_to_infer:
        for q in queries_to_infer:
            print(f"🔍 [INFERENCE] {q}")
        batch_results = roberta_classifier_text(queries_to_infer, batch_size=batch_size)
        for i, result in zip(indexes_to_infer, batch_results):
            cache[batch_queries[i]] = result
            cached_results.insert(i, result)

    return cached_results

@app.route("/adjust_tags_proximities_by_context_inference", methods=["POST"])
async def adjust_tags_proximities_by_context_inference():

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

    start_time = time.perf_counter()
    BATCH_SIZE = 128
    THRESHOLD = 0.82

    data = request.json
    term = preprocess_text(data.get("term", ""), True)
    tag_list = data.get("tag_list", [])
    premise_wrapper = data.get("premise_wrapper", "The photo featured {term}") # 'The photo contains a tag {term}'
    hypothesis_wrapper = data.get("hypothesis_wrapper", "The photo featured {term}")

    if not term or not tag_list:
        return jsonify({"error": "Missing required fields (term, tag_list)"}), 400

    batch_queries = [
        f"{premise_wrapper.format(term=preprocess_text(combine_tag_name_with_group(tag)))} [SEP] {hypothesis_wrapper.format(term=term)}"
        for tag in tag_list
    ]
    tag_names = [tag['name'] for tag in tag_list]

    batch_results = cached_inference(batch_queries, BATCH_SIZE)

    results = {}
    for tag_name, result in zip(tag_names, batch_results):
        label = result["label"].lower()
        score = result["score"]
        adjusted_score = score if label == "entailment" and score >= THRESHOLD else -score if label == "contradiction" else 0
        results[tag_name] = {"adjusted_proximity": adjusted_score, "label": label, "score": score}
        if (label == "entailment"):
            print(f"✅ [TAG MATCH] {tag_name} -> {term}: {label.upper()} con score {score:.4f}")
        # else:
        #     print(f"❌ [MATCH] {tag_name} !-> {term}: {label.upper()} con score {score:.4f}")


    print(f"⏳ Tiempo de ejecución: {time.perf_counter() - start_time:.4f} segundos")
    return jsonify(results)

@app.route("/adjust_descs_proximities_by_context_inference", methods=["POST"])
async def adjust_descs_proximities_by_context_inference():
    BATCH_SIZE = 128
    THRESHOLD = 0.55

    data = request.get_json()
    term = preprocess_text(data.get("term", ""), True)
    chunk_list = data.get("tag_list", [])
    premise_wrapper = data.get("premise_wrapper", "the photo has the following fragment in its description: '{term}'")
    hypothesis_wrapper = data.get("hypothesis_wrapper", "the photo features {term}")

    if not term or not chunk_list:
        return jsonify({"error": "Missing required fields (term, tag_list)"}), 400

    batch_queries = [
        f"{premise_wrapper.format(term=chunk['name'])} [SEP] {hypothesis_wrapper.format(term=term)}"
        for chunk in chunk_list
    ]
    chunk_names = [chunk['name'] for chunk in chunk_list]

    batch_results = cached_inference(batch_queries, BATCH_SIZE)

    results = {}
    for chunk_name, result in zip(chunk_names, batch_results):
        label = result["label"].lower()
        score = result["score"]
        adjusted_score = score if label == "entailment" and score >= THRESHOLD else -score if label == "contradiction" else 0
        results[chunk_name] = {"adjusted_proximity": adjusted_score, "label": label, "score": score}
        if (label == "entailment"):
            print(f"✅ [DESC MATCH] {chunk_name} -> {term}: {label.upper()} con score {score:.4f}")
        # else:
        #     print(f"❌ [MATCH] {chunk_name} !-> {term}: {label.upper()} con score {score:.4f}")

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

def remove_prefix(query):
    print(f"🔍 Processing query: {query}")
    
    PREFIXES = [
        "photos of", "images of", "pictures of", "I want to see images of", "show me pictures with", 
        "I'm looking for an image of", "I need a photo where", "an image with", "a photo that shows", 
        "I would like to explore pictures of", "photos resembling", "photos similar to", "photos inspired by", 
        "photos evoking", "photos reminiscent of", "photos capturing the essence of", "photos reflecting", 
        "photos resonating with", "images resembling", "images similar to", "images inspired by", 
        "images evoking", "images reminiscent of", "pictures resembling", "pictures similar to", 
        "pictures inspired by", "pictures reflecting", "pictures resonating with", "photos with"
    ]
    PREFIX_EMBEDDINGS = embeddings_model.encode(PREFIXES, convert_to_tensor=True)
    
    # Conservamos tanto la versión original como la en minúsculas para comparar
    words = query.split()
    words_lower = query.lower().split()
    
    for n in range(2, 7):
        if len(words_lower) >= n:
            segment_lower = " ".join(words_lower[:n])
            segment_embedding = embeddings_model.encode(segment_lower, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(segment_embedding, PREFIX_EMBEDDINGS)[0]
            print(f"🧐 Similarity for segment '{segment_lower}': {similarities.tolist()}")
            if any(similarity.item() > 0.85 for similarity in similarities):
                print(f"✅ Prefix detected and removed: {' '.join(words[:n])}")
                return " ".join(words[n:]).strip()
    
    print("❌ No irrelevant prefix detected.")
    return query

def clean_segment(segment):
    # Si el segmento coincide exactamente con un bloque detectado, lo dejamos intacto
    if segment in locked_texts_global:
        return segment
    doc = nlp(segment)
    filtered_words = [token.text for token in doc if token.pos_ not in {"DET", "ADP", "PRON", "AUX", "CCONJ", "SCONJ"}]
    return " ".join(filtered_words)

def remove_duplicate_words(segments):
    unique_segments = []
    seen_words = set()
    for segment in segments:
        words = segment.split()
        filtered_words = []
        for word in words:
            if word not in seen_words:
                filtered_words.append(word)
                seen_words.add(word)
        cleaned_segment = " ".join(filtered_words)
        if cleaned_segment:
            unique_segments.append(cleaned_segment)
    return unique_segments

def extract_named_entities(query):
    """
    Extrae las entidades nombradas usando el modelo global ner_model.
    Si el span empieza con una preposición, se descarta esa palabra.
    """
    ner_results = ner_model(query)
    doc = nlp(query)
    entities = []
    for i, ent in enumerate(ner_results):
        start_char = ent["start"]
        end_char = ent["end"]
        span = doc.char_span(start_char, end_char)
        if span is None:
            continue
        if span[0].pos_ == "ADP" and len(span) > 1:
            span = doc[span.start+1 : span.end]
        placeholder = f"[NER_{i}]"
        entities.append({"placeholder": placeholder, "text": span.text, "start": span.start_char, "end": span.end_char})
    return entities

def extract_prepositional_phrases(query):
    """
    Utiliza el Matcher de spaCy para detectar secuencias del tipo NOUN + (ADP + NOUN)+
    y las considera como bloques atómicos.
    """
    doc = nlp(query)
    matcher = Matcher(nlp.vocab)
    pattern = [
        {"POS": "NOUN"},
        {"POS": "ADP"},
        {"POS": "NOUN", "OP": "+"}
    ]
    matcher.add("PrepositionalPhrase", [pattern])
    matches = matcher(doc)
    locked_phrases = []
    for match_id, start, end in matches:
        # Evitamos solapamientos: por ejemplo, en "men in suits" preferimos el bloque completo en lugar de uno parcial
        phrase = doc[start:end].text
        placeholder = f"[PP_{start}_{end}]"
        start_char = doc[start].idx
        end_char = doc[end-1].idx + len(doc[end-1].text)
        locked_phrases.append({"placeholder": placeholder, "text": phrase, "start": start_char, "end": end_char})
    return locked_phrases

def replace_entities_with_placeholders(query, locked_items):
    """
    Reemplaza en el query las secuencias detectadas por marcadores, de derecha a izquierda.
    """
    locked_sorted = sorted(locked_items, key=lambda x: x["start"], reverse=True)
    modified_query = query
    for item in locked_sorted:
        modified_query = modified_query[:item["start"]] + item["placeholder"] + modified_query[item["end"]:]
    return modified_query

def reinsert_placeholders(text, locked_items):
    """
    Reemplaza en el texto los marcadores por la secuencia original.
    """
    for item in locked_items:
        text = text.replace(item["placeholder"], item["text"])
    return text

# Variable global para almacenar los textos bloqueados (frases atómicas)
locked_texts_global = set()

def segment_query(query):
    # Paso 1: Eliminar el prefijo antes de cualquier otro procesamiento
    query = remove_prefix(query)
    print(f"Query after prefix removal: {query}")
    
    # Paso 2: Extraer entidades nombradas y bloquearlas con placeholders
    entities = extract_named_entities(query)
    query_temp = replace_entities_with_placeholders(query, entities)
    
    # Paso 3: Extraer frases preposicionales y bloquearlas
    pp_phrases = extract_prepositional_phrases(query_temp)
    query_temp = replace_entities_with_placeholders(query_temp, pp_phrases)
    
    # Actualizamos la variable global con los textos bloqueados
    global locked_texts_global
    locked_texts_global = { item["text"] for item in (entities + pp_phrases) }
    
    # Paso 4: Procesar el query (con placeholders) con spaCy para segmentar
    doc = nlp(query_temp)
    segments = []
    for chunk in doc.noun_chunks:
        segments.append({"text": chunk.text, "start": chunk.start_char})
    
    # Adjuntar verbos al sintagma nominal más cercano (manteniendo el sujeto)
    verbs = [token for token in doc if token.pos_ == "VERB"]
    for verb in verbs:
        closest_chunk = min(doc.noun_chunks, key=lambda chunk: abs(chunk.start - verb.i), default=None)
        if closest_chunk:
            verb_with_object = f"{closest_chunk.text} {verb.text}"
            attached_object = None
            for child in verb.children:
                if child.dep_ == "dobj":
                    verb_with_object += f" {child.text}"
                    attached_object = child.text
            for seg in segments:
                if seg["text"] == closest_chunk.text:
                    seg["text"] = verb_with_object
                    break
            segments = [seg for seg in segments if seg["text"] != attached_object]
    
    # Reinsertar los placeholders en cada segmento
    locked_items = entities + pp_phrases
    final_segments = []
    for seg in segments:
        seg_text = reinsert_placeholders(seg["text"], locked_items)
        final_segments.append(seg_text)
    
    # Limpiar segmentos (excepto si coinciden exactamente con un bloque atómico)
    cleaned_segments = []
    for segment in final_segments:
        if segment in locked_texts_global:
            cleaned_segments.append(segment)
        else:
            cleaned_segments.append(clean_segment(segment))
    
    final_segments = remove_duplicate_words(cleaned_segments)
    structured_query = " | ".join(final_segments)
    print(f"🔹 Segmented query after cleaning: {structured_query}")
    return structured_query

@app.route("/structure_query", methods=["POST"])
def clean_query():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400
    print(f"📥 Received query: {query}")
    clear_query = segment_query(query)
    print(f"📤 Generated response: {clear_query}")
    return jsonify({"clear": clear_query})


# Endpoint extra para probar únicamente el modelo NER
@app.route("/test_ner", methods=["POST"])
def test_ner():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400
    
    print(f"🔍 Testing NER for query: {query}")
    ner_results = ner_model(query)
    print("NER output:", ner_results)
    words = [entity["word"] for entity in ner_results]
    return jsonify({"ner": words})


load_wordnet()
embeddings_model, roberta_classifier_text, nlp, ner_model = load_embeddings_model()
asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    uvicorn.run(asgi_app, host="0.0.0.0", port=5000, reload=True)
