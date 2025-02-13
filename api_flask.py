from flask import Flask, request, jsonify
import torch
from transformers import pipeline, AutoTokenizer
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
import re

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
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU Count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    device = 0 if torch.cuda.is_available() else -1
    embeddings_model = SentenceTransformer('all-mpnet-base-v2', device=device)
    roberta_classifier_text = pipeline(
        "text-classification", 
        model="roberta-large-mnli", 
        tokenizer=AutoTokenizer.from_pretrained("roberta-large-mnli", use_fast=True), 
        device=device
    )
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
    
    if to_singular:
        words = [p.singular_noun(word) or word for word in words]
    
    return ' '.join(words)

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
        # print(f"🔍 [INFERENCE] {queries_to_infer}")
        
        # Aquí es donde probablemente eliminaste el uso de Dataset
        dataset = Dataset.from_dict({"query": queries_to_infer})
        batch_results = roberta_classifier_text(dataset["query"], batch_size=batch_size)

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
        if (label == "entailment" and score >= THRESHOLD):
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
        if (label == "entailment" and score >= THRESHOLD):
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

NEGATORS_SET = {"no", "not", "without", "except", "excluding", "with no"}
NEGATORS_REGEX = r'\b(?:' + '|'.join(NEGATORS_SET) + r')\b'

def get_segment_type(segment, nlp):
    """
    Procesa el segmento con spaCy y, si existe una entidad que cubra la totalidad,
    devuelve su etiqueta; en caso contrario, devuelve "OTHER".
    """
    doc = nlp(segment)
    for ent in doc.ents:
        if ent.start_char == 0 and ent.end_char == len(segment):
            return ent.label_
    return "OTHER"

def remove_prefix(query):
    print(f"🔍 Processing query: {query}")
    PREFIXES = [
        "photos ", "photos of", "images of", "pictures of", "I want to see images of", "show me pictures with", 
        "I'm looking for an image of", "I need a photo where", "an image with", "a photo that shows", 
        "I would like to explore pictures of", "photos resembling", "photos similar to", "photos inspired by", 
        "photos evoking", "photos reminiscent of", "photos capturing the essence of", "photos reflecting", 
        "photos resonating with", "images resembling", "images similar to", "images inspired by", 
        "images evoking", "images reminiscent of", "pictures resembling", "pictures similar to",  "photos featuring", "images featuring",
        "pictures inspired by", "pictures reflecting", "pictures resonating with", "images for a project", "images for a series"
    ]
    PREFIX_EMBEDDINGS = embeddings_model.encode(PREFIXES, convert_to_tensor=True)
    words = query.lower().split()
    for n in range(2, 7):
        if len(words) >= n:
            segment = " ".join(words[:n])
            segment_embedding = embeddings_model.encode(segment, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(segment_embedding, PREFIX_EMBEDDINGS)[0]
            # print(f"🧐 Similarity for segment '{segment}': {similarities.tolist()}")
            if any(similarity.item() > 0.8 for similarity in similarities):
                print(f"✅ Prefix detected and removed: {segment}")
                return " ".join(query.split()[n:]).strip()
    print("❌ No irrelevant prefix detected.")
    return query

def clean_segment(segment, nlp):
    doc = nlp(segment)
    filtered_words = []
    for token in doc:
        # Se conserva el token si es un negador, utilizando el conjunto global
        if token.text.lower() in NEGATORS_SET:
            filtered_words.append(token.text)
        elif token.pos_ not in {"DET", "ADP", "PRON", "AUX", "CCONJ", "SCONJ"}:
            filtered_words.append(token.text)
    return " ".join(filtered_words)

def remove_duplicate_words(segments):
    unique_segments = []
    for segment in segments:
        seen_words = set()
        filtered_words = []
        for word in segment.split():
            # Siempre conservar la palabra si es negadora
            if word.lower() in NEGATORS_SET or word not in seen_words:
                filtered_words.append(word)
                seen_words.add(word)
        cleaned_segment = " ".join(filtered_words)
        if cleaned_segment:
            unique_segments.append(cleaned_segment)
    return unique_segments

def extract_named_entities_and_remove(query):
    """
    Usa ner_model para detectar entidades nombradas y las elimina del query.
    Retorna: (query_sin_entidades, lista_de_resultados del NER)
    """
    ner_results = ner_model(query)  # Cada resultado contiene, por ejemplo, "word", "score", "entity_group"
    ne_list = [res["word"] for res in ner_results]
    query_without_ne = query
    for ent in ne_list:
        query_without_ne = query_without_ne.replace(ent, "")
    return query_without_ne.strip(), ner_results

def extract_prepositional_phrases_and_remove(query):
    """
    Usa el Matcher para detectar secuencias tipo NOUN + ADP + NOUN y las elimina del query.
    Retorna: (query_sin_PP, lista_de_frases_pp)
    """
    from spacy.matcher import Matcher
    doc = nlp(query)
    matcher = Matcher(nlp.vocab)
    pattern = [
        {"POS": "NOUN"},
        {"POS": "ADP"},
        {"POS": "NOUN", "OP": "+"}
    ]
    matcher.add("PrepositionalPhrase", [pattern])
    matches = matcher(doc)
    pp_list = []
    query_without_pp = query
    for match_id, start, end in sorted(matches, key=lambda x: x[1], reverse=True):
        span = doc[start:end]
        pp_list.append(span.text)
        query_without_pp = query_without_pp[:span.start_char] + query_without_pp[span.end_char:]
    return query_without_pp.strip(), pp_list

def get_segment_type(segment):
    """
    Usa ner_model para procesar el segmento y, si se detecta una entidad que cubra el segmento,
    devuelve su etiqueta (entity_group o label); de lo contrario, devuelve "OTHER".
    """
    ner_results = ner_model(segment)
    if ner_results:
        best = max(ner_results, key=lambda x: x.get("score", 0))
        return best.get("entity_group", best.get("label", "OTHER"))
    return "OTHER"

def extract_negative_terms(query):
    """
    Extrae términos negativos del query usando regex.
    Busca patrones como "no <word>", "without <word>", "not <word>" o "except <word>".
    Retorna un conjunto de términos negativos (en minúsculas).
    """
    query_lower = query.lower()
    neg_terms = re.findall(NEGATORS_REGEX, query_lower)
    return set(neg_terms)

def remove_negators(segment):
    """
    Elimina palabras negadoras (usando NEGATORS_REGEX) del segmento y limpia la puntuación sobrante.
    """
    cleaned = re.sub(NEGATORS_REGEX, '', segment, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(" ,;:-")
    return cleaned


def segment_query(query):
    # Paso 1: Eliminar el prefijo
    query = remove_prefix(query)
    
    # Paso 2: Extraer las entidades nombradas usando ner_model y eliminarlas del query.
    query_no_ne, ner_results = extract_named_entities_and_remove(query)
    
    # Paso 3: Extraer las frases preposicionales y eliminarlas del query.
    query_clean, pp_list = extract_prepositional_phrases_and_remove(query_no_ne)
    
    # Paso 4: Segmentar el query limpio usando los noun_chunks de spaCy.
    doc = nlp(query_clean)
    noun_chunks = list(doc.noun_chunks)  # Almacena los chunks una sola vez
    segments = [chunk.text for chunk in noun_chunks]

    # Adjuntar verbos al sintagma nominal más cercano
    for verb in [token for token in doc if token.pos_ == "VERB"]:
        closest_chunk = min(noun_chunks, key=lambda chunk: abs(chunk.start - verb.i), default=None)
        if closest_chunk:
            verb_with_object = f"{closest_chunk.text} {verb.text}"
            attached_object = None
            for child in verb.children:
                if child.dep_ == "dobj":
                    verb_with_object += f" {child.text}"
                    attached_object = child.text
            # Ahora se busca en la lista previamente almacenada
            try:
                idx = segments.index(closest_chunk.text)
                segments[idx] = verb_with_object
            except ValueError:
                continue
            if attached_object and attached_object in segments:
                segments.remove(attached_object)
    
    # Limpiar cada segmento
    cleaned_segments = [clean_segment(segment, nlp) for segment in segments]
    final_segments = remove_duplicate_words(cleaned_segments)
    
    # Paso 5: Agregar al final los bloques extraídos (las entidades y las PP) si no están ya presentes.
    for res in ner_results:
        ent_text = res["word"]
        if ent_text and ent_text not in final_segments:
            final_segments.append(ent_text)
    for pp in pp_list:
        if pp and pp not in final_segments:
            final_segments.append(pp)
    
    # Paso 6: Aplicar limpieza de negadores a cada segmento y clasificar en positivos y negativos.
    final_segments_cleaned = []
    positive_segments = []
    negative_segments = []
    for seg in final_segments:
        cleaned = remove_negators(seg)
        final_segments_cleaned.append(cleaned)
        # Se clasifica como negativo si en el segmento original se detecta algún negador
        if re.search(NEGATORS_REGEX, seg, re.IGNORECASE):
            negative_segments.append(cleaned)
        else:
            positive_segments.append(cleaned)
    
    structured_query = " | ".join(final_segments_cleaned)
    
    # Paso 7: Para cada segmento, obtener el tipo usando ner_model.
    types = []
    for seg in final_segments_cleaned:
        seg_type = get_segment_type(seg)
        types.append(seg_type)
    
    return structured_query, types, positive_segments, negative_segments

@app.route("/structure_query", methods=["POST"])
def structure_query():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400
    print(f"📥 Received query: {query}")
    structured_query, types, positive_segments, negative_segments = segment_query(query)
    print(f"📤 Generated response: {structured_query}")
    return jsonify({
        "clear": structured_query,
        "types": types,
        "positive_segments": positive_segments,
        "negative_segments": negative_segments
    })


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
