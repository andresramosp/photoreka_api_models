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
    return embeddings_model, roberta_classifier_text

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
    
    # Common prefix phrases
    PREFIXES = [
        "photos of", "images of", "pictures of", "I want to see images of", "show me pictures with", 
        "I'm looking for an image of", "I need a photo where", "an image with", "a photo that shows", 
        "I would like to explore pictures of", "photos resembling", "photos similar to", "photos inspired by", 
        "photos evoking", "photos reminiscent of", "photos capturing the essence of", "photos reflecting", 
        "photos resonating with", "images resembling", "images similar to", "images inspired by", 
        "images evoking", "images reminiscent of", "pictures resembling", "pictures similar to", 
        "pictures inspired by", "pictures reflecting", "pictures resonating with"
    ]

    PREFIX_EMBEDDINGS = embeddings_model.encode(PREFIXES, convert_to_tensor=True)
    
    words = query.lower().split()
    
    # Try removing progressive sequences of 2 to 6 words
    for n in range(2, 7):
        if len(words) >= n:
            segment = " ".join(words[:n])
            segment_embedding = embeddings_model.encode(segment, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(segment_embedding, PREFIX_EMBEDDINGS)[0]
            print(f"🧐 Similarity for segment '{segment}': {similarities.tolist()}")
            
            if any(similarity.item() > 0.85 for similarity in similarities):
                print(f"✅ Prefix detected and removed: {segment}")
                return " ".join(words[n:]).strip()
    
    print("❌ No irrelevant prefix detected.")
    return query

def clean_segment(segment, nlp):
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

def segment_query(query):
    query = remove_prefix(query)
    
    # Initialize spaCy model
    nlp = spacy.load("en_core_web_sm")  # Load English model
    doc = nlp(query)
    
    segments = []
    used_objects = set()  # Track attached direct objects to avoid duplication
    
    for chunk in doc.noun_chunks:  # Extract noun phrases
        segments.append(chunk.text)
    
    # Extract verbs and attach them to the closest noun phrase
    verbs = [token for token in doc if token.pos_ == "VERB"]
    
    # Attach direct objects (dobj) without removing subjects (nsubj)
    for verb in verbs:
        closest_chunk = min(doc.noun_chunks, key=lambda chunk: abs(chunk.start - verb.i), default=None)
        if closest_chunk:
            verb_with_object = f"{closest_chunk.text} {verb.text}"  # Keep subject with verb
            attached_object = None
            for child in verb.children:
                if child.dep_ == "dobj":  # Detect direct objects
                    verb_with_object += f" {child.text}"
                    attached_object = child.text
            
            # Replace the noun chunk with the verb phrase
            segments[segments.index(closest_chunk.text)] = verb_with_object
            
            # Remove duplicate object if it exists as a separate segment
            if attached_object and attached_object in segments:
                segments.remove(attached_object)
    
    # Clean unnecessary connectors from each segment
    cleaned_segments = [clean_segment(segment, nlp) for segment in segments]
    
    # Remove duplicate words and redundant segments
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




load_wordnet()
embeddings_model, roberta_classifier_text = load_embeddings_model()
asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    uvicorn.run(asgi_app, host="0.0.0.0", port=5000, reload=True)
