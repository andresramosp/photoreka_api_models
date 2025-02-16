import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import torch
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
import uvicorn
import nltk
from nltk.stem import WordNetLemmatizer
import inflect
import time
import spacy
from spacy.matcher import Matcher
from cachetools import TTLCache
import re

app = FastAPI()
cache = TTLCache(maxsize=200000, ttl=3600)

# Funciones de inicialización y helpers (se mantienen igual)
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
        return p.singular_noun(lemmatized_word) or lemmatized_word if to_singular else lemmatized_word
    
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
        dataset = Dataset.from_dict({"query": queries_to_infer})
        # print(queries_to_infer)
        batch_results = roberta_classifier_text(dataset["query"], batch_size=batch_size)
        for i, result in zip(indexes_to_infer, batch_results):
            cache[batch_queries[i]] = result
            cached_results.insert(i, result)
    return cached_results

# Función auxiliar movida fuera del endpoint
def combine_tag_name_with_group(tag):
    if tag.get("group") == "symbols":
        return f"{tag['name']} (symbol or sign)"
    if tag.get("group") == "culture":
        return f"{tag['name']} culture"
    if tag.get("group") == "location":
        return f"{tag['name']} (place)"
    if tag.get("group") == "generic":
        return f"{tag['name']} (as general topic)"
    if tag.get("group") == "theme":
        return f"{tag['name']} (as general theme)"
    if tag.get("group") == "objects":
        return f"{tag['name']} (physical thing)"
    return tag["name"]

NEGATORS_SET = {"no", "not", "without", "except", "excluding", "with no"}
NEGATORS_REGEX = r'\b(?:' + '|'.join(NEGATORS_SET) + r')\b'

def remove_prefix(query):
    print(f"🔍 Processing query: {query}")
    PREFIXES = [
        "photos ", "photos of", "images of", "pictures of", "I want to see images of", "show me pictures with", 
        "I'm looking for an image of", "I need a photo where", "an image with", "a photo that shows", 
        "I would like to explore pictures of", "photos resembling", "photos similar to", "photos inspired by", 
        "photos evoking", "photos reminiscent of", "photos capturing the essence of", "photos reflecting", 
        "photos resonating with", "images resembling", "images similar to", "images inspired by", 
        "images evoking", "images reminiscent of", "pictures resembling", "pictures similar to", "photos featuring", "images featuring",
        "pictures inspired by", "pictures reflecting", "pictures resonating with", "images for a project", "images for a series"
    ]
    PREFIX_EMBEDDINGS = embeddings_model.encode(PREFIXES, convert_to_tensor=True)
    words = query.lower().split()
    for n in range(2, 7):
        if len(words) >= n:
            segment = " ".join(words[:n])
            segment_embedding = embeddings_model.encode(segment, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(segment_embedding, PREFIX_EMBEDDINGS)[0]
            if any(similarity.item() > 0.8 for similarity in similarities):
                print(f"✅ Prefix detected and removed: {segment}")
                return " ".join(query.split()[n:]).strip()
    print("❌ No irrelevant prefix detected.")
    return query

def clean_segment(segment, nlp):
    doc = nlp(segment)
    filtered_words = []
    for token in doc:
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
            if word.lower() in NEGATORS_SET or word not in seen_words:
                filtered_words.append(word)
                seen_words.add(word)
        cleaned_segment = " ".join(filtered_words)
        if cleaned_segment:
            unique_segments.append(cleaned_segment)
    return unique_segments

def extract_named_entities_and_remove(query):
    ner_results = ner_model(query)
    ne_list = [res["word"] for res in ner_results]
    query_without_ne = query
    for ent in ne_list:
        query_without_ne = query_without_ne.replace(ent, "")
    return query_without_ne.strip(), ner_results

def extract_prepositional_phrases_and_remove(query):
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
    ner_results = ner_model(segment)
    if ner_results:
        best = max(ner_results, key=lambda x: x.get("score", 0))
        return best.get("entity_group", best.get("label", "OTHER"))
    return "OTHER"

def extract_negative_terms(query):
    query_lower = query.lower()
    neg_terms = re.findall(NEGATORS_REGEX, query_lower)
    return set(neg_terms)

def remove_negators(segment):
    cleaned = re.sub(NEGATORS_REGEX, '', segment, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(" ,;:-")
    return cleaned

def segment_query(query):
    query_no_prefix = remove_prefix(query)
    query_no_ne, ner_results = extract_named_entities_and_remove(query_no_prefix)
    query_clean, pp_list = extract_prepositional_phrases_and_remove(query_no_ne)
    doc = nlp(query_clean)
    noun_chunks = list(doc.noun_chunks)
    segments = [chunk.text for chunk in noun_chunks]

    for verb in [token for token in doc if token.pos_ == "VERB"]:
        closest_chunk = min(noun_chunks, key=lambda chunk: abs(chunk.start - verb.i), default=None)
        if closest_chunk:
            verb_with_object = f"{closest_chunk.text} {verb.text}"
            attached_object = None
            for child in verb.children:
                if child.dep_ == "dobj":
                    verb_with_object += f" {child.text}"
                    attached_object = child.text
            try:
                idx = segments.index(closest_chunk.text)
                segments[idx] = verb_with_object
            except ValueError:
                continue
            if attached_object and attached_object in segments:
                segments.remove(attached_object)
    
    cleaned_segments = [clean_segment(segment, nlp) for segment in segments]
    final_segments = remove_duplicate_words(cleaned_segments)
    
    for res in ner_results:
        ent_text = res["word"]
        if ent_text and ent_text not in final_segments:
            final_segments.append(ent_text)
    for pp in pp_list:
        if pp and pp not in final_segments:
            final_segments.append(pp)
    
    final_segments_cleaned = []
    positive_segments = []
    negative_segments = []
    for seg in final_segments:
        cleaned = remove_negators(seg)
        final_segments_cleaned.append(cleaned)
        if re.search(NEGATORS_REGEX, seg, re.IGNORECASE):
            negative_segments.append(cleaned)
        else:
            positive_segments.append(cleaned)
    
    structured_query = " | ".join(final_segments_cleaned)
    types = [get_segment_type(seg) for seg in final_segments_cleaned]
    return structured_query, types, positive_segments, negative_segments, query_no_prefix

# --- Funciones de lógica para cada operación ---

def adjust_tags_proximities_by_context_inference_logic(data: dict):
    start_time = time.perf_counter()
    BATCH_SIZE = 128
    THRESHOLD = 0 # 0.82

    term = preprocess_text(data.get("term", ""), True)
    tag_list = data.get("tag_list", [])
    premise_wrapper = data.get("premise_wrapper", "the photo featured {term}")
    hypothesis_wrapper = data.get("hypothesis_wrapper", "the photo featured {term}")

    if not term or not tag_list:
        raise ValueError("Missing required fields (term, tag_list)")

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
        if label == "entailment" and score >= THRESHOLD:
            print(f"✅ [TAG MATCH] {tag_name} -> {term}: {label.upper()} con score {score:.4f}")

    print(f"⏳ Tiempo de ejecución: {time.perf_counter() - start_time:.4f} segundos")
    return results

def adjust_descs_proximities_by_context_inference_logic(data: dict):
    BATCH_SIZE = 128
    THRESHOLD = 0 # 0.55

    term = preprocess_text(data.get("term", ""), True)
    chunk_list = data.get("tag_list", [])
    premise_wrapper = data.get("premise_wrapper", "the photo has the following fragment in its description: '{term}'")
    hypothesis_wrapper = data.get("hypothesis_wrapper", "the photo features {term}")

    if not term or not chunk_list:
        raise ValueError("Missing required fields (term, tag_list)")

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
        if label == "entailment" and score >= THRESHOLD:
            print(f"✅ [DESC MATCH] {chunk_name} -> {term}: {label.upper()} con score {score:.4f}")

    return results

def get_embeddings_logic(data: dict):
    tags = data.get("tags", [])
    if not tags or not isinstance(tags, list):
        raise ValueError("Field 'tags' must be a list.")
    embeddings = embeddings_model.encode(tags, convert_to_tensor=False)
    return {"tags": tags, "embeddings": [emb.tolist() for emb in embeddings]}

def structure_query_logic(data: dict):
    query = data.get("query", "").strip()
    if not query:
        raise ValueError("Missing 'query' field")
    print(f"📥 Received query: {query}")
    structured_query_str, types, positive_segments, negative_segments, query_no_prefix = segment_query(query)
    print(f"📤 Generated response: {structured_query_str}")
    return {
        "clear": structured_query_str,
        "no_prefix": query_no_prefix,
        "types": types,
        "positive_segments": positive_segments,
        "negative_segments": negative_segments
    }

# --- Endpoints de FastAPI ---

@app.post("/adjust_tags_proximities_by_context_inference")
async def adjust_tags_proximities_by_context_inference(request: Request):
    data = await request.json()
    results = adjust_tags_proximities_by_context_inference_logic(data)
    return JSONResponse(content=results)

@app.post("/adjust_descs_proximities_by_context_inference")
async def adjust_descs_proximities_by_context_inference(request: Request):
    data = await request.json()
    results = adjust_descs_proximities_by_context_inference_logic(data)
    return JSONResponse(content=results)

@app.post("/get_embeddings")
async def get_embeddings(request: Request):
    data = await request.json()
    results = get_embeddings_logic(data)
    return JSONResponse(content=results)

@app.post("/structure_query")
async def structure_query(request: Request):
    data = await request.json()
    results = structure_query_logic(data)
    return JSONResponse(content=results)

# Inicializar recursos
load_wordnet()
embeddings_model, roberta_classifier_text, nlp, ner_model = load_embeddings_model()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
