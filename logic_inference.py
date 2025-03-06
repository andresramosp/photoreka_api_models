import time
import torch
from datasets import Dataset
from sentence_transformers import util
from transformers import pipeline
from nltk.stem import WordNetLemmatizer
import inflect
from summarizer import Summarizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
# import asyncio
import concurrent.futures

nltk.download('punkt_tab')

# Importar dependencias comunes desde api.py
from api import embeddings_model, roberta_classifier_text, cache, bart_classifier

def generate_groups_for_tags(data: dict):
    batch_size = 16
    threshold = 0.2
    tags = data.get("tags", [])
    groups = data.get("groups", ['person', 'objects', 'animals', 'places', 'feeling', 'weather', 'symbols', 'concept or idea'])
    candidate_groups = [f"main subject is a {group}" for group in groups]
    
    def process_batch(batch_tags):
        batch_result = {}
        for tag in batch_tags:
            res = bart_classifier(tag, candidate_groups)
            if res['scores'][0] < threshold:
                best_group = "misc"
            else:
                best_group = res['labels'][0].replace("main subject is a ", "")
            batch_result[tag] = best_group
        return batch_result

    final_results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, tags[i:i+batch_size]) for i in range(0, len(tags), batch_size)]
        for future in concurrent.futures.as_completed(futures):
            final_results.update(future.result())
    
    return [f"{tag} | {group}" for tag, group in final_results.items()]


def extractive_summarize_text(data: dict):

    ratio = data.get("ratio", 0.9)
    texts = data.get("texts", [])
    
    if not texts or not isinstance(texts, list):
        raise ValueError("Falta el campo requerido 'texts' o no es una lista.")
    
    model = Summarizer()
    summaries = []
    
    for text in texts:
        summary = model(text, ratio=ratio)
        summaries.append(summary)
    
    return {"summaries": summaries}


def preprocess_text(text, to_singular=False):
    lemmatizer = WordNetLemmatizer()
    p = inflect.engine()
    normalized_text = text.lower().replace('_', ' ')
    words = normalized_text.split()
    
    if len(words) == 1:
        lemmatized_word = lemmatizer.lemmatize(normalized_text)
        return (p.singular_noun(lemmatized_word) or lemmatized_word) if to_singular else lemmatized_word
    
    if to_singular:
        words = [p.singular_noun(word) or word for word in words]
    
    return ' '.join(words)

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
        with torch.inference_mode():
            batch_results = roberta_classifier_text(dataset["query"], batch_size=batch_size)
        for i, result in zip(indexes_to_infer, batch_results):
            cache[batch_queries[i]] = result
            cached_results.insert(i, result)
    return cached_results

def adjust_tags_proximities_by_context_inference_logic(data: dict):
    start_time = time.perf_counter()
    BATCH_SIZE = 128

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
        if label == "entailment":
            adjusted_score = 1 + score
        elif label == "neutral":
            adjusted_score = score
        else:
            adjusted_score = -score

        results[tag_name] = {"adjusted_proximity": adjusted_score, "label": label, "score": score}
        if label == "entailment":
            print(f"✅ [TAG MATCH] {tag_name} -> {term}: {label.upper()} con score {score:.4f}")

    print(f"⏳ Tiempo de ejecución: {time.perf_counter() - start_time:.4f} segundos")
    return results

def adjust_descs_proximities_by_context_inference_logic(data: dict):
    BATCH_SIZE = 128

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
        if label == "entailment":
            adjusted_score = 1 + score
        elif label == "neutral":
            adjusted_score = score
        elif label == "contradiction":
            adjusted_score = -score
        else:
            adjusted_score = 0

        results[chunk_name] = {"adjusted_proximity": adjusted_score, "label": label, "score": score}
        if label == "entailment":
            print(f"✅ [DESC MATCH] {chunk_name} -> {term}: {label.upper()} con score {score:.4f}")

    return results

def get_embeddings_logic(data: dict):
    tags = data.get("tags", [])
    if not tags or not isinstance(tags, list):
        raise ValueError("Field 'tags' must be a list.")
    embeddings = embeddings_model.encode(tags, convert_to_tensor=False)
    return {"tags": tags, "embeddings": [emb.tolist() for emb in embeddings]}

def purge_text(text: str, purge_list: list) -> str:
    """
    Elimina del texto todas las ocurrencias de las cadenas de purge_list.
    Primero se hace una eliminación directa de coincidencias exactas y luego,
    tokenizando el texto, se eliminan aquellos tokens cuya similitud semántica con
    algún elemento de purge_list sea >= 0.85.
    """
    # Eliminación directa de coincidencias exactas
    for candidate in purge_list:
        text = text.replace(candidate, "")

    return text
    
    # Tokenización y eliminación por similitud semántica
    tokens = word_tokenize(text)
    # Obtener embeddings de la lista de cadenas a purgar
    candidate_embeddings = embeddings_model.encode(purge_list, convert_to_tensor=False)
    # Obtener embeddings para cada token
    token_embeddings = embeddings_model.encode(tokens, convert_to_tensor=False)
    
    tokens_to_keep = []
    for token, token_emb in zip(tokens, token_embeddings):
        remove = False
        for cand_emb in candidate_embeddings:
            if util.cos_sim(token_emb, cand_emb).item() >= 0.45:
                remove = True
                break
        if not remove:
            tokens_to_keep.append(token)
    
    # Reconstruir el texto utilizando el detokenizador de NLTK
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    detokenizer = TreebankWordDetokenizer()
    cleaned_text = detokenizer.detokenize(tokens_to_keep)
    return cleaned_text


def clean_texts(data: dict) -> list:
    texts = data.get("texts")
    purge_list = data.get("purge_list")
    extract_ratio = data.get("extract_ratio", 0.9)

    # Resumir todos los textos con una única instancia del modelo
    model = Summarizer()
    summaries = [model(text, ratio=extract_ratio) for text in texts]
    
    # Precomputar las embeddings de purge_list para reutilizarlas
    # candidate_embeddings = embeddings_model.encode(purge_list, convert_to_tensor=False)
    
    # Limpiar los resúmenes en paralelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        cleaned_texts = list(executor.map(lambda s: purge_text(s, purge_list), summaries))
    return cleaned_texts

