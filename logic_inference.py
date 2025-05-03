import time
import torch
from datasets import Dataset
from sentence_transformers import util
import nltk
import re
import concurrent.futures
import models

def remove_leading_articles(text: str) -> str:
    return re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)

def generate_groups_for_tags(data: dict):
    batch_size = 16
    threshold = 0.2
    tags = data.get("tags", [])
    groups = data.get("groups", ['person', 'objects', 'animals', 'places', 'feeling', 'weather', 'symbols', 'concept or idea'])
    candidate_groups = [f"main subject is a {group}" for group in groups]
    bart_classifier = models.MODELS["bart_classifier"]
    
    def process_batch(batch_tags):
        batch_result = {}
        for tag in batch_tags:
            res = bart_classifier(tag, candidate_groups)
            best_group = "misc" if res['scores'][0] < threshold else res['labels'][0].replace("main subject is a ", "")
            batch_result[tag] = best_group
        return batch_result

    final_results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, tags[i:i+batch_size]) for i in range(0, len(tags), batch_size)]
        for future in concurrent.futures.as_completed(futures):
            final_results.update(future.result())
    return [f"{tag} | {group}" for tag, group in final_results.items()]

def extract_tags_spacy(text: str, allowed_groups: list):
    nlp = models.MODELS["nlp"]
    doc = nlp(text)
    tags = {ent.text for ent in doc.ents}
    tags |= {chunk.text for chunk in doc.noun_chunks}
    all_tags = generate_groups_for_tags({"tags": list(tags), "groups": allowed_groups})
    filtered_tags = [item for item in all_tags if item.split(" | ")[1] in allowed_groups]
    def remove_leading_articles(text: str) -> str:
        return re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    return [f"{remove_leading_articles(tag.split(' | ')[0].strip())} | {tag.split(' | ')[1]}" for tag in filtered_tags]

def extract_tags_ntlk(text: str):
    sentences = nltk.sent_tokenize(text)
    tags = []
    grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        tree = cp.parse(tagged)
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            tags.append(" ".join(word for word, tag in subtree.leaves()))
    return generate_groups_for_tags({"tags": list(set(tags))})

def extractive_summarize_text(data: dict):
    from summarizer import Summarizer
    ratio = data.get("ratio", 0.9)
    texts = data.get("texts", [])
    if not texts or not isinstance(texts, list):
        raise ValueError("Falta el campo requerido 'texts' o no es una lista.")
    model = Summarizer()
    return {"summaries": [model(text, ratio=ratio) for text in texts]}

def preprocess_text(text, to_singular=False):
    from nltk.stem import WordNetLemmatizer
    import inflect
    lemmatizer = WordNetLemmatizer()
    p = inflect.engine()
    normalized_text = text.lower().replace('_', ' ')
    # Eliminar el punto final si está presente
    if normalized_text.endswith('.'):
        normalized_text = normalized_text[:-1]
    words = normalized_text.split()
    if len(words) == 1:
        lemmatized = lemmatizer.lemmatize(normalized_text)
        return (p.singular_noun(lemmatized) or lemmatized) if to_singular else lemmatized
    if to_singular:
        words = [p.singular_noun(word) or word for word in words]
    return ' '.join(words)


def combine_tag_name_with_group(tag):
    if tag.get("group") == "symbols":
        return f"{tag['name']} (symbol or sign)"
    if tag.get("group") == "environment":
        return f"{tag['name']} (place)"
    if tag.get("group") == "abstract concept":
        return f"{tag['name']} (as general topic)"
    if tag.get("group") == "objects":
        return f"{tag['name']} (physical thing)"
    return tag["name"]

def cached_inference(batch_queries, batch_size):
    cache = models.MODELS["cache"]
    roberta_classifier_text = models.MODELS["roberta_classifier_text"]
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
    premise = data.get("premise_wrapper", "the photo featured {term}")
    hypothesis = data.get("hypothesis_wrapper", "the photo featured {term}")
    if not term or not tag_list:
        raise ValueError("Missing required fields (term, tag_list)")
    batch_queries = [
        f"{premise.format(term=preprocess_text(combine_tag_name_with_group(tag)))} [SEP] {hypothesis.format(term=term)}"
        for tag in tag_list
    ]
    tag_names = [tag['name'] for tag in tag_list]
    batch_results = cached_inference(batch_queries, BATCH_SIZE)
    results = {}
    for tag_name, result in zip(tag_names, batch_results):
        label = result["label"].lower()
        score = result["score"]
        adjusted = 1 + score if label == "entailment" else score if label == "neutral" else -score
        results[tag_name] = {"adjusted_proximity": adjusted, "label": label, "score": score}
    print(f"⏳ [Adjust Tags Proximities - {term}] Tiempo de ejecución para {len(batch_queries)}: {time.perf_counter() - start_time:.4f} segundos")

    torch.cuda.empty_cache()
    return results

def adjust_descs_proximities_by_context_inference_logic(data: dict):
    start_time = time.perf_counter()

    BATCH_SIZE = 128
    term = preprocess_text(data.get("term", ""), True)
    chunk_list = data.get("tag_list", [])
    premise = data.get("premise_wrapper", "the photo has the following fragment in its description: '{term}'")
    hypothesis = data.get("hypothesis_wrapper", "the photo features {term}")
    if not term or not chunk_list:
        raise ValueError("Missing required fields (term, tag_list)")
    batch_queries = [
        f"{premise.format(term=chunk['name'])} [SEP] {hypothesis.format(term=term)}"
        for chunk in chunk_list
    ]
    chunk_names = [chunk['name'] for chunk in chunk_list]
    batch_results = cached_inference(batch_queries, BATCH_SIZE)
    results = {}
    for chunk_name, result in zip(chunk_names, batch_results):
        label = result["label"].lower()
        score = result["score"]
        if label == "entailment":
            adjusted = 1 + score
        elif label == "neutral":
            adjusted = score
        elif label == "contradiction":
            adjusted = -score
        else:
            adjusted = 0
        results[chunk_name] = {"adjusted_proximity": adjusted, "label": label, "score": score}
    print(f"⏳ [Adjust Descs Proximities - {term}] Tiempo de ejecución para {len(batch_queries)}: {time.perf_counter() - start_time:.4f} segundos")

    torch.cuda.empty_cache()
    return results

def get_embeddings_logic(data: dict):
    print("ENTRA EN EL PUTO METODO")
    start_time = time.perf_counter()
    tags = data.get("tags", [])
    if not tags or not isinstance(tags, list):
        raise ValueError("Field 'tags' must be a list.")
    embeddings_model = models.MODELS["embeddings_model"]
    with torch.inference_mode():
        embeddings_tensor = embeddings_model.encode(tags, batch_size=16, convert_to_tensor=True)
    embeddings = embeddings_tensor.cpu().tolist()
    print(f"⏳ [Get Embeddings] Tiempo de ejecución: {time.perf_counter() - start_time:.4f} segundos")

    return {"tags": tags, "embeddings": embeddings}

def purge_text(text: str, purge_list: list) -> str:
    for candidate in purge_list:
        text = text.replace(candidate, "")
    return text

def clean_texts(data: dict) -> list:
    texts = data.get("texts")
    purge_list = data.get("purge_list")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(lambda s: purge_text(s, purge_list), texts))
