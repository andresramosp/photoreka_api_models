from pydantic import BaseModel
import re
from itertools import cycle
import time
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sentence_transformers import util
import models

lemmatizer = WordNetLemmatizer()

class QueryRequest(BaseModel):
    query: str

def remove_photo_prefix(query: str):
    start_time = time.perf_counter()

    embeddings_model = models.MODELS["embeddings_model"]
    QUERY_PREFIXES = [
        "I would like to explore pictures of", "I'm looking for an image of", "I want to see images of",
        "I need a photo where", "show me pictures with", "photos capturing the essence of",
        "photos resonating with", "photos reminiscent of", "photos reflecting", "photos inspired by",
        "images for a series related to", "images for a series about", "images for a series",
        "pictures resonating with", "pictures inspired by", "pictures reflecting", "photos evoking",
        "images reminiscent of", "images resembling", "images inspired by", "photos featuring",
        "images featuring", "pictures similar to", "pictures resembling", "images evoking",
        "images similar to", "photos similar to", "photos resembling", "photos in", "images of",
        "pictures of", "photos of", "an image with", "a photo that shows", "photos for an article",
        "photos for a blog post", "photos for a mood board", "pictures for a creative project",
        "images for social media", "stock photos of", "professional photos of", "aesthetic pictures of"
    ]
    QUERY_PREFIXES = sorted(QUERY_PREFIXES, key=len, reverse=True)
    PREFIX_EMBEDDINGS = embeddings_model.encode(QUERY_PREFIXES, convert_to_tensor=True)
    words = query.lower().split()
    matched_prefix = None
    for n in range(2, 7):
        if len(words) >= n:
            segment = " ".join(words[:n])
            segment_embedding = embeddings_model.encode(segment, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(segment_embedding, PREFIX_EMBEDDINGS)[0]
            if any(sim.item() > 0.78 for sim in similarities):
                matched_prefix = segment
    print(f"⏳  [Remove Photo Prefix - {query}] Tiempo de ejecución: {time.perf_counter() - start_time:.4f} segundos")

    return query[len(matched_prefix):].strip() if matched_prefix else query

def remove_dumb_connectors(query: str) -> str:
    parts = re.split(r'(\[.*?\])', query)
    DUMB_CONNECTORS = {"a", "an", "the", "some", "any", "this", "that", "these", "those", 
                       "my", "your", "his", "her", "its", "our", "their", "one", "two", 
                       "few", "several", "many", "much", "each", "every", "either", "neither", 
                       "another", "such"}
    processed = []
    for part in parts:
        if part.startswith("[") and part.endswith("]"):
            processed.append(part)
        else:
            processed.append(" ".join(w for w in part.split() if w.lower() not in DUMB_CONNECTORS))
    return "".join(processed)

def split_query_with_connectors(query: str):
    parts = re.split(r'(\[.*?\])', query)
    segments = []
    CONNECTORS = {"in", "at", "with", "near", "on", "under", "over", "between", "and"}
    for part in parts:
        if part.startswith("[") and part.endswith("]"):
            segments.append(part)
        else:
            words = part.lower().replace(",", "").split()
            current = []
            for word in words:
                if word in CONNECTORS and current:
                    segments.append(" ".join(current))
                    current = []
                else:
                    current.append(word)
            if current:
                segments.append(" ".join(current))
    print(f"Segmentos por conectores: {segments}")
    return segments

def get_pos_spacy_no_context(word):
    nlp = models.MODELS["nlp"]
    doc = nlp(word)
    return doc[0].pos_ if doc[0].pos_ in ["ADJ", "NOUN", "VERB", "ADV", "ADP"] else "UNKNOWN"

def get_pos_spacy(word: str, sentence: str = None) -> str:
    nlp = models.MODELS["nlp"]
    if sentence:
        doc = nlp(sentence)
        for token in doc:
            if token.text.lower() == word.lower():
                return token.pos_ if token.pos_ in ["ADJ", "NOUN", "VERB", "ADV", "ADP"] else "UNKNOWN"
    else:
        print("DEBUG: No se proporcionó oración; analizando palabra de forma aislada.")
    doc = nlp(word)
    return doc[0].pos_ if doc[0].pos_ in ["ADJ", "NOUN", "VERB", "ADV", "ADP"] else "UNKNOWN"

def is_adjective(word, query=None):
    lemma = lemmatizer.lemmatize(word, pos='a')
    return any(ss.pos() == 'a' for ss in wn.synsets(lemma)) or get_pos_spacy(word, query) == "ADJ"

def is_noun(word, query):
    lemma = lemmatizer.lemmatize(word, pos='n')
    return any(ss.pos() == 'n' for ss in wn.synsets(lemma)) or get_pos_spacy(word, query) == "NOUN"

def is_verb(word, query):
    lemma = lemmatizer.lemmatize(word, pos='v')
    return any(ss.pos() == 'v' for ss in wn.synsets(lemma)) or get_pos_spacy(word, query) == "VERB"

def is_adverb(word, query):
    lemma = lemmatizer.lemmatize(word, pos='r')
    return any(ss.pos() == 'r' for ss in wn.synsets(lemma)) or get_pos_spacy(word, query) == "ADV"

def is_preposition(word, query):
    return word in {"of", "by"}

def block_predefined(query: str) -> str:
    PREDEFINES_BLOCKED = [
        "men in suits", "man in suits", "woman in red", "man in black", "woman in white",
        "boy in jeans", "girl in dress", "man in tie", "woman in scarf", "child in costume",
        "man in uniform", "woman in boots", "man with beard", "woman with sunglasses"
        # ... (lista completa)
    ]
    for phrase in PREDEFINES_BLOCKED:
        query = re.sub(re.escape(phrase), lambda m: f"[{m.group(0)}]", query, flags=re.IGNORECASE)
    return query

def block_er_entities(query: str, use_model: bool = True) -> str:
    if use_model:
        ner_model = models.MODELS["ner_model"]
        entities = ner_model(query)
        for ent in sorted(entities, key=lambda x: x['start'], reverse=True):
            start, end = ent['start'], ent['end']
            query = query[:start] + f"[{query[start:end]}]" + query[end:]
        return query
    else:
        pattern = r'\b(?:[A-Z][a-z]*(?:\s+[A-Z][a-z]*)+)\b'
        return re.sub(pattern, lambda m: f"[{m.group(0)}]", query)

def replace_quotes_by_brackets(text: str) -> str:
    replacer = cycle(["[", "]"])
    return re.sub(r"[\"']", lambda m: next(replacer), text)

def preprocess_query(query: str):
    no_prefix = remove_photo_prefix(query)
    print(f"No-prefix query: {no_prefix}")
    with_quotes = replace_quotes_by_brackets(no_prefix)
    print(f"Query con comillas reemplazadas: {with_quotes}")
    blocked = block_predefined(with_quotes)
    print(f"Query con bloques predefinidos: {blocked}")
    blocked_er = block_er_entities(blocked)
    print(f"Query con entidades bloqueadas: {blocked_er}")
    clean = remove_dumb_connectors(blocked_er)
    print(f"Query limpio: {clean}")
    segments = split_query_with_connectors(clean)
    print(f"Query segmentado: {segments}")
    return segments, no_prefix

def remove_contained_segments_simple(segments_str: str) -> list:
    segments = [s.strip() for s in segments_str.split("|") if s.strip()]
    segments = list(dict.fromkeys(segments))
    final_segments = [seg for seg in segments if not any(seg != other and seg in other for other in segments)]
    return final_segments

def minimal_segment_cover(segments_str: str, original: str) -> list:
    segments = [s.strip() for s in segments_str.split("|") if s.strip()]
    segments = list(dict.fromkeys(segments))
    seg_to_words = {seg: set(seg.split()) for seg in segments}
    universe = set().union(*seg_to_words.values())
    best_cover, best_size, best_total = None, float('inf'), float('inf')
    n = len(segments)
    for i in range(1, 1 << n):
        subset = [segments[j] for j in range(n) if (i >> j) & 1]
        covered = set().union(*(seg_to_words[s] for s in subset))
        if covered == universe:
            size, total = len(subset), sum(len(s.split()) for s in subset)
            if size < best_size or (size == best_size and total < best_total):
                best_cover, best_size, best_total = subset, size, total
    if best_cover is None:
        return segments
    best_cover.sort(key=lambda s: original.find(s))
    return best_cover

def query_segment(query: str) -> dict:
    preprocessed, no_prefix = preprocess_query(query)
    all_segments, processed, named_entities = set(), set(), []
    for segment in preprocessed:
        for entity in re.findall(r'\[(.*?)\]', segment):
            if entity.strip():
                print(f"Bloque detectado: {entity.strip()}")
                all_segments.add(entity.strip())
                processed.add(entity.strip())
                named_entities.append(entity.strip())
        segment_clean = re.sub(r'\[.*?\]', '', segment)
        words = segment_clean.split()
        print(f"Procesando segmento: {segment_clean}")
        for i in range(len(words)):
            for name, pattern in [
                ("ADJ_NOUN", [is_adjective, is_noun]),
                ("NOUN_ADJ", [is_noun, is_adjective]),
                ("ADJ_NOUN_VERB", [is_adjective, is_noun, is_verb]),
                ("NOUN_PREP_NOUN", [is_noun, is_preposition, is_noun]),
                ("NOUN_VERB", [is_noun, is_verb]),
                ("NOUN_NOUN", [is_noun, is_noun]),
                ("NOUN_VERB_CD", [is_noun, is_verb, is_noun]),
                ("NOUN_VERB_ADJ_NOUN", [is_noun, is_verb, is_adjective, is_noun]),
                ("NOUN_VERB_ADJ", [is_noun, is_verb, is_adjective]),
                ("VERB_ALONE", [is_verb]),
                ("VERB_PREP_NOUN", [is_verb, is_preposition, is_noun]),
                ("NOUN_ALONE", [is_noun]),
            ]:
                if i + len(pattern) <= len(words):
                    word_seg = " ".join(words[i:i+len(pattern)])
                    if any(word_seg in larger for larger in processed):
                        continue
                    if all(pattern[j](words[i+j], 'photos of ' + no_prefix) for j in range(len(pattern))):
                        all_segments.add(word_seg)
                        processed.add(word_seg)
    segments_str = " | ".join(all_segments)
    return {"positive_segments": minimal_segment_cover(segments_str, no_prefix),
            "no_prefix": no_prefix,
            "original": query,
            "named_entities": named_entities}
