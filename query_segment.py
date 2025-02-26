from pydantic import BaseModel
import re
from itertools import cycle
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sentence_transformers import util

# Importar dependencias comunes desde api.py
from api import embeddings_model, nlp, ner_model

# Instanciar lemmatizer una única vez
lemmatizer = WordNetLemmatizer()

# Clase de ejemplo, en caso de necesitarla para validación o uso en otros módulos
class QueryRequest(BaseModel):
    query: str


def remove_photo_prefix(query: str):
    print(f"🔍 Processing query: {query}")

    PREFIX_EMBEDDINGS = embeddings_model.encode(QUERY_PREFIXES, convert_to_tensor=True)
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

def remove_dumb_connectors(query: str) -> str:
    parts = re.split(r'(\[.*?\])', query)
    processed_parts = []
    for part in parts:
        if part.startswith("[") and part.endswith("]"):
            processed_parts.append(part)
        else:
            words = part.split()
            filtered = [w for w in words if w.lower() not in DUMB_CONNECTORS]
            processed_parts.append(" ".join(filtered))
    return "".join(processed_parts)

def split_query_with_connectors(query: str):
    parts = re.split(r'(\[.*?\])', query)
    segments = []
    for part in parts:
        if part.startswith("[") and part.endswith("]"):
            segments.append(part)
        else:
            words = part.lower().replace(",", "").split()
            current_segment = []
            for word in words:
                if word in CONNECTORS:
                    if current_segment:
                        segments.append(" ".join(current_segment))
                        current_segment = []
                else:
                    current_segment.append(word)
            if current_segment:
                segments.append(" ".join(current_segment))
    print(f"\nSegmentos por conectores: {segments}")
    return segments

def get_pos_spacy_no_context(word):
    doc = nlp(word)
    pos = doc[0].pos_
    return pos if pos in ["ADJ", "NOUN", "VERB", "ADV", "ADP"] else "UNKNOWN"

def get_pos_spacy(word: str, sentence: str = None) -> str:
    if sentence:
        doc = nlp(sentence)
        for token in doc:
            if token.text.lower() == word.lower():
                pos = token.pos_
                return pos if pos in ["ADJ", "NOUN", "VERB", "ADV", "ADP"] else "UNKNOWN"
    else:
        print("DEBUG: No se proporcionó oración; analizando la palabra de forma aislada.")
    doc = nlp(word)
    pos = doc[0].pos_
    return pos if pos in ["ADJ", "NOUN", "VERB", "ADV", "ADP"] else "UNKNOWN"

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
    for phrase in PREDEFINES_BLOCKED:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        query = pattern.sub(lambda m: f"[{m.group(0)}]", query)
    return query

def block_er_entities(query: str, use_model: bool = True) -> str:
    if use_model:
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
    no_prefix_query = remove_photo_prefix(query)
    print(f" No-prefix query: {no_prefix_query}")

    query_with_intentional_brackets = replace_quotes_by_brackets(no_prefix_query)
    print(f" Query with intentional brackets: {query_with_intentional_brackets}")

    blocked_predefined_query = block_predefined(query_with_intentional_brackets)
    print(f" Blocked predefined query: {blocked_predefined_query}")

    blocked_er_query = block_er_entities(blocked_predefined_query)
    print(f" Blocked ER query: {blocked_er_query}")

    clean_query = remove_dumb_connectors(blocked_er_query)
    print(f" Clean query: {clean_query}")

    splitted_query = split_query_with_connectors(clean_query)
    print(f" Splitted query: {splitted_query}")

    return splitted_query, no_prefix_query

def remove_contained_segments_simple(segments_str: str) -> list[str]:
    segments = [s.strip() for s in segments_str.split("|") if s.strip()]
    segments = list(dict.fromkeys(segments))
    
    final_segments = []
    for seg in segments:
        if not any(seg != other and seg in other for other in segments):
            final_segments.append(seg)
    return final_segments

def minimal_segment_cover(segments_str: str, original: str) -> list[str]:
    segments = [s.strip() for s in segments_str.split("|") if s.strip()]
    segments = list(dict.fromkeys(segments))
    
    seg_to_words = {seg: set(seg.split()) for seg in segments}
    universe = set().union(*seg_to_words.values())
    
    best_cover = None
    best_cover_size = float('inf')
    best_cover_total_words = float('inf')
    n = len(segments)
    
    for i in range(1, 1 << n):
        subset = [segments[j] for j in range(n) if (i >> j) & 1]
        covered = set()
        for seg in subset:
            covered |= seg_to_words[seg]
        if covered == universe:
            subset_size = len(subset)
            total_words = sum(len(seg.split()) for seg in subset)
            if subset_size < best_cover_size or (subset_size == best_cover_size and total_words < best_cover_total_words):
                best_cover = subset
                best_cover_size = subset_size
                best_cover_total_words = total_words

    if best_cover is None:
        return segments
    # Ordenar según el orden en que aparecen en la query original
    best_cover.sort(key=lambda s: original.find(s))
    return best_cover

# Lista de conectores para dividir la query
CONNECTORS = {"in", "at", "with", "near", "on", "under", "over", "between", "and"}

# Lista de estructuras de segmentos a detectar
PATTERNS = [
    ("ADJ_NOUN", [is_adjective, is_noun]),  # lazy girl
    ("NOUN_ADJ", [is_noun, is_adjective]),  # market sunny
    ("ADJ_NOUN_VERB", [is_adjective, is_noun, is_verb]),  # lazy girl sleeping
    ("NOUN_PREP_NOUN", [is_noun, is_preposition, is_noun]),  # concept of chaos
    ("NOUN_VERB", [is_noun, is_verb]),  # girl sleeping
    ("NOUN_NOUN", [is_noun, is_noun]),  # farm animals
    ("NOUN_VERB_CD", [is_noun, is_verb, is_noun]),  # girl eating icecream
    ("NOUN_VERB_ADJ_NOUN", [is_noun, is_verb, is_adjective, is_noun]),  # girl eating nice icecream
    ("NOUN_VERB_ADJ", [is_noun, is_verb, is_adjective]),  # girl working hard
    ("VERB_ALONE", [is_verb]),  # sleeping
    ("VERB_PREP_NOUN", [is_verb, is_preposition, is_noun]),  # surrounded by animals
    ("NOUN_ALONE", [is_noun]),  # girl
]

# Usar patrón tipo {persona} with {traje|objeto}, de forma que solo definamos las personas y los objetos
# Diferenciar estos bloques de las Named Entities para cuando se haga la expansión semántica
PREDEFINES_BLOCKED = [
    "men in suits",
    "man in suits",
    "woman in red",
    "man in black",
    "woman in white",
    "boy in jeans",
    "girl in dress",
    "man in tie",
    "woman in scarf",
    "child in costume",
    "man in uniform",
    "woman in boots",
    "man with beard",
    "woman with sunglasses",
    "man in leather jacket",
    "woman in blue",
    "man in cap",
    "woman in skirt",
    "man with glasses",
    "woman with braid",
    "man in raincoat",
    "woman in trench coat",
    "man in overcoat",
    "woman in cardigan",
    "man in sneakers",
    "woman in heels",
    "man in casual wear",
    "woman in summer dress",
    "man in denim jacket",
    "woman in vintage dress",
    "man in bomber jacket",
    "woman in elegant gown",
    "man in army uniform",
    "woman in business suit",
    "man in suit and tie",
    "woman in floral dress",
    "man with mustache",
    "woman in red lipstick",
    "man in formal attire",
    "woman in cocktail dress",
    "man in sportswear",
    "woman in workout gear",
    "man in overalls",
    "woman in jumpsuit",
    "man in a hat",
    "woman in a dress",
    "man in cargo pants",
    "woman in high heels",
    "man in a vest",
    "woman in a blazer",
    "man in streetwear",
    "woman in casual wear",
    "man in traditional attire",
    "old man with cane",
    "woman with red hat",
    "man with sunglasses",
    "girl with ponytail",
    "boy with backpack",
    "man with beard",
    "woman with glasses",
    "child with balloon",
    "man with hat",
    "woman with handbag",
    "man with mustache",
    "woman with earrings",
    "man with scars",
    "woman with tattoo",
    "man with guitar",
    "woman with violin",
    "man with camera",
    "woman with smartphone",
    "child with teddy bear",
    "man with walking stick",
    "woman with umbrella",
    "man with pipe",
    "woman with book",
    "man with sword",
    "woman with bracelet",
    "child with toy",
    "old woman with shawl",
    "young man with cap",
    "girl with freckles",
    "boy with glasses",
    "man with tattoo",
    "woman with bun",
    "man with hair gel",
    "child with red balloon",
    "man with backpack",
    "woman with purse",
    "man with frown",
    "woman with smile",
    "man with cigarette",
    "woman with perfume",
    "child with crayons",
    "man with tie",
    "woman with scarf",
    "old man with hat",
    "old woman with cane",
    "man with walking cane",
    "woman with headphones",
    "man with laptop",
    "woman with camera bag",
    "man with briefcase"
]


QUERY_PREFIXES = [
        "photos ", "photos of", "images of", "photos in", "pictures of", "I want to see images of", "show me pictures with", 
        "I'm looking for an image of", "I need a photo where", "an image with", "a photo that shows", 
        "I would like to explore pictures of", "photos resembling", "photos similar to", "photos inspired by", 
        "photos evoking", "photos reminiscent of", "photos capturing the essence of", "photos reflecting", 
        "photos resonating with", "images resembling", "images similar to", "images inspired by", 
        "images evoking", "images reminiscent of", "pictures resembling", "pictures similar to", "photos featuring", "images featuring",
        "pictures inspired by", "pictures reflecting", "pictures resonating with", "images for a project", "images for a series"
]

DUMB_CONNECTORS = {"a", "an", "the", "some", "any", "this", "that", "these", "those", 
    "my", "your", "his", "her", "its", "our", "their", "one", "two", 
    "few", "several", "many", "much", "each", "every", "either", "neither", 
    "another", "such"}


def query_segment(query: str) -> str:
    preprocessed_segments, no_prefix_query = preprocess_query(query)
    all_segments = set()
    processed_segments = set()
    named_entities = list()

    for segment in preprocessed_segments:
        blocked_entities = re.findall(r'\[(.*?)\]', segment)
        for entity in blocked_entities:
            entity_clean = entity.strip()
            if entity_clean:
                print(f"    ✅ Bloqueo detectado: {entity_clean}")
                all_segments.add(entity_clean)
                processed_segments.add(entity_clean)
                named_entities.append(entity_clean)
        
        segment_without_blocks = re.sub(r'\[.*?\]', '', segment)
        words = segment_without_blocks.split()
        
        print(f"\nProcesando segmento: {segment_without_blocks}")
        
        i = 0
        while i < len(words):
            for name, pattern in PATTERNS:
                if i + len(pattern) <= len(words):
                    word_segment = " ".join(words[i:i+len(pattern)])
                    if any(word_segment in larger_segment for larger_segment in processed_segments):
                        continue
                    
                    if all(pattern[j](words[i+j], 'photos of ' + no_prefix_query) for j in range(len(pattern))):
                        all_segments.add(word_segment)
                        processed_segments.add(word_segment)
            i += 1

    segments = " | ".join(all_segments)
    
    return {"positive_segments": minimal_segment_cover(segments, no_prefix_query), "no_prefix": no_prefix_query, "original": query, "named_entities": named_entities}
