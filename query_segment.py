from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import wordnet as wn
import nltk
from nltk.stem import WordNetLemmatizer
import uvicorn
import spacy
from sentence_transformers import SentenceTransformer, util


# Descargar datos de WordNet si no están disponibles
nltk.download("wordnet")
nltk.download("omw-1.4")
embeddings_model = SentenceTransformer('all-mpnet-base-v2', device=0)
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Solo usamos POS tagging

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


# Lista de conectores para dividir la query
CONNECTORS = {"in", "at", "with", "near", "on", "under", "over", "between", "and"}



def remove_photo_prefix(query: str):
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

def remove_dumb_connectors(query: str):
    DUMB_CONNECTORS = {"a", "an", "the", "some", "any", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "one", "two", "few", "several", "many", "much", "each", "every", "either", "neither", "another", "such"}
    
    words = query.split()
    filtered_words = [word for word in words if word.lower() not in DUMB_CONNECTORS]
    
    return " ".join(filtered_words)


# Función de respaldo con Spacy para etiquetar palabras desconocidas
def get_pos_spacy_no_context(word):
    doc = nlp(word)
    pos = doc[0].pos_
    return pos if pos in ["ADJ", "NOUN", "VERB", "ADV", "ADP"] else "UNKNOWN"

def get_pos_spacy(word: str, sentence: str = None) -> str:
    if sentence:
        print("DEBUG: Analizando la oración completa:", sentence)
        doc = nlp(sentence)
        print("DEBUG: Tokens de la oración:")
        for token in doc:
            print(f"    Token: '{token.text}' - POS: {token.pos_}")
        # Buscar el token que coincida con la palabra
        for token in doc:
            if token.text.lower() == word.lower():
                print(f"DEBUG: Coincidencia encontrada para '{word}': Token '{token.text}' con POS: {token.pos_}")
                pos = token.pos_
                final_pos = pos if pos in ["ADJ", "NOUN", "VERB", "ADV", "ADP"] else "UNKNOWN"
                print(f"DEBUG: Retornando: {final_pos}")
                return final_pos
        print(f"DEBUG: La palabra '{word}' no se encontró en la oración. Se analizará de forma aislada.")
    else:
        print("DEBUG: No se proporcionó oración; analizando la palabra de forma aislada.")
    
    # Análisis aislado
    doc = nlp(word)
    print("DEBUG: Token aislado:", doc[0].text, "con POS:", doc[0].pos_)
    pos = doc[0].pos_
    final_pos = pos if pos in ["ADJ", "NOUN", "VERB", "ADV", "ADP"] else "UNKNOWN"
    print(f"DEBUG: Retornando: {final_pos}")
    return final_pos

# Funciones para determinar la categoría gramatical de una palabra con WordNet y Spacy como respaldo
def is_adjective(word, query = None):
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
    return word in {"of"}

# Lista de estructuras de segmentos a detectar
patterns = [
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
    ("NOUN_ALONE", [is_noun])  # girl
]

def split_query_with_connectors(query: str):
    words = query.lower().replace(",", "").split()
    segments = []
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

def preprocess_query(query: str):
    no_prefix_query = remove_photo_prefix(query)
    print(f" No-prefix query: {no_prefix_query}")

    clean_query = remove_dumb_connectors(no_prefix_query)
    print(f" Clean query: {clean_query}")

    splitted_query = split_query_with_connectors(clean_query)
    return splitted_query, no_prefix_query

def segment_query_v2(query: str) -> str:
    preprocessed_segments, no_prefix_query = preprocess_query(query)
    all_segments = set()
    
    for segment in preprocessed_segments:
        words = segment.split()
        
        print(f"\nProcesando segmento: {segment}")
        
        i = 0
        while i < len(words):
            for name, pattern in patterns:
                if i + len(pattern) <= len(words):
                    word_segment = words[i:i+len(pattern)]
                    if all(pattern[j](word_segment[j], 'photos of ' + no_prefix_query) for j in range(len(pattern))):
                        segment_text = " ".join(word_segment)
                        print(f"    ✅ Match: {segment_text} ({name})")
                        all_segments.add(segment_text)
            i += 1
    
    return " | ".join(all_segments)

def remove_contained_segments_simple(segments_str: str) -> list[str]:
    # Separamos la cadena por "|" y limpiamos espacios
    segments = [s.strip() for s in segments_str.split("|") if s.strip()]
    # Eliminamos duplicados manteniendo el orden
    segments = list(dict.fromkeys(segments))
    
    final_segments = []
    for seg in segments:
        if not any(seg != other and seg in other for other in segments):
            final_segments.append(seg)
    return final_segments

def minimal_segment_cover(segments_str: str) -> list[str]:
    # Separamos la cadena y eliminamos duplicados
    segments = [s.strip() for s in segments_str.split("|") if s.strip()]
    segments = list(dict.fromkeys(segments))
    
    # Asociamos cada segmento a su conjunto de palabras (en minúsculas)
    seg_to_words = {seg: set(seg.split()) for seg in segments}
    # Universo: todas las palabras presentes en algún segmento
    universe = set().union(*seg_to_words.values())
    
    best_cover = None
    best_cover_size = float('inf')
    best_cover_total_words = float('inf')
    n = len(segments)
    
    # Recorremos todos los subconjuntos de segmentos (brute force, n pequeño)
    for i in range(1, 1 << n):
        subset = [segments[j] for j in range(n) if (i >> j) & 1]
        covered = set()
        for seg in subset:
            covered |= seg_to_words[seg]
        if covered == universe:
            subset_size = len(subset)
            total_words = sum(len(seg.split()) for seg in subset)
            # Prioriza el menor número de segmentos, y en empate, el de menos palabras
            if subset_size < best_cover_size or (subset_size == best_cover_size and total_words < best_cover_total_words):
                best_cover = subset
                best_cover_size = subset_size
                best_cover_total_words = total_words

    if best_cover is None:
        return segments
    # Ordenamos según el orden de aparición original
    best_cover.sort(key=lambda s: segments.index(s))
    return best_cover


@app.post("/segment-query")
def segment_query(request: QueryRequest):
    result = segment_query_v2(request.query)
    filtered_result = minimal_segment_cover(result)
    return {"segments": result, "filtered_segments": filtered_result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)