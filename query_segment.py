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
def get_pos_spacy(word):
    doc = nlp(word)
    pos = doc[0].pos_
    return pos if pos in ["ADJ", "NOUN", "VERB", "ADV", "ADP"] else "UNKNOWN"

# Funciones para determinar la categoría gramatical de una palabra con WordNet y Spacy como respaldo
def is_adjective(word):
    lemma = lemmatizer.lemmatize(word, pos='a')
    return any(ss.pos() == 'a' for ss in wn.synsets(lemma)) or get_pos_spacy(word) == "ADJ"

def is_noun(word):
    lemma = lemmatizer.lemmatize(word, pos='n')
    return any(ss.pos() == 'n' for ss in wn.synsets(lemma)) or get_pos_spacy(word) == "NOUN"

def is_verb(word):
    lemma = lemmatizer.lemmatize(word, pos='v')
    return any(ss.pos() == 'v' for ss in wn.synsets(lemma)) or get_pos_spacy(word) == "VERB"

def is_adverb(word):
    lemma = lemmatizer.lemmatize(word, pos='r')
    return any(ss.pos() == 'r' for ss in wn.synsets(lemma)) or get_pos_spacy(word) == "ADV"

def is_preposition(word):
    return word in {"of"}

# Lista de estructuras de segmentos a detectar
patterns = [
    ("ADJ_NOUN", [is_adjective, is_noun]),  # lazy girl
    ("ADJ_NOUN_VERB", [is_adjective, is_noun, is_verb]),  # lazy girl sleeping
    ("NOUN_PREP_NOUN", [is_noun, is_preposition, is_noun]),  # concept of chaos
    ("NOUN_VERB", [is_noun, is_verb]),  # girl sleeping
    ("NOUN_VERB_CD", [is_noun, is_verb, is_noun]),  # girl eating icecream
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
    return splitted_query

def segment_query_v2(query: str) -> str:
    preprocessed_segments = preprocess_query(query)
    all_segments = set()
    
    for segment in preprocessed_segments:
        words = segment.split()
        
        print(f"\nProcesando segmento: {segment}")
        
        i = 0
        while i < len(words):
            for name, pattern in patterns:
                if i + len(pattern) <= len(words):
                    word_segment = words[i:i+len(pattern)]
                    if all(pattern[j](word_segment[j]) for j in range(len(pattern))):
                        segment_text = " ".join(word_segment)
                        print(f"    ✅ Match: {segment_text} ({name})")
                        all_segments.add(segment_text)
            i += 1
    
    return " | ".join(all_segments)

@app.post("/segment-query")
def segment_query(request: QueryRequest):
    result = segment_query_v2(request.query)
    return {"segments": result}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)