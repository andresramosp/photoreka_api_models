from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import wordnet as wn
import nltk
from nltk.stem import WordNetLemmatizer
import uvicorn
import spacy

# Descargar datos de WordNet si no están disponibles
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Solo usamos POS tagging

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


# Lista de conectores para dividir la query
CONNECTORS = {"in", "at", "with", "near", "on", "under", "over", "between", "and"}

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
    ("ADJ_NOUN", [is_adjective, is_noun]),  # Adjetivo + Sustantivo
    ("NOUN_PREP_NOUN", [is_noun, is_preposition, is_noun]),  # Sustantivo + Preposición + Sustantivo
    ("NOUN_VERB", [is_noun, is_verb]),  # Sustantivo + Verbo
    ("NOUN_VERB_CD", [is_noun, is_verb, is_noun]),  # Sustantivo + Verbo + CD
    ("NOUN_VERB_ADJ", [is_noun, is_verb, is_adjective]),  # Sustantivo + Verbo + Adjetivo
    ("ADV_ADJ_NOUN", [is_adverb, is_adjective, is_noun]),  # Adverbio + Adjetivo + Sustantivo
    ("VERB_ALONE", [is_verb]),  # Verbo solo (gerundios, acciones sueltas)
    ("NOUN_ALONE", [is_noun])  # Sustantivo solo
]

def preprocess_query(query: str):
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