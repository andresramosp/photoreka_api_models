# ModernCE Endpoints

Se han añadido dos nuevos endpoints que utilizan el modelo ModernCE-large-nli de Hugging Face para inferencia de lenguaje natural (NLI). Este modelo es más moderno, eficiente y preciso que el modelo RoBERTa-MNLI utilizado anteriormente.

## Nuevos Endpoints

### 1. `/adjust_tags_proximities_by_context_inference_modern`

Versión moderna del endpoint de proximidad de tags que utiliza ModernCE-large-nli.

**Características:**

- Utiliza CrossEncoder de sentence-transformers
- Soporte para secuencias hasta 8192 tokens
- Mayor precisión (92% en MNLI, 91% en SNLI)
- Retorna scores detallados para todas las categorías (contradiction, entailment, neutral)

### 2. `/adjust_descs_proximities_by_context_inference_modern`

Versión moderna del endpoint de proximidad de descripciones.

## Diferencias con endpoints originales

| Aspecto         | Original (RoBERTa-MNLI)    | Modern (ModernCE-large-nli) |
| --------------- | -------------------------- | --------------------------- |
| Arquitectura    | RoBERTa-large              | ModernBERT-large            |
| Parámetros      | ~355M                      | 395M                        |
| Contexto máximo | 512 tokens                 | 8192 tokens                 |
| Precisión MNLI  | ~90%                       | 92.02%                      |
| Precisión SNLI  | ~91%                       | 91.10%                      |
| Formato entrada | "premise [SEP] hypothesis" | tupla (premise, hypothesis) |
| Salida          | {label, score}             | {label, score, all_scores}  |

## Formato de respuesta

Los endpoints modernos retornan información más detallada:

```json
{
  "tag_name": {
    "adjusted_proximity": 1.85,
    "label": "entailment",
    "score": 0.85,
    "all_scores": {
      "contradiction": 0.05,
      "entailment": 0.85,
      "neutral": 0.1
    }
  }
}
```

## Ejemplo de uso

```python
import requests

data = {
    "term": "cat",
    "tag_list": [
        {"name": "animal", "group": "animals"},
        {"name": "dog", "group": "animals"}
    ],
    "premise_wrapper": "the photo featured {term}",
    "hypothesis_wrapper": "the photo featured {term}"
}

response = requests.post(
    "http://localhost:5000/adjust_tags_proximities_by_context_inference_modern",
    json=data
)

result = response.json()
```

## Migración

Los endpoints originales siguen funcionando sin cambios. Los nuevos endpoints son completamente compatibles en términos de parámetros de entrada, solo difieren en el formato de salida que incluye información adicional.

Para migrar a los endpoints modernos, simplemente cambia la URL y opcionalmente aprovecha la información adicional en `all_scores`.
