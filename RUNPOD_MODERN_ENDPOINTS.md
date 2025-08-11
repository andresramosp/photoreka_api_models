# ModernCE Endpoints para RunPod Serverless

Los nuevos endpoints modernos con ModernCE-large-nli están ahora disponibles en el serverless de RunPod.

## Nuevas operaciones disponibles

### 1. `adjust_tags_proximities_by_context_inference_modern`

### 2. `adjust_descs_proximities_by_context_inference_modern`

## Formato de payload para RunPod

```json
{
  "input": {
    "operation": "adjust_tags_proximities_by_context_inference_modern",
    "data": {
      "term": "cat",
      "tag_list": [
        { "name": "animal", "group": "animals" },
        { "name": "dog", "group": "animals" }
      ],
      "premise_wrapper": "the photo featured {term}",
      "hypothesis_wrapper": "the photo featured {term}"
    }
  }
}
```

## Ejemplo de uso con curl

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "operation": "adjust_tags_proximities_by_context_inference_modern",
      "data": {
        "term": "cat",
        "tag_list": [
          {"name": "animal", "group": "animals"},
          {"name": "dog", "group": "animals"}
        ],
        "premise_wrapper": "the photo featured {term}",
        "hypothesis_wrapper": "the photo featured {term}"
      }
    }
  }'
```

## Ejemplo de uso con Python

```python
import requests

url = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

payload = {
    "input": {
        "operation": "adjust_tags_proximities_by_context_inference_modern",
        "data": {
            "term": "cat",
            "tag_list": [
                {"name": "animal", "group": "animals"},
                {"name": "dog", "group": "animals"}
            ],
            "premise_wrapper": "the photo featured {term}",
            "hypothesis_wrapper": "the photo featured {term}"
        }
    }
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()
print(result)
```

## Respuesta esperada

```json
{
  "delayTime": 1234,
  "executionTime": 5678,
  "id": "request-id",
  "output": {
    "animal": {
      "adjusted_proximity": 1.85,
      "label": "entailment",
      "score": 0.85,
      "all_scores": {
        "contradiction": 0.05,
        "entailment": 0.85,
        "neutral": 0.1
      }
    },
    "dog": {
      "adjusted_proximity": 0.65,
      "label": "neutral",
      "score": 0.65,
      "all_scores": {
        "contradiction": 0.15,
        "entailment": 0.2,
        "neutral": 0.65
      }
    }
  },
  "status": "COMPLETED"
}
```

## Modelos cargados en el serverless

El serverless ahora carga:

- `roberta_classifier_text` (modelo original)
- `modern_ce_classifier` (ModernCE-large-nli)

## Ventajas en RunPod

1. **Escalabilidad**: Los endpoints se benefician del auto-scaling de RunPod
2. **Eficiencia**: ModernCE es más eficiente que RoBERTa para inferencia
3. **Precisión**: Mayor accuracy en las predicciones NLI
4. **Compatibilidad**: Los endpoints originales siguen funcionando

## Consideraciones de rendimiento

- **Cold start**: El primer request puede tardar más por la carga de modelos
- **Warm requests**: Requests subsecuentes son mucho más rápidos
- **Batch processing**: El código maneja batches internamente para eficiencia
- **Memory**: ModernCE requiere ~1.5GB de VRAM adicional

## Migración desde endpoints originales

Para migrar del endpoint original al moderno:

```diff
payload = {
    "input": {
-       "operation": "adjust_tags_proximities_by_context_inference",
+       "operation": "adjust_tags_proximities_by_context_inference_modern",
        "data": { ... }
    }
}
```

La estructura de datos de entrada es idéntica, solo cambia el output que incluye información adicional.
