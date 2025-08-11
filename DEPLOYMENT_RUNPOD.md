# Deployment RunPod - ModernCE Endpoints

## Archivos modificados para soporte ModernCE

### 1. `serverless_logic.py`

- ✅ Añadidas importaciones de funciones modernas
- ✅ Incluido `modern_ce_classifier` en la carga de modelos
- ✅ Añadidos handlers para las nuevas operaciones

### 2. `models.py`

- ✅ Añadido modelo `modern_ce_classifier` con CrossEncoder
- ✅ Configurado cache y device management

### 3. `logic_inference.py`

- ✅ Implementadas funciones `*_modern` con ModernCE
- ✅ Manejo de scores detallados y formato de tuplas

### 4. `requirements.txt`

- ✅ `sentence-transformers` ya incluido

## Instrucciones de deployment

### 1. Build del container

```bash
# En el directorio del proyecto
docker build -f Dockerfile_Logic -t photoreka-logic-modern .
```

### 2. Push a registry

```bash
docker tag photoreka-logic-modern:latest YOUR_REGISTRY/photoreka-logic-modern:latest
docker push YOUR_REGISTRY/photoreka-logic-modern:latest
```

### 3. Deploy en RunPod

1. Crear nuevo endpoint en RunPod Console
2. Usar la imagen: `YOUR_REGISTRY/photoreka-logic-modern:latest`
3. Configurar:
   - **GPU**: RTX 4090 o superior (recomendado)
   - **RAM**: 8GB mínimo, 16GB recomendado
   - **Container Disk**: 10GB mínimo
   - **Handler**: Usar el default (`runpod.serverless.start`)

### 4. Variables de entorno (opcional)

```bash
TRANSFORMERS_CACHE=/runpod-volume/models
HF_HOME=/runpod-volume/models
```

## Testing después del deployment

### 1. Test básico con ping

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"operation": "ping"}}'
```

### 2. Test ModernCE endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "operation": "adjust_tags_proximities_by_context_inference_modern",
      "data": {
        "term": "cat",
        "tag_list": [{"name": "animal", "group": "animals"}],
        "premise_wrapper": "the photo featured {term}",
        "hypothesis_wrapper": "the photo featured {term}"
      }
    }
  }'
```

## Monitoring

### Logs típicos esperados:

```
Inicializando modelos comunes...
[OK] roberta_classifier_text cargado.
[OK] modern_ce_classifier cargado.
[OK] Cache TTL inicializada.
```

### Métricas de rendimiento:

- **Cold start**: ~10-15 segundos (carga de modelos)
- **Warm inference**: ~1-3 segundos por request
- **Memory usage**: ~3-4GB VRAM total

## Troubleshooting

### Error: "modern_ce_classifier not found"

- Verificar que `modern_ce_classifier` esté en la lista de modelos a cargar
- Comprobar logs de carga de modelos

### Error: "sentence_transformers not found"

- Verificar que `sentence-transformers` esté en requirements.txt
- Rebuild el container

### Timeout en cold start

- Aumentar timeout en RunPod a 60-120 segundos
- Considerar usar GPU más potente

### Memory issues

- Usar GPUs con más VRAM (A100, RTX 4090)
- Reducir batch_size en las funciones si es necesario

## Rollback plan

Si hay problemas con los endpoints modernos:

1. Los endpoints originales siguen funcionando
2. Puede comentar `modern_ce_classifier` en `get_models()`
3. Rebuild y redeploy solo con modelos originales
4. Los clientes pueden seguir usando operaciones sin `_modern`

## Next steps

1. Deploy y test en staging
2. Comparar rendimiento vs endpoints originales
3. Update clients gradualmente a endpoints modernos
4. Monitor memory usage y costs
