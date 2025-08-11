#!/usr/bin/env python3
"""
Script de prueba para los nuevos endpoints modernos en RunPod Serverless
"""

import requests
import json
import time

# Cambia esta URL por la de tu endpoint de RunPod
RUNPOD_ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
RUNPOD_API_KEY = "YOUR_API_KEY"

def test_runpod_modern_endpoints():
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Datos de prueba
    test_data = {
        "term": "cat",
        "tag_list": [
            {"name": "animal", "group": "animals"},
            {"name": "dog", "group": "animals"},
            {"name": "car", "group": "objects"}
        ],
        "premise_wrapper": "the photo featured {term}",
        "hypothesis_wrapper": "the photo featured {term}"
    }
    
    print("🧪 Probando endpoints modernos en RunPod...")
    
    # Test 1: Ping para calentar
    ping_payload = {
        "input": {
            "operation": "ping"
        }
    }
    
    try:
        print("📡 Enviando ping para calentar el pod...")
        response = requests.post(RUNPOD_ENDPOINT_URL, json=ping_payload, headers=headers, timeout=60)
        print(f"✅ Ping - Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Ping result: {result}")
    except Exception as e:
        print(f"❌ Error en ping: {e}")
    
    # Test 2: adjust_tags_proximities_by_context_inference_modern
    tags_payload = {
        "input": {
            "operation": "adjust_tags_proximities_by_context_inference_modern",
            "data": test_data
        }
    }
    
    try:
        print("\n🔍 Probando tags proximities modern...")
        start_time = time.time()
        response = requests.post(RUNPOD_ENDPOINT_URL, json=tags_payload, headers=headers, timeout=120)
        end_time = time.time()
        
        print(f"✅ Tags modern - Status: {response.status_code} - Tiempo: {end_time - start_time:.2f}s")
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Tags result sample: {json.dumps(result, indent=2)[:300]}...")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Error en tags modern: {e}")
    
    # Test 3: adjust_descs_proximities_by_context_inference_modern
    descs_payload = {
        "input": {
            "operation": "adjust_descs_proximities_by_context_inference_modern",
            "data": test_data
        }
    }
    
    try:
        print("\n🔍 Probando descs proximities modern...")
        start_time = time.time()
        response = requests.post(RUNPOD_ENDPOINT_URL, json=descs_payload, headers=headers, timeout=120)
        end_time = time.time()
        
        print(f"✅ Descs modern - Status: {response.status_code} - Tiempo: {end_time - start_time:.2f}s")
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Descs result sample: {json.dumps(result, indent=2)[:300]}...")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Error en descs modern: {e}")

def compare_endpoints():
    """Compara los resultados entre endpoints originales y modernos"""
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    test_data = {
        "term": "cat",
        "tag_list": [
            {"name": "animal", "group": "animals"}
        ],
        "premise_wrapper": "the photo featured {term}",
        "hypothesis_wrapper": "the photo featured {term}"
    }
    
    print("\n🔬 Comparando endpoints originales vs modernos...")
    
    # Test original
    original_payload = {
        "input": {
            "operation": "adjust_tags_proximities_by_context_inference",
            "data": test_data
        }
    }
    
    # Test modern
    modern_payload = {
        "input": {
            "operation": "adjust_tags_proximities_by_context_inference_modern",
            "data": test_data
        }
    }
    
    try:
        # Original
        print("📊 Endpoint original:")
        response_orig = requests.post(RUNPOD_ENDPOINT_URL, json=original_payload, headers=headers, timeout=120)
        if response_orig.status_code == 200:
            result_orig = response_orig.json()
            print(f"   {json.dumps(result_orig, indent=4)}")
        
        # Modern
        print("\n📊 Endpoint moderno:")
        response_mod = requests.post(RUNPOD_ENDPOINT_URL, json=modern_payload, headers=headers, timeout=120)
        if response_mod.status_code == 200:
            result_mod = response_mod.json()
            print(f"   {json.dumps(result_mod, indent=4)}")
            
    except Exception as e:
        print(f"❌ Error en comparación: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando tests de ModernCE endpoints en RunPod...")
    print("\n⚠️  IMPORTANTE: Actualiza RUNPOD_ENDPOINT_URL y RUNPOD_API_KEY antes de ejecutar")
    
    # Descomenta estas líneas después de configurar las credenciales
    # test_runpod_modern_endpoints()
    # compare_endpoints()
    
    print("\n📋 Para usar estos tests:")
    print("1. Actualiza RUNPOD_ENDPOINT_URL con tu endpoint ID")
    print("2. Actualiza RUNPOD_API_KEY con tu clave API")
    print("3. Descomenta las llamadas a las funciones de test")
    print("4. Ejecuta el script")
