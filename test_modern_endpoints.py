#!/usr/bin/env python3
"""
Script de prueba para los nuevos endpoints con ModernCE
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_modern_endpoints():
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
    
    print("🧪 Probando endpoints modernos...")
    
    # Test endpoint 1: adjust_tags_proximities_by_context_inference_modern
    try:
        response = requests.post(
            f"{BASE_URL}/adjust_tags_proximities_by_context_inference_modern",
            json=test_data,
            timeout=30
        )
        print(f"✅ Endpoint tags modern - Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Resultado sample: {json.dumps(result, indent=2)[:200]}...")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Error en tags modern: {e}")
    
    # Test endpoint 2: adjust_descs_proximities_by_context_inference_modern
    try:
        response = requests.post(
            f"{BASE_URL}/adjust_descs_proximities_by_context_inference_modern",
            json=test_data,
            timeout=30
        )
        print(f"✅ Endpoint descs modern - Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Resultado sample: {json.dumps(result, indent=2)[:200]}...")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Error en descs modern: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando tests de ModernCE endpoints...")
    test_modern_endpoints()
