import os
import numpy as np
import torch
import torch.nn.functional as F
import base64
import json
import io
import uvicorn
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageOps
from transformers import AutoProcessor

# Configuración
SAVE_DIR = "processed_images"
PROCESSED_DIR = "final_inputs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Inicialización de FastAPI
app = FastAPI()

# Carga el procesador (sin modelo, solo el preprocesamiento)
model_dir = 'allenai/Molmo-7B-D-0924'
processor = AutoProcessor.from_pretrained(
    model_dir, trust_remote_code=True, torch_dtype="auto", device_map="auto"
)

def process_batch(prompt, images_list, front_preprocessed=False, ids=[]):
    batch_texts = [f"User: {prompt} Assistant:" for _ in images_list]
    tokens_list = [processor.tokenizer.encode(" " + text, add_special_tokens=False) for text in batch_texts]
    outputs_list = []
    images_kwargs = {
        "max_crops": 12,
        "overlap_margins": [4, 4],
        "base_image_input_size": [336, 336],
        "image_token_length_w": 12,
        "image_token_length_h": 12,
        "image_patch_size": 14,
        "image_padding_mask": True,
    }
    
    for i, image in enumerate(images_list):
        tokens = tokens_list[i]
        image_id = ids[i]
        if not front_preprocessed:
            image = image.convert("RGB")
            image = ImageOps.exif_transpose(image)
            images_array = [np.array(image)]
            out = processor.image_processor.multimodal_preprocess(
                images=images_array,
                image_idx=[-1],
                tokens=np.asarray(tokens).astype(np.int32),
                sequence_length=1536,
                image_patch_token_id=processor.special_token_ids["<im_patch>"],
                image_col_token_id=processor.special_token_ids["<im_col>"],
                image_start_token_id=processor.special_token_ids["<im_start>"],
                image_end_token_id=processor.special_token_ids["<im_end>"],
                **images_kwargs,
            )
            
            # Guardar la imagen después del preprocesamiento
            processed_image = Image.fromarray((out["images"][0] * 255).astype(np.uint8))
            processed_image_path = os.path.join(PROCESSED_DIR, f"{image_id}_processed.png")
            processed_image.save(processed_image_path)
        else:
            image_arr = np.array(image)
            out = {
                "input_ids": np.asarray(tokens).astype(np.int32),
                "images": image_arr,
                "image_input_idx": np.array([-1]),
                "image_masks": np.ones(image_arr.shape[:2], dtype=np.int32),
            }
        outputs_list.append(out)
    
    batch_outputs = {k: torch.nn.utils.rnn.pad_sequence([torch.from_numpy(out[k]) for out in outputs_list], batch_first=True, padding_value=-1)
                     for k in outputs_list[0].keys()}
    
    bos = processor.tokenizer.bos_token_id or processor.tokenizer.eos_token_id
    batch_outputs["input_ids"] = F.pad(batch_outputs["input_ids"], (1, 0), value=bos)
    
    if "image_input_idx" in batch_outputs:
        image_input_idx = batch_outputs["image_input_idx"]
        batch_outputs["image_input_idx"] = torch.where(
            image_input_idx < 0, image_input_idx, image_input_idx + 1
        )
    return batch_outputs

class RequestPayload(BaseModel):
    inputs: dict

@app.post("/debug-process")
def debug_process(payload: RequestPayload):
    try:
        inputs_data = payload.inputs
        prompt = inputs_data.get("prompt", "Describe this image.")
        front_preprocessed = inputs_data.get("front_preprocessed", False)
        batch_size = inputs_data.get("batch_size", len(inputs_data.get("images", [])))
        
        images_list = []
        ids = []
        
        if "images" in inputs_data and isinstance(inputs_data["images"], list):
            for item in inputs_data["images"]:
                try:
                    image_id = item.get("id", "unknown")
                    b64 = item.get("base64", "")
                    if not b64:
                        continue
                    if b64.startswith("data:image") and "," in b64:
                        _, b64 = b64.split(",", 1)
                    decoded = base64.b64decode(b64)
                    image = Image.open(io.BytesIO(decoded)).convert("RGB")
                    
                    image_path = os.path.join(SAVE_DIR, f"{image_id}.png")
                    image.save(image_path)
                    
                    images_list.append(image)
                    ids.append(image_id)
                except Exception:
                    traceback.print_exc()
                    continue
        else:
            raise HTTPException(status_code=400, detail="Se requiere una lista de imágenes en 'inputs.images'.")
        
        if len(images_list) == 0:
            raise HTTPException(status_code=400, detail="No se pudo cargar ninguna imagen.")
        
        results = []
        
        for start in range(0, len(images_list), batch_size):
            batch_images = images_list[start:start+batch_size]
            batch_ids = ids[start:start+batch_size]
            inputs_batch = process_batch(prompt, batch_images, front_preprocessed, batch_ids)
            
            batch_data = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in inputs_batch.items()
            }
            
            json_path = os.path.join(PROCESSED_DIR, "input_data.json")
            with open(json_path, "w") as f:
                json.dump(batch_data, f, indent=4)
            
            for idx in range(len(batch_ids)):
                results.append({
                    "id": batch_ids[idx],
                    "processed_data": {key: batch_data[key][idx] for key in batch_data}
                })
        
        return {"prompt": prompt, "batch_size": batch_size, "results": results}
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
