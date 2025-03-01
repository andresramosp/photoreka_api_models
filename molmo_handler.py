import numpy as np
import torch
import torch.nn.functional as F
import requests
from PIL import Image, ImageOps
import io, base64, json, traceback, time
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

class EndpointHandler:
    def __init__(self, model_dir):
        print("[INFO] Cargando modelo con trust_remote_code=True...")
        self.processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def process_batch(self, prompt, images_list):
        batch_texts = [f"User: {prompt} Assistant:" for _ in images_list]
        tokens_list = [
            self.processor.tokenizer.encode(" " + text, add_special_tokens=False)
            for text in batch_texts
        ]
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
        for i in range(len(batch_texts)):
            tokens = tokens_list[i]
            image = images_list[i].convert("RGB")
            image = ImageOps.exif_transpose(image)
            images_array = [np.array(image)]
            out = self.processor.image_processor.multimodal_preprocess(
                images=images_array,
                image_idx=[-1],
                tokens=np.asarray(tokens).astype(np.int32),
                sequence_length=1536,
                image_patch_token_id=self.processor.special_token_ids["<im_patch>"],
                image_col_token_id=self.processor.special_token_ids["<im_col>"],
                image_start_token_id=self.processor.special_token_ids["<im_start>"],
                image_end_token_id=self.processor.special_token_ids["<im_end>"],
                **images_kwargs,
            )
            outputs_list.append(out)
        batch_outputs = {}
        for key in outputs_list[0].keys():
            tensors = [torch.from_numpy(out[key]) for out in outputs_list]
            batch_outputs[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=-1
            )
        bos = self.processor.tokenizer.bos_token_id or self.processor.tokenizer.eos_token_id
        batch_outputs["input_ids"] = F.pad(batch_outputs["input_ids"], (1, 0), value=bos)
        if "image_input_idx" in batch_outputs:
            image_input_idx = batch_outputs["image_input_idx"]
            batch_outputs["image_input_idx"] = torch.where(
                image_input_idx < 0, image_input_idx, image_input_idx + 1
            )
        return batch_outputs

    def __call__(self, data=None):
        global_start_time = time.time()
        print("[INFO] Iniciando procesamiento por lotes...")
        if not data:
            return {"error": "El cuerpo de la petición está vacío."}
        if "inputs" not in data:
            return {"error": "Se requiere un campo 'inputs' en la petición JSON."}

        inputs_data = data["inputs"]

        # Se espera un array de prompts
        prompts = inputs_data.get("prompts", [])
        if not prompts or not isinstance(prompts, list):
            return {"error": "Se requiere un array de 'prompts' en la petición JSON."}

        batch_size = inputs_data.get("batch_size", len(inputs_data.get("images", [])))
        print(f"[DEBUG] Prompts: {prompts} | batch_size: {batch_size}")

        images_list = []
        ids = []
        if "images" in inputs_data and isinstance(inputs_data["images"], list):
            for item in inputs_data["images"]:
                try:
                    image_id = item.get("id", "desconocido")
                    b64 = item.get("base64", "")
                    if not b64:
                        continue
                    if b64.startswith("data:image") and "," in b64:
                        _, b64 = b64.split(",", 1)
                    decoded = base64.b64decode(b64)
                    image = Image.open(io.BytesIO(decoded)).convert("RGB")
                    images_list.append(image)
                    ids.append(image_id)
                except Exception:
                    traceback.print_exc()
                    continue
        else:
            return {"error": "Se requiere una lista de imágenes en 'inputs.images'."}

        if len(images_list) == 0:
            return {"error": "No se pudo cargar ninguna imagen."}

        generation_config = inputs_data.get("generation_config", {})
        gen_config = GenerationConfig(
            max_new_tokens=generation_config.get("max_new_tokens", 500),
            min_new_tokens=generation_config.get("min_new_tokens", 200),
            temperature=generation_config.get("temperature", 0.5),
            do_sample=True,
            stop_sequences=["<|endoftext|>"],
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        # Diccionario para almacenar los resultados parciales por imagen
        final_results = {img_id: [] for img_id in ids}

        # Procesamiento para cada prompt
        for prompt in prompts:
            print(f"[LOG] Inicio del procesamiento para prompt: '{prompt}'")
            prompt_start_time = time.time()
            # Procesamos imágenes en lotes
            for start in range(0, len(images_list), batch_size):
                batch_start_time = time.time()
                batch_images = images_list[start:start + batch_size]
                batch_ids = ids[start:start + batch_size]
                print(f"[LOG] Procesando lote para prompt '{prompt}' de imágenes {start} a {start + len(batch_images)-1}")
                inputs_batch = self.process_batch(prompt, batch_images)
                inputs_batch = {k: v.to(self.model.device) for k, v in inputs_batch.items()}
                inputs_batch["images"] = inputs_batch["images"].to(torch.bfloat16)
                
                with torch.inference_mode():
                    outputs = self.model.generate_from_batch(
                        inputs_batch,
                        gen_config,
                        tokenizer=self.processor.tokenizer,
                    )

                input_len = inputs_batch["input_ids"].shape[1]
                generated_texts = self.processor.tokenizer.batch_decode(
                    outputs[:, input_len:], skip_special_tokens=True
                )
                for idx, text in enumerate(generated_texts):
                    try:
                        parsed = json.loads(text)
                        description = parsed.get("description", text)
                    except Exception:
                        description = text
                    final_results[batch_ids[idx]].append(description)
                    torch.cuda.empty_cache()
                batch_end_time = time.time()
                print(f"[LOG] Lote completado en {batch_end_time - batch_start_time:.2f} segundos.")
            prompt_end_time = time.time()
            print(f"[LOG] Procesamiento del prompt '{prompt}' completado en {prompt_end_time - prompt_start_time:.2f} segundos.")

        # Se combinan las inferencias de cada prompt para cada imagen
        combined_results = []
        for img_id, descriptions in final_results.items():
            combined_text = " | ".join(descriptions)
            combined_results.append({"id": img_id, "description": combined_text})
        
        global_end_time = time.time()
        print(f"[LOG] Tiempo total de procesamiento: {global_end_time - global_start_time:.2f} segundos.")
        return combined_results
