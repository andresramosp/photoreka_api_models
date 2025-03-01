import numpy as np
import torch
import torch.nn.functional as F
import requests
from PIL import Image, ImageOps
import io, base64, json, traceback
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

    def process_batch(self, prompt, images_list, front_preprocessed=False):
        batch_texts = [f"User: {prompt} Assistant:" for _ in images_list]
        tokens_list = [self.processor.tokenizer.encode(" " + text, add_special_tokens=False) for text in batch_texts]
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
            if not front_preprocessed:
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
            else:
                image_arr = np.array(images_list[i])
                out = {
                    "input_ids": np.asarray(tokens).astype(np.int32),
                    "images": image_arr,
                    "image_input_idx": np.array([-1]),
                    "image_masks": np.ones(image_arr.shape[:2], dtype=np.int32),
                }
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
        print("[INFO] Iniciando procesamiento por lotes...")
        if not data:
            return {"error": "El cuerpo de la petición está vacío."}
        if "inputs" not in data:
            return {"error": "Se requiere un campo 'inputs' en la petición JSON."}

        inputs_data = data["inputs"]
        prompt = inputs_data.get("prompt", "Describe this image.")
        front_preprocessed = inputs_data.get("front_preprocessed", False)
        batch_size = inputs_data.get("batch_size", len(inputs_data.get("images", [])))
        print(f"[DEBUG] Prompt: {prompt} | front_preprocessed: {front_preprocessed} | batch_size: {batch_size}")

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
                except Exception as e:
                    traceback.print_exc()
                    continue
        else:
            return {"error": "Se requiere una lista de imágenes en 'inputs.images'."}

        if len(images_list) == 0:
            try:
                fallback = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)
                images_list.append(fallback)
                ids.append("fallback")
            except Exception as e:
                traceback.print_exc()
                return {"error": "No se pudo cargar ninguna imagen."}

        results = []
        # Procesamiento en lotes
        for start in range(0, len(images_list), batch_size):
            batch_images = images_list[start:start+batch_size]
            batch_ids = ids[start:start+batch_size]
            inputs_batch = self.process_batch(prompt, batch_images, front_preprocessed)
            inputs_batch = {k: v.to(self.model.device) for k, v in inputs_batch.items()}
            inputs_batch["images"] = inputs_batch["images"].to(torch.bfloat16)
            
            generation_config = data.get("generation_config", {})
            gen_config = GenerationConfig(
                max_new_tokens=generation_config.get("max_new_tokens", 500),
                min_new_tokens=generation_config.get("min_new_tokens", 200),
                temperature=generation_config.get("temperature", 0.5),
                do_sample=True,
                stop_sequences=["<|endoftext|>"],
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

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
                except Exception:
                    parsed = {"description": text, "id": batch_ids[idx]}
                else:
                    parsed["id"] = batch_ids[idx]
                results.append(parsed)
                torch.cuda.empty_cache()
        return results
