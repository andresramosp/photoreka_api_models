import numpy as np
import torch
import torch.nn.functional as F
import requests
from PIL import Image, ImageOps
import io, base64, json, traceback, time
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

class EndpointHandler:
    def __init__(self, model_dir, default_float16=True):
        try:
            print("[INFO] Cargando modelo con trust_remote_code=True...")
            dtype = torch.bfloat16 if default_float16 else "auto"
            self.processor = AutoProcessor.from_pretrained(
                model_dir, trust_remote_code=True, torch_dtype="auto", device_map="auto"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto"
            )
        except Exception:
            logging.exception("Error en la inicialización del modelo")
            raise

    def process_batch(self, prompts_list, images_list, images_config=None):
        """
        Procesa un lote de prompts y sus correspondientes imágenes.
        """
        try:
            # Construimos el texto que va antes del prompt real
            batch_texts = [f"User: {p} Assistant:" for p in prompts_list]
            
            # Tokenizamos la lista de textos de forma automática, con padding y truncado
            tokenized = self.processor.tokenizer(
                batch_texts,
                padding='max_length',  # o 'longest' para padding dinámico
                truncation=True,
                max_length=1536,
                return_tensors='pt'
            )

            print(tokenized)

            outputs_list = []
            images_kwargs = {
                "max_crops": images_config.get("max_crops", 12) if images_config else 12,
                "overlap_margins": images_config.get("overlap_margins", [4, 4]) if images_config else [4, 4],
                "base_image_input_size": [336, 336],
                "image_token_length_w": 12,
                "image_token_length_h": 12,
                "image_patch_size": 14,
                "image_padding_mask": True,
            }

            # Procesamos cada imagen y su correspondiente prompt
            for i in range(len(batch_texts)):
                try:
                    # Extraemos la secuencia tokenizada correspondiente y la convertimos a lista
                    tokens = tokenized["input_ids"][i].tolist()
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
                except Exception:
                    logging.exception("Error procesando la imagen número %d", i)
                    raise

            # Combinamos las salidas en formato batch usando el token de padding del tokenizer
            pad_token_id = self.processor.tokenizer.pad_token_id
            batch_outputs = {}
            for key in outputs_list[0].keys():
                try:
                    tensors = [torch.from_numpy(out[key]) for out in outputs_list]
                    batch_outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        tensors, batch_first=True, padding_value=pad_token_id
                    )
                except Exception:
                    logging.exception("Error al agrupar la key '%s' en outputs_list", key)
                    raise

            # Generamos la máscara de atención a partir del token de padding
            batch_outputs["attention_mask"] = (batch_outputs["input_ids"] != pad_token_id).long()

            # Añadimos el token BOS al inicio de la secuencia
            bos = self.processor.tokenizer.bos_token_id or self.processor.tokenizer.eos_token_id
            batch_outputs["input_ids"] = F.pad(batch_outputs["input_ids"], (1, 0), value=bos)

            # Ajustamos la posición de image_input_idx si existe
            if "image_input_idx" in batch_outputs:
                image_input_idx = batch_outputs["image_input_idx"]
                batch_outputs["image_input_idx"] = torch.where(
                    image_input_idx < 0, image_input_idx, image_input_idx + 1
                )

            return batch_outputs
        except Exception:
            logging.exception("Error en process_batch")
            raise

    def __call__(self, data=None):
        global_start_time = time.time()
        try:
            print("[INFO] Iniciando procesamiento por lotes...")
            if not data:
                return {"error": "El cuerpo de la petición está vacío."}
            if "inputs" not in data:
                return {"error": "Se requiere un campo 'inputs' en la petición JSON."}
        except Exception:
            logging.exception("Error en la verificación inicial de datos")
            return {"error": "Error en la verificación de datos."}

        try:
            inputs_data = data["inputs"]
        except Exception:
            logging.exception("Error al acceder al campo 'inputs'")
            return {"error": "Error al acceder al campo 'inputs'."}

        # Cargar imágenes y sus IDs
        images_list = []
        ids = []
        try:
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
                        logging.exception("Error loading image with id %s", item.get("id", "desconocido"))
                        continue
            else:
                return {"error": "Se requiere una lista de imágenes en 'inputs.images'."}
        except Exception:
            logging.exception("Error procesando la lista de imágenes")
            return {"error": "Error al procesar la lista de imágenes."}

        # Obtener prompts globales y específicos
        try:
            global_prompts_list = inputs_data.get("prompts", [])
            prompts_per_image = inputs_data.get("prompts_per_image", [])
            # Diccionario: { image_id (str): [ {id, text}, {id, text}, ... ] }
            specific_prompts = {}
            for item in prompts_per_image:
                if "id" in item and "prompts" in item:
                    specific_prompts.setdefault(str(item["id"]), []).extend(item["prompts"])
        except Exception:
            logging.exception("Error al construir el mapeo de prompts por imagen")
            return {"error": "Error al construir el mapeo de prompts por imagen."}

        # Preparamos la salida final
        final_results = {img_id: [] for img_id in ids}

        # Configuración de generación
        try:
            batch_size = inputs_data.get("batch_size", len(images_list))
            generation_config = inputs_data.get("generation_config", {})
            use_bfloat16 = generation_config.get("float16", True)
            gen_config = GenerationConfig(
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                max_new_tokens=generation_config.get("max_new_tokens", 200),
                temperature=generation_config.get("temperature", 0.2),
                top_p=generation_config.get("top_p", 1),
                top_k=generation_config.get("top_k", 50),
                length_penalty=generation_config.get("length_penalty", 1),
                stop_strings="<|endoftext|>",
                do_sample=True
            )
        except Exception:
            logging.exception("Error al configurar la generación")
            return {"error": "Error al configurar la generación."}

        # 1) Aplanamos todos los pares (imagen, image_id, prompt_id, prompt_text)
        flattened = []
        try:
            for img, img_id in zip(images_list, ids):
                image_prompts = specific_prompts.get(str(img_id), global_prompts_list)
                for p in image_prompts:
                    flattened.append((img, img_id, p["id"], p["text"]))
        except Exception:
            logging.exception("Error aplanando prompts por imagen")
            return {"error": "Error aplanando prompts por imagen."}

        # 2) Procesamos en lotes la lista aplanada
        print(f"[Info] Inicio de proceso por lotes sobre diccionario: {flattened}.")
        try:
            for start in range(0, len(flattened), batch_size):
                chunk = flattened[start:start+batch_size]
                batch_imgs = [x[0] for x in chunk]
                batch_img_ids = [x[1] for x in chunk]
                batch_prompt_ids = [x[2] for x in chunk]
                batch_prompt_texts = [x[3] for x in chunk]

                inputs_batch = self.process_batch(batch_prompt_texts, batch_imgs, generation_config)
                inputs_batch = {k: v.to(self.model.device) for k, v in inputs_batch.items()}

                if use_bfloat16 and "images" in inputs_batch:
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
                    final_results[batch_img_ids[idx]].append({
                        "id_prompt": batch_prompt_ids[idx],
                        "description": text
                    })
                
                torch.cuda.empty_cache()

        except Exception:
            logging.exception("Error al procesar los lotes aplanados")
            return {"error": "Error al procesar los lotes aplanados."}

        # 4) Preparamos la salida final
        try:
            combined_results = [
                {"id": img_id, "descriptions": descs}
                for img_id, descs in final_results.items()
            ]
            print(f"[DEBUG] Tiempo total de procesamiento: {time.time() - global_start_time:.2f} segundos.")
            return combined_results
        except Exception:
            logging.exception("Error al combinar los resultados finales")
            return {"error": "Error al combinar los resultados finales."}
