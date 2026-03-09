import runpod
import torch
import base64
import io
import os
import requests
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from PIL import Image

HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
LORA_ID  = "Alissonerdx/BFS-Best-Face-Swap"

print(f"[1] Starting worker...")
print(f"[2] HF_TOKEN present: {bool(HF_TOKEN)}")
print(f"[3] CUDA available: {torch.cuda.is_available()}")
print(f"[4] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"[5] Loading pipeline: {MODEL_ID}")

try:
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    ).to("cuda")
    print("[6] Base pipeline loaded successfully")
except Exception as e:
    print(f"[FATAL] Pipeline load failed: {e}")
    raise

try:
    pipe.load_lora_weights(LORA_ID, token=HF_TOKEN)
    print("[7] LoRA loaded successfully")
except Exception as e:
    print(f"[FATAL] LoRA load failed: {e}")
    raise

print("[8] Pipeline ready. Starting serverless handler.")

def handler(event):
    print(f"[handler] Received event: {event}")
    job_input = event["input"]
    
    prompt      = job_input.get("prompt", "Turn this cat into a dog")
    image_url   = job_input.get("image_url")
    image_b64   = job_input.get("image_b64")
    num_steps   = job_input.get("num_inference_steps", 4)
    guidance    = job_input.get("guidance_scale", 3.5)

    print(f"[handler] prompt={prompt}, steps={num_steps}, guidance={guidance}")

    if image_url:
        print(f"[handler] Loading image from URL: {image_url}")
        input_image = load_image(image_url)
    elif image_b64:
        print("[handler] Decoding base64 image")
        img_bytes   = base64.b64decode(image_b64)
        input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        return {"error": "Provide image_url or image_b64"}

    print("[handler] Running inference...")
    result = pipe(
        image=input_image,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
    ).images[0]

    print("[handler] Inference complete. Encoding output...")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    print("[handler] Done.")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})