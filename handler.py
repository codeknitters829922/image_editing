import runpod
import torch
import base64
import io
import os
import requests
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from PIL import Image

# ── Load model once at worker startup (not inside handler) ──────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
LORA_ID  = "Alissonerdx/BFS-Best-Face-Swap"

print("Loading pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
).to("cuda")

pipe.load_lora_weights(LORA_ID, token=HF_TOKEN)
print("Pipeline ready.")

# ── Handler ─────────────────────────────────────────────────────────────────
def handler(event):
    job_input = event["input"]
    
    prompt      = job_input.get("prompt", "Turn this cat into a dog")
    image_url   = job_input.get("image_url")
    image_b64   = job_input.get("image_b64")
    num_steps   = job_input.get("num_inference_steps", 4)  # klein is distilled, 4 steps default
    guidance    = job_input.get("guidance_scale", 3.5)

    # Accept image via URL or base64
    if image_url:
        input_image = load_image(image_url)
    elif image_b64:
        img_bytes   = base64.b64decode(image_b64)
        input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        return {"error": "Provide image_url or image_b64"}

    result = pipe(
        image=input_image,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
    ).images[0]

    # Return as base64
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})