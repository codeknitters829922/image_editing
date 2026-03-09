import os

# DO THIS FIRST - Before importing diffusers or transformers
os.environ["HF_HOME"] = "/runpod-volume/huggingface"
os.environ["HF_HUB_CACHE"] = "/runpod-volume/huggingface"

# Ensure the directory exists
os.makedirs("/runpod-volume/huggingface", exist_ok=True)
import runpod
import torch
import base64
import io
import os
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from PIL import Image

HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

print(f"[1] CUDA available: {torch.cuda.is_available()}")
print(f"[2] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"[3] Loading pipeline: {MODEL_ID}")

try:
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="balanced",
        token=HF_TOKEN,
    )
    print("[4] Pipeline loaded successfully")
except Exception as e:
    print(f"[FATAL] Pipeline load failed: {e}")
    raise

print("[5] Ready.")

def handler(event):
    job_input = event["input"]

    prompt    = job_input.get("prompt", "Turn this cat into a dog")
    image_url = job_input.get("image_url")
    image_b64 = job_input.get("image_b64")
    num_steps = job_input.get("num_inference_steps", 28)
    guidance  = job_input.get("guidance_scale", 3.5)

    print(f"[handler] prompt={prompt}, steps={num_steps}, guidance={guidance}")

    if image_url:
        print(f"[handler] Loading image from URL: {image_url}")
        input_image = load_image(image_url)
    elif image_b64:
        print("[handler] Decoding base64 image")
        input_image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    else:
        return {"error": "Provide image_url or image_b64"}

    print("[handler] Running inference....")
    result = pipe(
        image=input_image,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
    ).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    print("[handler] Done.")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})