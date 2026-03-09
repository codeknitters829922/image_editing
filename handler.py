import io
import base64
import torch
import runpod
from PIL import Image
from diffusers.utils import load_image
import os
hf_token = os.environ.get("HF_TOKEN")  # ✅ GOOD: Safe to commit
from diffusers import Flux2KleinPipeline, FluxTransformer2DModel

# Load base structure (text encoder, vae, etc.)
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B",
    torch_dtype=torch.bfloat16,
)

# Load only the FP8 transformer weights
transformer_fp8 = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors",
    torch_dtype=torch.bfloat16,
)

pipe.transformer = transformer_fp8.to("cuda")  # or device_map

pipe.enable_model_cpu_offload()

# Optional helper to handle both URLs and base64 string images
def parse_input_image(image_str):
    if image_str.startswith("http://") or image_str.startswith("https://"):
        return load_image(image_str)
    else:
        # Assume base64
        if "," in image_str:
            image_str = image_str.split(",")[1]
        image_data = base64.b64decode(image_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")


# ----------------------------------------------------------------------------
# 2. Define the RunPod Handler
# ----------------------------------------------------------------------------
def handler(job):
    try:
        job_input = job.get("input", {})
        
        # Extract prompt and image from the request, with your defaults
        prompt = job_input.get("prompt", "Turn this cat into a dog")
        image_input = job_input.get(
            "image", 
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        )
        
        # Load the input image
        input_image = parse_input_image(image_input)

        # Generate the image
        # You can also add dynamically passed variables here like num_inference_steps
        result_image = pipe(
            image=input_image, 
            prompt=prompt
        ).images[0]

        # Convert the PIL image to base64 so RunPod can return it as JSON
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "image_base64": img_str
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# ----------------------------------------------------------------------------
# 3. Start the Serverless Worker
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
