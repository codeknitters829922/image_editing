import os
import torch
import runpod
from diffusers import Flux2KleinPipeline # Specific pipeline for this model

# 1. SET CACHE TO VOLUME FIRST
os.environ["HF_HOME"] = "/runpod-volume/huggingface"
os.environ["HF_HUB_CACHE"] = "/runpod-volume/huggingface"
os.makedirs("/runpod-volume/huggingface", exist_ok=True)

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

print(f"Loading pipeline: {MODEL_ID}")

try:
    # Use torch_dtype (not dtype) 
    # Use Flux2KleinPipeline (Specific for Klein)
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16, 
        device_map="balanced"
    )
    
    # CRITICAL: For 24GB GPUs (4090/3090), use this to make it fast!
    # This requires 'pip install optimum-quanto' in requirements.txt
    from optimum.quanto import freeze, qfloat8, quantize
    quantize(pipe.transformer, weights=qfloat8)
    freeze(pipe.transformer)
    
    print("Pipeline loaded successfully with FP8 Quantization")
except Exception as e:
    print(f"Error: {e}")
    raise

def handler(event):
    # ... your existing handler code ...
    pass

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})