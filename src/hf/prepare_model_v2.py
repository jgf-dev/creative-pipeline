import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_id = "OBLITERATUS/gemma-4-E4B-it-OBLITERATED"
save_path = os.path.join(os.path.dirname(__file__), "gemma-4b-pruned-sharded")

print(f"Loading original model '{model_id}' on CPU... this will take a few minutes.")
# Load in half precision on CPU to keep memory manageable
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map={"": "cpu"}, 
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Prune multimodal parts to reduce size and VRAM requirement
print("Pruning multimodal components (vision and audio)...")
# Note: We use del to remove the modules from the model object
if hasattr(model.model, "vision_tower"): 
    print("Removing vision_tower")
    del model.model.vision_tower
if hasattr(model.model, "audio_tower"): 
    print("Removing audio_tower")
    del model.model.audio_tower
if hasattr(model.model, "embed_vision"): 
    print("Removing embed_vision")
    del model.model.embed_vision
if hasattr(model.model, "embed_audio"): 
    print("Removing embed_audio")
    del model.model.embed_audio

# Update config to reflect removal (optional but helpful)
model.config.vision_config = None
model.config.audio_config = None

tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Saving pruned and sharded model to {save_path} in 500MB chunks...")
model.save_pretrained(save_path, max_shard_size="500MB")
tokenizer.save_pretrained(save_path)
print("Done! You now have a sharded, text-focused model.")
