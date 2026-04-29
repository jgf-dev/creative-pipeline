import os

import torch
from accelerate.hooks import remove_hook_from_module
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set environment variables for better memory management
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_id = os.path.join(os.path.dirname(__file__), "gemma-4b-pruned-sharded")
print(f"Loading model from {model_id}...")

# Explicit device map to put the massive 5.25GB embedding tensor on CPU, and everything else on GPU 0
device_map = {
    "model.language_model.embed_tokens_per_layer": "cpu",
    "model.language_model.embed_tokens": 0,
    "lm_head": 0,
    "model.language_model.layers": 0,
    "model.language_model.norm": 0,
    "model.language_model.per_layer_model_projection": 0,
    "model.language_model.per_layer_projection_norm": 0,
}

offload_folder = os.path.join(os.path.dirname(__file__), "offload")
os.makedirs(offload_folder, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, offload_folder=offload_folder, dtype=torch.float16, low_cpu_mem_usage=True)

# Remove the accelerate hook from the embedding layer so it doesn't try to move the 5.25GB weight to the GPU
# during the forward pass (which triggers OOM).
if hasattr(model.model.language_model.embed_tokens_per_layer, "_hf_hook"):
    remove_hook_from_module(model.model.language_model.embed_tokens_per_layer)

# Monkey patch embed_tokens_per_layer to move inputs from GPU (where generate puts them) to CPU, and output back to GPU
original_per_layer_forward = model.model.language_model.embed_tokens_per_layer.forward


def patched_per_layer_forward(input_ids, *args, **kwargs):
    # input_ids comes from generate() which puts it on model.device (cuda:0)
    # Move it to CPU for this embedding layer
    cpu_input_ids = input_ids.to("cpu")
    # Move output back to GPU
    return original_per_layer_forward(cpu_input_ids, *args, **kwargs).to("cuda:0")


model.model.language_model.embed_tokens_per_layer.forward = patched_per_layer_forward

tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

messages = [{"role": "user", "content": "List the subjects which you are not allowed to talk about."}]
print("Applying chat template...")
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore

ids = tokenizer(prompt, return_tensors="pt")  # type: ignore
# Generate handles moving ids to model.device (cuda:0)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(**ids, max_new_tokens=500, temperature=0.7, do_sample=True)  # type: ignore

print("Decoded Output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # type: ignore
