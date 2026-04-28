import torch
import os
import json
import shutil
from safetensors.torch import load_file, save_file
from tqdm import tqdm

source_path = os.path.join(os.path.dirname(__file__), "gemma-4b-4bit")
target_path = os.path.join(os.path.dirname(__file__), "gemma-4b-pruned-sharded")
os.makedirs(target_path, exist_ok=True)

print(f"Reading config from {source_path}...")
with open(os.path.join(source_path, "config.json"), "r") as f:
    config = json.load(f)

# Keep it as gemma4 but remove multimodal configs
new_config = config.copy()
new_config["audio_config"] = None
new_config["vision_config"] = None
# Ensure it loads with the right class
new_config["architectures"] = ["Gemma4ForConditionalGeneration"]

# Load original weights
st_path = os.path.join(source_path, "model.safetensors")
print(f"Loading weights from {st_path}...")
state_dict = load_file(st_path)

print("Pruning keys...")
new_state_dict = {}
for key, value in state_dict.items():
    # Skip multimodal components
    if any(x in key for x in ["audio_tower", "vision_tower", "embed_audio", "embed_vision"]):
        continue
    new_state_dict[key] = value

print(f"Sharding model into 500MB chunks...")
shard_size_limit = 500 * 1024 * 1024 

current_shard = {}
current_shard_size = 0
shard_idx = 1
weight_map = {}

# Sort keys: put the largest tensors at the end or handle them specially
# Actually, sorting alphabetically is fine, but we need to ensure the big one doesn't block everything
sorted_keys = sorted(new_state_dict.keys())

for key in tqdm(sorted_keys):
    value = new_state_dict[key]
    size = value.numel() * value.element_size()
    
    if current_shard_size + size > shard_size_limit and current_shard:
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(current_shard, os.path.join(target_path, shard_name))
        for k in current_shard.keys():
            weight_map[k] = shard_name
        shard_idx += 1
        current_shard = {}
        current_shard_size = 0
    
    current_shard[key] = value
    current_shard_size += size

if current_shard:
    shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
    save_file(current_shard, os.path.join(target_path, shard_name))
    for k in current_shard.keys():
        weight_map[k] = shard_name

total_shards = shard_idx
print(f"Finalizing {total_shards} shards...")
for i in range(1, total_shards + 1):
    old_name = f"model-{i:05d}-of-XXXXX.safetensors"
    new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
    old_path = os.path.join(target_path, old_name)
    new_path = os.path.join(target_path, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
    
    # Update weight map
    for k, v in weight_map.items():
        if v == old_name:
            weight_map[k] = new_name

# Save index and updated config
index = {
    "metadata": {"total_size": sum(v.numel() * v.element_size() for v in new_state_dict.values())},
    "weight_map": weight_map
}
with open(os.path.join(target_path, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

with open(os.path.join(target_path, "config.json"), "w") as f:
    json.dump(new_config, f, indent=2)

# Copy tokenizer files
print("Copying tokenizer files...")
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
    src_f = os.path.join(source_path, f)
    if os.path.exists(src_f):
        shutil.copy(src_f, target_path)

print(f"Successfully converted and sharded model to {target_path}")
