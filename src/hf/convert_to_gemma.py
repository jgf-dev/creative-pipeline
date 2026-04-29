import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

source_path = os.path.join(os.path.dirname(__file__), "gemma-4b-4bit")
target_path = os.path.join(os.path.dirname(__file__), "gemma-4b-pruned-sharded")

# Clean up target directory to save space
if os.path.exists(target_path):
    shutil.rmtree(target_path)
os.makedirs(target_path, exist_ok=True)

print(f"Reading config from {source_path}...")
with open(os.path.join(source_path, "config.json"), "r") as f:
    config = json.load(f)

# Keep it as gemma4 but remove multimodal configs
new_config = config.copy()
new_config["audio_config"] = None
new_config["vision_config"] = None
new_config["architectures"] = ["Gemma4ForConditionalGeneration"]

st_path = os.path.join(source_path, "model.safetensors")
print(f"Opening {st_path} for streaming...")

shard_size_limit = 500 * 1024 * 1024  # 500MB
current_shard = {}
current_shard_size = 0
shard_idx = 1
weight_map = {}

total_size_processed = 0

# Use safe_open to avoid loading everything into RAM
with safe_open(st_path, framework="pt", device="cpu") as f:
    keys = f.keys()
    # Filter keys to exclude multimodal parts
    filtered_keys = [k for k in keys if not any(x in k for x in ["audio_tower", "vision_tower", "embed_audio", "embed_vision"])]

    print(f"Processing {len(filtered_keys)} tensors out of {len(keys)}...")

    for key in tqdm(sorted(filtered_keys)):
        tensor = f.get_tensor(key)
        size = tensor.numel() * tensor.element_size()
        total_size_processed += size

        # If adding this tensor exceeds the limit, save the current shard first
        # UNLESS the current shard is empty (case for tensors > 500MB)
        if current_shard_size + size > shard_size_limit and current_shard:
            shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            save_file(current_shard, os.path.join(target_path, shard_name))
            for k in current_shard.keys():
                weight_map[k] = shard_name
            shard_idx += 1
            current_shard = {}
            current_shard_size = 0

        current_shard[key] = tensor
        current_shard_size += size

        # If a single tensor is huge, we save it immediately in its own shard
        if current_shard_size > shard_size_limit:
            shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            save_file(current_shard, os.path.join(target_path, shard_name))
            for k in current_shard.keys():
                weight_map[k] = shard_name
            shard_idx += 1
            current_shard = {}
            current_shard_size = 0

# Save any remaining tensors
if current_shard:
    shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
    save_file(current_shard, os.path.join(target_path, shard_name))
    for k in current_shard.keys():
        weight_map[k] = shard_name
else:
    shard_idx -= 1  # Adjust if the last tensor was saved in the loop

total_shards = shard_idx
print(f"Finalizing {total_shards} shards...")
for i in range(1, total_shards + 1):
    old_name = f"model-{i:05d}-of-XXXXX.safetensors"
    new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
    old_path = os.path.join(target_path, old_name)
    new_path = os.path.join(target_path, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)

    # Update weight map with final names
    for k, v in weight_map.items():
        if v == old_name:
            weight_map[k] = new_name

# Save index and updated config
index = {"metadata": {"total_size": total_size_processed}, "weight_map": weight_map}

with open(os.path.join(target_path, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

with open(os.path.join(target_path, "config.json"), "w") as f:
    json.dump(new_config, f, indent=2)

# Copy tokenizer files
print("Copying tokenizer files...")
for f_name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
    src_f = os.path.join(source_path, f_name)
    if os.path.exists(src_f):
        shutil.copy(src_f, target_path)

print(f"Successfully converted and sharded model to {target_path}")
