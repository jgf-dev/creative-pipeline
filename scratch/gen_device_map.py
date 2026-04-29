import json
import os

model_path = "src/hf/gemma-4b-pruned-sharded/model.safetensors.index.json"
with open(model_path, "r") as f:
    index = json.load(f)

weight_map = index["weight_map"]

# Tensors we want to offload to CPU (the massive ones)
cpu_tensors = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.embed_tokens_per_layer.weight",
    "lm_head.weight"
]

device_map = {}
for weight_name in weight_map.keys():
    # Remove the .weight suffix to map the module if possible, or keep it for the parameter
    module_name = weight_name
    if module_name.endswith(".weight"):
        module_name = module_name[:-7]
    
    # We need to decide if we map the module or the full parameter name
    # Accelerate usually prefers module names
    
    is_cpu = False
    for cpu_tensor in cpu_tensors:
        if weight_name == cpu_tensor:
            is_cpu = True
            break
    
    if is_cpu:
        device_map[module_name] = "cpu"
    else:
        # Check if we already have a parent module mapped
        parent_parts = module_name.split(".")
        has_parent_mapped = False
        for i in range(len(parent_parts)):
            parent = ".".join(parent_parts[:i])
            if parent in device_map:
                has_parent_mapped = True
                break
        
        if not has_parent_mapped:
            device_map[module_name] = 0

print(json.dumps(device_map, indent=2))
