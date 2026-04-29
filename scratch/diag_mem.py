import torch
from accelerate.utils import get_max_memory
from transformers import AutoConfig

config = AutoConfig.from_pretrained("src/hf/gemma-4b-pruned-sharded")
max_memory = get_max_memory()
print(f"Accelerate max_memory: {max_memory}")

# Check torch cuda memory
if torch.cuda.is_available():
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"Free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
