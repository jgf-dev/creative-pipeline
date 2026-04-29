import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"Free: {free / 1024**3:.2f} GB")
    print(f"Total: {total / 1024**3:.2f} GB")
    print(f"Properties Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
