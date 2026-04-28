import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

model_id = "OBLITERATUS/gemma-4-E4B-it-OBLITERATED"
save_path = os.path.join(os.path.dirname(__file__), "gemma-4b-4bit")

print(f"Loading model '{model_id}' on CPU for quantization... this will take a few minutes.")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load on CPU explicitly to avoid GPU VRAM peak
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map={"": "cpu"},
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Saving quantized model to {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Done! You can now load the model from this local folder.")
