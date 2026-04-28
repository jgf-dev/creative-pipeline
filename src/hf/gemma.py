import os
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GemmaConfig, GemmaTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)


model_id = os.path.join(os.path.dirname(__file__), "gemma-4b-4bit")
print(f"Loading model from {model_id}...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

messages = [{"role": "user", "content": "Your prompt here"}]

if hasattr(tokenizer, "apply_chat_template"):
    # Using return_dict=True to ensure we can access ["input_ids"] safely
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, return_dict=True)
    ids = inputs["input_ids"].to(model.device)
    
    print("Generating...")
    outputs = model.generate(
        input_ids=ids, 
        max_new_tokens=500, 
        temperature=0.7, 
        top_p=0.9, 
        top_k=40, 
        repetition_penalty=1.1, 
        do_sample=True
    )
    print(tokenizer.decode(outputs[0][ids.shape[-1]:], skip_special_tokens=True))
else:
    print(f"Tokenizer {tokenizer.__class__.__name__} does not support chat templates.")
