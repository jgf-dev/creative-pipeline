from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("OBLITERATUS/gemma-4-E4B-it-OBLITERATED")
messages = [{"role": "user", "content": "Your prompt here"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, return_dict=False)
