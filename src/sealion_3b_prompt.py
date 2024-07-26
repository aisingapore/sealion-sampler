"""
This script uses the Sea Lion 3B model from AISingapore to generate text based
on an input prompt. It demonstrates loading a pre-trained model and tokenizer,
encoding input text into tokens, generating new tokens, and decoding the output.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("aisingapore/sea-lion-3b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("aisingapore/sea-lion-3b", trust_remote_code=True)

tokens = tokenizer("The sea lion is a", return_tensors="pt")
output = model.generate(tokens["input_ids"], max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0], skip_special_tokens=True))
