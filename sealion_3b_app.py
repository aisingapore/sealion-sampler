from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("aisingapore/sea-lion-3b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("aisingapore/sea-lion-3b", trust_remote_code=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', '')
    purpose = data.get('purpose','textGeneration')
    language = data.get('language','English')
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 40)
    stop_strings= None

    # Update prompt if used for Question and Answer
    if purpose=="questionAnswer":
        prompt=f"""Question: {prompt}
        
        Answer:"""
        stop_strings = ['Question:']

    elif purpose=="translation":
        prompt=f"""'{prompt}'

        In {language}, this translates to: """

    # Tokenize input prompt
    tokens = tokenizer(text=prompt, return_tensors="pt")
    
    # Generate text
    output = model.generate(
        tokens["input_ids"],
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        stop_strings=stop_strings,
        eos_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove question if doing Question and Answer
    if purpose!="textGeneration":
        prompt_end_posn = len(prompt)
        generated_text = generated_text[prompt_end_posn:].strip()
        if purpose=="questionAnswer":
            # Remove stop_strings from generated_text
            for stop_string in stop_strings:
                generated_text = generated_text.replace(stop_string, '')

    return jsonify(generated_text)

if __name__ == '__main__':
    app.run(debug=True)
