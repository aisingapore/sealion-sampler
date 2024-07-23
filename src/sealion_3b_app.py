from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()

# Get the API URL and model from environment variables
API_URL = os.getenv('API_URL')
API_MODEL = os.getenv('API_MODEL')

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
    model = data.get('model','local')
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

    if model == 'local':
        # Handle the request locally
        generated_text = local_gen_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_strings=stop_strings,
            purpose=purpose
            )
    elif model == 'online':
        # Handle the request by sending it to the online API
        generated_text = api_gen_text(prompt)

    return jsonify(generated_text)

def local_gen_text(prompt,max_tokens,temperature,stop_strings,purpose):
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
    
    gen_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove question if doing Question and Answer
    if purpose!="textGeneration":
        prompt_end_posn = len(prompt)
        gen_text = gen_text[prompt_end_posn:].strip()
        if purpose=="questionAnswer":
            # Remove stop_strings from gen_text
            for stop_string in stop_strings:
                gen_text = gen_text.replace(stop_string, '')

    return gen_text

def api_gen_text(prompt):
    # Read API_URL and API_MODEL from environment variables
    payload = {
        "model": API_MODEL,
        "prompt": prompt
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        response_text = response.text
        lines = response_text.splitlines()

        full_response = ""

        for line in lines:
            json_line = json.loads(line)
            full_response += json_line.get("response", "")
            if json_line.get("done") and json_line.get("done_reason") == "stop":
                break
        return full_response
    else:
        return f"Error: {response.status_code}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
