"""
sealion_3b_app.py
-----------------

Script for running Flask app to send POST requests to local model or server

Routes:
    - @app.route("/"): Renders the home page.
    - @app.route("/generate", methods=["POST"]): Receives a POST request and returns generated text.

Functions:
    - home: Renders the home page for the Flask app.
    - generate_text: Receives a POST request, processes the input, and returns generated text.
    - local_gen_text: Passes inputs from the Flask app to the locally run model,
        returning the predicted text output.
    - oll_gen_text: Passes inputs from the Flask app to the model running on ollama,
        returning the predicted text output.
    - tgi_gen_text: Passes inputs from the Flask app to the model running on TGI server,
        returning the predicted text output.


"""
import os

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables from .env file
load_dotenv()

# Get the API URL and model from environment variables
OLL_API_URL = os.getenv("OLL_API_URL")
OLL_API_MODEL = os.getenv("OLL_API_MODEL")
# For TGI inference server with multiple models
TGI_API_URL = os.getenv("TGI_API_URL")
TGI_API_KEY = os.getenv("TGI_API_KEY", default=None)
TGI_SEALION = os.getenv("TGI_SEALION")
TGI_LLAMA = os.getenv("TGI_LLAMA", default=None)

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("aisingapore/sea-lion-3b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("aisingapore/sea-lion-3b", trust_remote_code=True)


@app.route("/")
def home():
    """Generates the home page for the Flask app.

    Returns:
        str:  The rendered HTML of the home page.
    """
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_text():
    """Receives a POST request from the Flask app frontend.

    Inputs from the form are received, the prompt is enhanced according to the
    intended purpose, and then sent to the model of choice.

    Returns:
        Response: A JSON response containing the generated text.
    """
    data = request.json
    model = data.get("model", "local")
    prompt = data.get("prompt", "")
    purpose = data.get("purpose", "textGeneration")
    language = data.get("language", "English")
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 40)
    stop_strings = None

    # Update prompt if used for Question and Answer
    if purpose == "questionAnswer":
        prompt = f"""Question: {prompt}
        
        Answer:"""
        stop_strings = ["Question:"]

    elif purpose == "translation":
        prompt = f"""'{prompt}'

        In {language}, this translates to: """

    if model == "local":
        # Handle the request locally
        generated_text = local_gen_text(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop_strings=stop_strings, purpose=purpose
        )
    elif model == "ollama":
        # Handle the request by sending it to the online API
        generated_text = oll_gen_text(prompt, temperature, max_tokens)
    elif "tgi" in model:
        generated_text = tgi_gen_text(prompt, model, temperature, max_tokens)

    return jsonify(generated_text)


def local_gen_text(prompt, max_tokens, temperature, stop_strings, purpose):
    """Passes inputs from the Flask app to the locally run model, returning the predicted text output.

    Args:
        prompt (str): Input prompt from Flask app, after processing
        max_tokens (int): The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        temperature (float): The value used to modulate the next token probabilities.
        stop_strings (str or List[str]): A string or a list of strings that should terminate generation
            if the model outputs them.
        purpose (str): The role the model is playing (E.g. textGeneration,questionAnswer)

    Returns:
        gen_text (str): processed output response from model
    """
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
        tokenizer=tokenizer,
    )

    gen_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove question if doing Question and Answer
    if purpose != "textGeneration":
        prompt_end_posn = len(prompt)
        gen_text = gen_text[prompt_end_posn:].strip()
        if purpose == "questionAnswer":
            # Remove stop_strings from gen_text
            for stop_string in stop_strings:
                gen_text = gen_text.replace(stop_string, "")

    return gen_text


def oll_gen_text(prompt, temperature, max_tokens):
    """Passes inputs from the Flask app to the model running on ollama, returning the predicted text output.

    Args:
        prompt (str): Input prompt from Flask app, after processing
        temperature (float): The value used to modulate the next token probabilities.
        max_tokens (int): The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.

    Returns:
        str: processed output response from model for successful response, or returns error message if otherwise.
    """
    print("Using ollama model")
    # Read API_URL and API_MODEL from environment variables
    payload = {
        "model": OLL_API_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "max_new_tokens": max_tokens},
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(OLL_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        print(response.text)
        return response.json().get("response")
    else:
        return f"Error: {response.status_code}"


def tgi_gen_text(prompt, model, temperature, max_tokens):
    """Passes inputs from the Flask app to the model running on TGI server, returning the predicted text output.

    Args:
        prompt (str): Input prompt from Flask app, after processing
        model (str): Choice of model on server (if running multiple models)
        temperature (float): The value used to modulate the next token probabilities.
        max_tokens (int): The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.

    Returns:
        str: processed output response from model for successful response, or returns error message if otherwise.
    """

    print("Using tgi inference server")
    model_list = {"tgi-sealion": TGI_SEALION, "tgi-llama": TGI_LLAMA}
    model_choice = model_list.get(model)
    print("Using model: ", model_choice)
    headers = {"Content-Type": "application/json", "x-api-key": TGI_API_KEY}

    payload = {
        "frequency_penalty": 1,
        "logit_bias": [0],
        "logprobs": False,
        "max_tokens": max_tokens,
        "messages": [{"content": prompt, "role": "user"}],
        "model": model_choice,
        "n": 2,
        "presence_penalty": 0.1,
        "seed": 42,
        "stop": None,
        "stream": False,
        "temperature": temperature,
        "tool_prompt": '"You will be presented with a JSON schema representing a set of tools.\n \
        If the user request lacks of sufficient information to make a precise tool selection: \
        Do not invent any tool\'s properties, instead notify with an error message.\n\nJSON Schema:\n"',
        "tools": None,
        "top_logprobs": 5,
        "top_p": 0.95,
    }
    response = requests.post(TGI_API_URL, json=payload, headers=headers)
    print("\nresponse: ", response)
    print(response.text)

    if response.status_code == 200:
        response_str = response.json().get("choices")[0]["message"]["content"]
        return response_str
    else:
        return f"Error: {response.status_code}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
