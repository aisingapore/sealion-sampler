# Sea-Lion Base LLM Scripts

## Introduction
This repository contains basic scripts for running the Sea-Lion base LLM locally. It has been tested with the Sea-Lion-3b base LLM. The scripts available facilitate running prompts in the terminal, as well as running text completion, question and answer, and translation via a Flask app.

## Specifications
The scripts provided have been tested in the following environments:
### MacBook Pro
- Processor: 2.3GHz Quad-Core Intel Core i7
- Memory: 32GB
- OS: MacOS Sonoma version 14.5
- Chip Architecture: x86-64

### Debian GNU/Linux 11 (Bullseye) VM
- Memory: 16GB
- OS: Debian GNU/Linux 11 (Bullseye)
- Chip Architecture: x86-64

## Quick Startup

### Conda Installation
1. Download Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).
2. Follow the installation instructions for your operating system.

### Creation of Virtual Environment
Open your terminal and execute the following commands:

```sh
# Create a new conda environment
conda create -n sealion_env python=3.9

# Activate the environment
conda activate sealion_env
```

### Installation of Required Packages
Ensure you have requirements.txt in your project directory. Run:

```sh
# Install required packages
pip install -r requirements.txt
```
### Test prompt via script
To test the model with a simple prompt, run the following Python script via terminal:

```sh
python src.sealion_3b_prompt.py
```
It will prompt the model with the string `The sea lion is a` and should return a continuation of this sentence.


### Running the Flask App
To start up the Flask app:

```sh
flask -A src.sealion_3b_app run
```
OR
```sh
python src.sealion_3b_app.py
```
Once the terminal returns the following, you should proceed to access the app via [port 5000](http://127.0.0.1:5000):
```sh
 * Serving Flask app 'src.sealion_3b_app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
127.0.0.1 - - [DD/MMM/YYYY HH:MM:SS] "GET / HTTP/1.1" 200 -
```
Currently there are three functions available, which prompt the base LLM differently:
- Text Generation: No additional template, LLM proceeds with text generation, continuing from where the input prompt ends
- Question and Answer: Prompt is input to a `Question: {prompt} Answer:` template
- Translation: An additional `Language` option will be provided (Default: `English`). Prompt is input to a `'{prompt}' In {language}, this translates to:` template. To date, performance is still sub-par and pending further experimentation. <br>


There are two parameters provided:
- Temperature (Default: `0.7`, Range:`0.0-1.0`)
- Max New Tokens (Default: `40`)

### For Further Exploration
To run the script for prompting via LangChain:

```sh
python src.sealion_3b_langchain.py
```
This script is currently a work in progress, with different parts commented out to test different ways of prompting the SEA-LION base model. Feel free to experiment with the script!