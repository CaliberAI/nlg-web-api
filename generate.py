from flask import Blueprint, jsonify, request
from flask_restplus import inputs
from os import environ
from dotenv import load_dotenv

from nltk.tokenize import sent_tokenize
import nltk
import torch
import bcrypt
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#FIX: Shouldn't need to be in this module but gunicorn needs it
load_dotenv('.env')

#TODO: Replace with dynamic per-user token store
MASTER_API_KEY = environ.get('MASTER_API_KEY')

if not MASTER_API_KEY:
    raise Exception('MASTER_API_KEY not set')

generate_blueprint = Blueprint('generate', __name__)

model = None
tokenizer = None
model_name = environ.get('MODEL_NAME') or 'gpt2'

def first_load():
    global model, tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def hash_func(key):
    return bcrypt.hashpw(key.encode("UTF-8"), bcrypt.gensalt())

def get_sentences(input, split_sentences, tokenizer):
    text = tokenizer.decode(input, skip_special_tokens=True)
    return sent_tokenize(text) if split_sentences else [text]

@generate_blueprint.route('/generate')
def generate():
    #TODO: Add .yaml document for X-Api-Key; otherwise, need to use cURL, or need to switch to second auth variable for directly putting password in URL
    auth = request.headers.get("X-Api-Key")
    #auth = request.args.get('password')

    if not auth or not bcrypt.checkpw(auth.encode("UTF-8"), hash_func(MASTER_API_KEY)):
        return "Not authenticated to use this app. Please provide a password."

    global model, tokenizer

    meta = {}
    params = {}
    params['seed'] = request.args.get('seed', '')
    params['max_length'] = request.args.get('max_length', 250, int)
    params['num_return_sequences'] = request.args.get(
        'num_return_sequences', 1, int)
    params['split_sentences'] = request.args.get(
        'split_sentences', True, inputs.boolean)
    params['sample'] = request.args.get('sample', True, inputs.boolean)
    params['repetition_penalty'] = request.args.get(
        'repetition_penalty', 1, float)
    params['num_beams'] = request.args.get('num_beams', 1, int)
    params['temperature'] = request.args.get('temperature', 1.0, float)
    params['top_k'] = request.args.get('top_k', 50, int)
    params['top_p'] = request.args.get('top_p', 1, float)
    params['length_penalty'] = request.args.get('length_penalty', 2, float)
    params['model'] = model_name
    meta['model_params'] = params

    if params['num_return_sequences'] > 1 and not params['sample']:
        meta['advice'] = 'num_return_sequences > 1 works best with sample=true'

    if not model:
        first_load()

    sequences = []

    if len(params['seed']) > 0:
        input_ids = torch.tensor(tokenizer.encode(params['seed'])).unsqueeze(0)
    else:
        input_ids = None

    output = model.generate(input_ids=input_ids,
                            max_length=params['max_length'],
                            num_return_sequences=params['num_return_sequences'],
                            do_sample=params['sample'],
                            repetition_penalty=params['repetition_penalty'],
                            num_beams=params['num_beams'],
                            top_k=params['top_k'],
                            temperature=params['temperature'],
                            top_p=params['top_p'],
                            length_penalty=params['length_penalty'])

    if params['num_return_sequences'] > 1:
        for i in range(len(output[0])):
            sequences.append(get_sentences(
                output[0][i], params['split_sentences'], tokenizer))
    else:
        sequences.append(get_sentences(
            output[0], params['split_sentences'], tokenizer))

    return jsonify({'sequences': sequences,
                    'meta': meta})

   
