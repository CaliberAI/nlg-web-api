from flask import Blueprint, jsonify, request
from flask_restplus import inputs

from nltk.tokenize import sent_tokenize
import nltk
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

generate_blueprint = Blueprint('generate', __name__)

def get_sentences(input, split_sentences, tokenizer):
    text = tokenizer.decode(input.tolist(), skip_special_tokens=True)
    if split_sentences:
        text = sent_tokenize(text)
        return text
    else:
        return [text]

@generate_blueprint.route('/generate')
def generate():
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
    params['model'] = request.args.get('model', 'gpt2')
    meta['model_params'] = params

    if params['num_return_sequences'] > 1 and not params['sample']:
      meta['advice'] = 'num_return_sequences > 1 works best with sample=true'

    sequences = []

    model = GPT2LMHeadModel.from_pretrained(params['model'])
    tokenizer = GPT2Tokenizer.from_pretrained(params['model'])

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

    for j in range(len(output)):
        if params['num_return_sequences'] > 1:
            for i in range(len(output[j])):
                sequences.append(get_sentences(
                    output[j][i], params['split_sentences'], tokenizer))
        else:
            sequences.append(get_sentences(
                output[j], params['split_sentences'], tokenizer))

    return jsonify({'sequences': sequences,
                    'meta': meta})
