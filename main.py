from flask import Flask, jsonify, request

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

app = Flask(__name__)


@app.route('/generate')
def index():
    seed = request.args.get('seed', 'The answer to life')
    max_length = request.args.get('max_length', 20, int)
    num_return_sequences = request.args.get('num_return_sequences', 10, int)

    input_ids = torch.tensor(tokenizer.encode(seed)).unsqueeze(0)
    output = model.generate(input_ids=input_ids,
                            max_length=max_length,
                            num_return_sequences=num_return_sequences,
                            do_sample=True)
    sequences = []
    for j in range(len(output)):
        for i in range(len(output[j])):
            sequences.append(tokenizer.decode(
                output[j][i].tolist(), skip_special_tokens=True))
    return jsonify(sequences)


if __name__ == '__main__':
    app.run()
