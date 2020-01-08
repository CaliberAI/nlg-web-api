from flask import Flask, jsonify, request

from nltk.tokenize import sent_tokenize
import nltk
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
nltk.download('punkt')

app = Flask(__name__)


@app.route('/generate')
def index():
    seed = request.args.get('seed', 'The answer to life')
    max_length = request.args.get('max_length', 20, int)
    num_return_sequences = request.args.get('num_return_sequences', 10, int)
    split_sentences = request.args.get('split_sentences', True, bool)

    input_ids = torch.tensor(tokenizer.encode(seed)).unsqueeze(0)
    output = model.generate(input_ids=input_ids,
                            max_length=max_length,
                            num_return_sequences=num_return_sequences,
                            do_sample=True)
    sequences = []
    for j in range(len(output)):
        for i in range(len(output[j])):
            text = tokenizer.decode(
                output[j][i].tolist(), skip_special_tokens=True)
            if split_sentences:
                text = sent_tokenize(text)[0]
            sequences.append(text)
    return jsonify(sequences)


if __name__ == '__main__':
    app.run()
