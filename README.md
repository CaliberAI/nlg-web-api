# NLG Web API
A Python Flask web API for NLG provided by [HuggingFace Transformers models](https://github.com/huggingface/transformers/).

## Usage
- `GET /generate?max_length=500&num_return_sequences=2&seed=The world is not enough.`

## Development
To run locally;
- `python3 -m venv .`
- `echo 'FLASK_ENV=development' > .env`
- `pip install -r requirements.txt`
- `python setup.py`
- `flask run`

## Clean up
- `pip uninstall -r requirements.txt -y`
