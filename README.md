# NLG Web API
A Python Flask web API for NLG provided by [HuggingFace Transformers models](https://github.com/huggingface/transformers/).

## Usage
- `GET /generate?max_length=500&num_return_sequences=2&seed=The world is not enough.`

## Development
To run locally;
- `python3 -m venv .`
- `source ./bin/activate`
- `echo 'FLASK_ENV=development' > .env`
- `pip install -r requirements.txt`
- `./download-model.sh`
- `python application.py`

## Clean up
- `pip uninstall -r requirements.txt -y`

## Deploy
- `eb deploy`

### AWS Elastic Beanstalk
- `eb init`
- `eb create nlg-web-api-production`

## Configuration
- Set `MODEL_NAME` in `.env` to one of;
  - `distilgpt2`
  - `gpt2` (default)
  - `gpt2-medium`
  - `gpt2-large`
  - `gpt2-xl`

