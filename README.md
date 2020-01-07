# GPT-2 Web API
A web API for GPT-2.

## Development
To run locally;
- `pip install -r requirements.txt`
- `export FLASK_APP=main.py`
- `export FLASK_ENV=development`
- `flask run`

## Clean up
- `pip uninstall -r requirements.txt -y`

## Deployment
This app works well with Heroku.
- Install the Heroku CLI and set it up
- `heroku create`
- `git push heroku master`
