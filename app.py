from flask import Flask
from flask_cors import CORS
from generate import generate_blueprint
from index import index_blueprint


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(index_blueprint)
    app.register_blueprint(generate_blueprint)

    return app


app = create_app()
