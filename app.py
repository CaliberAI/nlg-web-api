from flask import Flask
from generate import generate_blueprint
from index import index_blueprint


def create_app():
    app = Flask(__name__)
    app.register_blueprint(index_blueprint)
    app.register_blueprint(generate_blueprint)

    return app
