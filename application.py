from flask import Flask
from flask_cors import CORS
from generate import generate_blueprint
from index import index_blueprint


def create_application():
    application = Flask(__name__)
    CORS(application)
    application.register_blueprint(index_blueprint)
    application.register_blueprint(generate_blueprint)

    return application


if __name__ == "__main__":
    application = create_application()
    application.run()
