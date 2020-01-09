from flask import Blueprint, jsonify

index_blueprint = Blueprint('index', __name__)


@index_blueprint.route('/')
def index():
    return jsonify('Try /generate')
