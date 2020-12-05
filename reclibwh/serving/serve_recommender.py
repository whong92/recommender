import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

from .RecommenderService import MFAsymRecService, sanitize_update_req_data, ALSRecommenderService
from ..utils.ItemMetadata import ExplicitDataFromCSV
from flask import Flask, Blueprint, request, app, current_app, Response, jsonify
from flask_cors import CORS
import traceback

import argparse

# TODO: remove all use of postgres, get user ratings via the update request instead!

class ValidationError(Exception):

    def __init__(self, msg: str):
        super(ValidationError, self). __init__()
        self.msg = msg
    
    def get_msg(self):
        return self.msg


rec_blueprint = Blueprint('rec_blueprint', __name__, template_folder='templates')

@rec_blueprint.route('/')
def say_hello():
    return 'hello world'

@rec_blueprint.route('/user_recommend', methods=('POST',))
def user_update_and_recommend():
    try:
        rc = current_app.stuff['rec_context']
        update_req = request.get_json()
        res = rc.update_and_recommend(sanitize_update_req_data(update_req))
        return jsonify(res)
    except ValidationError as e:
        traceback.print_exc()
        return Response('fuck {}'.format(e), 400)
    except Exception as e:
        traceback.print_exc()
        return Response('fuck {}'.format(e), 400)

    
def create_app(model_path, data_path):
    app = Flask(__name__)
    app.register_blueprint(rec_blueprint)
    CORS(app)
    data = ExplicitDataFromCSV(True, data_folder=data_path)
    # app.stuff = {'rec_context': MFAsymRecService(save_path=model_path, data=data)}
    app.stuff = {'rec_context': ALSRecommenderService(save_path=model_path, data=data)}
    return app

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to the model", required=True)
    parser.add_argument("--data_path", help="path to the data", default="data/ml-20m")
    parser.add_argument("--host", help="port", default="0.0.0.0")
    parser.add_argument("--port", help="port", default="5000")
    args = parser.parse_args()

    app = create_app(args.model_path, args.data_path)
    app.run(host=args.host, port=args.port)