import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

from ..recommender.recommenderMF import RecommenderMFAsymCached
from ..recommender.recommenderALS import RecommenderALS
from ..recommender.recommenderEnsemble import RecommenderEnsemble
from ..utils.ItemMetadata import ExplicitDataFromCSV, ExplicitDataFromPostgres, ExplicitDataFromSql3
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import os
import asyncio
import time
import numpy as np

from flask import g
from flask import Flask, Blueprint, request, app, current_app, Response, jsonify
from flask_script import Manager, Server, Option
from flask_cors import CORS

import argparse

class ValidationError(Exception):

    def __init__(self, msg: str):
        super(ValidationError, self). __init__()
        self.msg = msg
    
    def get_msg(self):
        return self.msg

class RecommenderContext:

    def __init__(self, postgres_config, model_path_ex, model_path_im):

        recs = []
        datas = []
        if model_path_im: 
            recs.append(RecommenderALS(mode='predict',model_path=model_path_im))
            dunorm = ExplicitDataFromPostgres(
                postgres_config,
                rt='backend_rating', rt_user_col='user_id', rt_item_fk_col='film_id', rt_rating_col='rating',
                it='backend_film', it_item_id_col='dataset_id', it_item_mean_col='mean_rating', ut='auth_user', ut_id_col='id',
                user_offset=recs[-1].config['n_users'], normalize=None
            )
            datas.append(dunorm)
        if model_path_ex: 
            recs.append(RecommenderMFAsymCached(mode='predict', model_path=model_path_ex))
            dnorm = ExplicitDataFromPostgres(
                postgres_config,
                rt='backend_rating', rt_user_col='user_id', rt_item_fk_col='film_id', rt_rating_col='rating',
                it='backend_film', it_item_id_col='dataset_id', it_item_mean_col='mean_rating', ut='auth_user', ut_id_col='id',
                user_offset=recs[-1].config['n_users'], normalize={'loc': 0.0, 'scale': 5.0}
            )
            datas.append(dnorm)

        self.rec = RecommenderEnsemble(recs)
        self.N_offset = self.rec.config['n_users']
        
        self.d = datas[0]
        self.rec.input_data(datas)
        self.updated_users = set({}) # in memory cache of which users have and have not been trained, think of a better fix than this!
        self.t = ThreadPoolExecutor(max_workers=1)
    
    def get_user_ratings(self, users):
        ratings, _ = self.d.make_training_datasets(dtype='dense', users=users)
        return ratings

    def submit_recommend_job(self, users):
        def recommend_job():
            m_users = max(users)
            print(m_users, self.rec.config['n_users'])
            
            if m_users >= self.rec.config['n_users']:
                self.rec.add_users(m_users - self.rec.config['n_users'] + 1)
                print(m_users - self.rec.config['n_users'] + 1)
            
            users_to_update = set(users).difference(self.updated_users)
            self.updated_users = self.updated_users.union(users_to_update)
            if len(users_to_update) > 0:
                self.rec.train_update(list(users_to_update), test=False)
            return self.rec.recommend(list(users))
        return self.t.submit(recommend_job)
    
    def submit_update_job(self, users):
        def update_job():
            m_users = max(users)
            if m_users >= self.rec.config['n_users']:
                self.rec.add_users(m_users - self.rec.config['n_users'] + 1)
            self.updated_users = self.updated_users.union(users)
            return self.rec.train_update(users, test=False)
        return self.t.submit(update_job)
    
    def join(self):        
        self.t.shutdown()

rec_blueprint = Blueprint('rec_blueprint', __name__, template_folder='templates')

@rec_blueprint.route('/')
def say_hello():
    return 'hello world'

@rec_blueprint.route('/user_recommend', methods=('POST',))
def user_recommend():
    try:
        rC = current_app.stuff['rec']
        stuff = request.get_json()
        if 'users' not in stuff: raise ValidationError('list of users required')
        users = stuff['users']
        users = [user+rC.N_offset for user in users]
        rated_users, rated, _ = rC.get_user_ratings(users)
        result = rC.submit_recommend_job(users)
        recs, dists = result.result()
        res = {}
        for user, rec, dist in zip(users, recs, dists):
            user_rated = np.array(rated[rated_users==(user)])
            mask = np.isin(rec, user_rated, invert=True)
            rec = rec[mask]
            dist = dist[mask]
            res[int(user-rC.N_offset)] = {'rec': list(map(lambda x: int(x), rec[:200])), 'dist': list(map(lambda x: float(x), dist[:200]))}
        return jsonify(res)
    except ValidationError as e:
        return Response('fuck {}'.format(e), 400)

@rec_blueprint.route('/user_update', methods=('POST',))
def user_update():
    try:
        rC = current_app.stuff['rec']
        stuff = request.get_json()
        users = stuff['users']
        if 'users' not in stuff: raise ValidationError('list of users required')
        users = [user+rC.N_offset for user in users]
        update = rC.submit_update_job(users) # update
        result = rC.submit_recommend_job(users) # recommend
        update.result() # wait for update
        recs, dists = result.result() # wait for results

        rated_users, rated, _ = rC.get_user_ratings(users)
        res = {}
        for user, rec, dist in zip(users, recs, dists):
            user_rated = np.array(rated[rated_users==(user)])
            mask = np.isin(rec, user_rated, invert=True)
            rec = rec[mask]
            dist = dist[mask]
            res[int(user-rC.N_offset)] = {'rec': list(map(lambda x: int(x), rec[:200])), 'dist': list(map(lambda x: float(x), dist[:200]))}
        return jsonify(res)
    except ValidationError as e:
        return Response('fuck {}'.format(e), 400)
    
def create_app(postgres_config, model_path_ex, model_path_im):
    app = Flask(__name__)
    app.register_blueprint(rec_blueprint)
    CORS(app)
    app.stuff = {'rec': RecommenderContext(postgres_config, model_path_ex, model_path_im)}
    return app

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--postgres_config", help="path to postgres config", required=True)
    parser.add_argument("--model_ex", help="path to the explicit model")
    parser.add_argument("--model_im", help="path to the implicit model")
    parser.add_argument("--host", help="port", default="0.0.0.0")
    parser.add_argument("--port", help="port", default="5000")
    args = parser.parse_args()

    app = create_app(args.postgres_config, args.model_ex, args.model_im)
    app.run(host=args.host, port=args.port)