from ..recommender.recommenderALS import RecommenderALS
from ..utils.ItemMetadata import ExplicitDataFromCSV, ExplicitDataFromSql3
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import os
import asyncio
import time
import numpy as np

from flask import g
from flask import Flask, request, app, Response, jsonify
from flask_script import Manager, Server

app = Flask(__name__)
manager = Manager(app)

class ValidationError(Exception):

    def __init__(self, msg: str):
        super(ValidationError, self). __init__()
        self.msg = msg
    
    def get_msg(self):
        return self.msg

class RecommenderContext:

    def __init__(self, data_folder, model_path):

        self.rec = RecommenderALS(mode='predict', model_path=model_path)
        self.N_offset = self.rec.config['n_users']
        self.d = ExplicitDataFromSql3(
            '/home/ong/personal/FiML/FiML/db.sqlite3',
            rt='backend_rating',
            rt_user_col='user_id',
            rt_item_fk_col='film_id',
            rt_rating_col='rating',
            it='backend_film', 
            it_item_id_col='id',
            ut='auth_user',
            ut_id_col='id',
            user_offset=self.N_offset
        )
        self.rec.input_data(self.d)
        self.updated_users = set({}) # in memory cache of which users have and have not been trained, think of a better fix than this!
        self.t = ThreadPoolExecutor(max_workers=1)

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
                self.rec.train_update(list(users_to_update))
            return self.rec.recommend(list(users))
        return self.t.submit(recommend_job)
    
    def submit_update_job(self, users):
        def update_job():
            m_users = max(users)
            if m_users >= self.rec.config['n_users']:
                self.rec.add_users(m_users - self.rec.config['n_users'] + 1)
            self.updated_users = self.updated_users.union(users)
            return self.rec.train_update(users)
        return self.t.submit(update_job)
    
    def join(self):        
        self.t.shutdown()

@app.route('/')
def say_hello():
    return 'hello world'

@app.route('/user_recommend', methods=('POST',))
def user_recommend():
    try:
        rec = app.stuff['rec']
        stuff = request.get_json()
        if 'users' not in stuff: raise ValidationError('list of users required')
        users = stuff['users']
        users = [user+rec.N_offset for user in users]
        result = rec.submit_recommend_job(users)
        recs, dists = result.result()
        res = {
            user: {'rec': list(map(lambda x: int(x), rec)), 'dist': list(map(lambda x: float(x), dist))} for user, rec, dist in zip(users, recs[:,:200], dists[:,:200])
        }
        return jsonify(res)
    except ValidationError as e:
        return Response('fuck {}'.format(e), 400)

@app.route('/user_update', methods=('POST',))
def user_update():
    try:
        rec = app.stuff['rec']
        stuff = request.get_json()
        users = stuff['users']
        if 'users' not in stuff: raise ValidationError('list of users required')
        users = [user+rec.N_offset for user in users]
        update = rec.submit_update_job(users) # update
        result = rec.submit_recommend_job(users) # recommend
        update.result() # wait for update
        recs, dists = result.result() # wait for results
        res = {
            user: {'rec': list(map(lambda x: int(x), rec)), 'dist': list(map(lambda x: float(x), dist))} for user, rec, dist in zip(users, recs, dists)
        }
        return jsonify(res)
    except ValidationError as e:
        return Response('fuck {}'.format(e), 400)
    

data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'
model_folder = '/home/ong/personal/recommender/models'
model_path = os.path.join(model_folder, 'ALS_2020-04-28.00-22-47')

class CustomServer(Server):
    def __call__(self, app, *args, **kwargs):
        app.stuff = {
            'rec': RecommenderContext(data_folder, model_path)
        }
        #Hint: Here you could manipulate app
        return Server.__call__(self, app, *args, **kwargs)

# Remeber to add the command to your Manager instance
manager.add_command('runserver', CustomServer())

if __name__ == "__main__":
    manager.run()