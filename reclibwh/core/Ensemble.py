import numpy as np
from ..utils.ItemMetadata import ExplicitDataFromCSV

from reclibwh.core.AsymSVD import AsymSVDCachedEnv
from reclibwh.core.Models import STANDARD_KERAS_SAVE_FMT, initialize_from_json
from reclibwh.core.EvalProto import AUCEval
from reclibwh.core.Environment import UpdateAlgo, Environment, RecAlgo, SimAlgo
from reclibwh.core.ALS import ALSEnv
from ..data.PresetIterators import AUC_data_iter_preset

class EnsembleUpdateAlgo(UpdateAlgo):

    def __init__(self, env: Environment):
        UpdateAlgo.__init__(self)
        self.__env = env
        self.__models = None

    def __initialize(self):
        self.__models = self.__env['model']

    def update_user(self, datas):
        self.__initialize()
        for model, data in zip(self.__models, datas): model.update_user(data)

    def make_update_data(self, data):
        self.__initialize()
        return [
            model.make_update_data(data) for model in self.__models
        ]

class EnsembleRecAlgo(RecAlgo):

    def __init__(self, env: Environment):
        UpdateAlgo.__init__(self)
        self.__env = env
        self.__models = None

    def __initialize(self):
        self.__models = self.__env['model']

    def recommend(self, user):
        self.__initialize()
        M = self.__env['data_conf']['M']
        p = np.ones(shape=(len(user), M), dtype=float)
        for model in self.__models:
            rec_aps, rec_ps = model.recommend(user)
            for u, (rec_ap, rec_p) in enumerate(zip(rec_aps, rec_ps)):
                p[u] = np.multiply(p[u], rec_p[np.argsort(rec_ap)])
        return np.argsort(p)[:, ::-1], np.sort(p)[:, ::-1]

class EnsembleSimAlgo(SimAlgo):

    def __init__(self, env: Environment):
        UpdateAlgo.__init__(self)
        self.__env = env
        self.__models = None

    def __initialize(self):
        self.__models = self.__env['model']

    def similar(self, item):
        self.__initialize()
        M = self.__env['data_conf']['M']
        p = np.ones(shape=(len(item), M), dtype=float)
        for model in self.__models:
            sim_aps, sim_ps = model.similar(item)
            for i, (sim_ap, sim_p) in enumerate(zip(sim_aps, sim_ps)):
                p[i] = np.multiply(p[i], sim_p[np.argsort(sim_ap)])
        return np.argsort(p)[:, ::-1], np.sort(p)[:, ::-1]

class EnsembleEnv(
    Environment,
    EnsembleUpdateAlgo,
    EnsembleRecAlgo,
    AUCEval, EnsembleSimAlgo
):

    def __init__(
        self, path, model, data, state, med_score=3.0
    ):
        Environment.__init__(self, path, model, data, state)
        EnsembleUpdateAlgo.__init__(self, self) # <--- this is silly, fix!
        EnsembleRecAlgo.__init__(self, self)
        AUCEval.__init__(self, self, self, med_score)
        EnsembleSimAlgo.__init__(self, self)

if __name__=="__main__":

    asvdc_dir = "models/ASVDC_2020-12-16.16-11.00"
    als_dir = "models/ALS_2020-12-16.15-48-46-serve"
    data_dir = "data/ml-20m"

    d = ExplicitDataFromCSV(True, data_folder=data_dir)
    item_mean_ratings = np.array(d.get_item_mean_ratings(None))
    Bi = np.array(item_mean_ratings)

    data_conf = {"M": d.M, "N": d.N}
    env_vars = {
        "save_fmt": STANDARD_KERAS_SAVE_FMT,
        "data_conf": data_conf, "data": {}
    }
    m = initialize_from_json(data_conf=data_conf, config_path="ALS.json.template")
    als_env = ALSEnv(als_dir, m, d, env_vars)

    data_conf = {"M": d.M, "N": d.N}
    mc = initialize_from_json(data_conf=data_conf, config_path="SVD_asym_cached.json.template")

    env_varsc = {
        'data_conf': data_conf,
        'data': {'rnorm': {'loc': 0.0, 'scale': 5.0}, 'Bi': Bi}
    }
    asvdc_env = AsymSVDCachedEnv(asvdc_dir, mc, None, env_varsc)

    # Test the ensemble
    Utrain, Utest = d.make_training_datasets(dtype='sparse')
    max_num_ratings = 50
    num_ratings = Utrain.getnnz(axis=1)
    users_to_test = np.nonzero(num_ratings < max_num_ratings)[0]
    users_to_test = users_to_test[np.arange(0, len(users_to_test), max(len(users_to_test) // 300, 1))]
    auc_test_data = AUC_data_iter_preset(Utest, rows=users_to_test)
    auc_train_data = AUC_data_iter_preset(Utrain, rows=users_to_test)

    ensemble_env = EnsembleEnv(
        "bla",
        [als_env, asvdc_env],
        {"auc_data": {'test': auc_test_data, 'train': auc_train_data},},
        {'data_conf': data_conf}
    )
    # ensemble_env.evaluate()
    rows = np.array([0, 0, 0, 0])
    cols = np.array([1, 2, 3, 4])
    vals = np.array([1., 2., 3., 4.])
    datas = ensemble_env.make_update_data((rows, cols, vals))
    ensemble_env.update_user(datas)