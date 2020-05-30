from reclibwh.utils.ItemMetadata import ExplicitDataFromCSV
from reclibwh.recommender.recommenderMF import RecommenderMF, RecommenderMFAsym
from datetime import datetime
import os

if __name__=="__main__":

    data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'
    model_folder = '/home/ong/personal/recommender/models'

    d = ExplicitDataFromCSV(True, data_folder=data_folder, normalize={'loc': 0.0, 'scale': 5.0})

    # training
    save_path = os.path.join(model_folder, "MF_{:s}".format(datetime.now().strftime("%Y-%m-%d.%H-%M-%S")))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rmf = RecommenderMFAsym(
        mode='train', n_users=d.N, n_items=d.M, n_ranked=d.Nranked,
        mf_kwargs={
            'config_path': '/home/ong/personal/recommender/reclibwh/core/model_templates/SVD_asym.json.template'
        },
        model_path=save_path
    )

    # array input format - onnly for smaller datasets
    rmf.input_data(d)
    rmf.train(early_stopping=False)

    rmf.save()
