from .recommenderMF import RecommenderMF
import pandas as pd
from ..utils.mongodbutils import DataService
from ..hash.LSH import LSHDB

if __name__=="__main__":

    dbconn = ("localhost", 27017)
    db_name = "test"
    url = "mongodb://{:s}:{:d}/".format(*dbconn)
    ds = DataService(url, db_name)

    ds.drop_features('hullabalooza')
    for i in range(10):
        ds.drop_lsh('hullabalooza', i)

    # prediction
    df_mov = pd.read_csv('D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\movies.csv')
    rmf = RecommenderMF(
        mode='predict',
        model_path='D:/PycharmProjects/recommender/models/model/1557057050',
        lsh_path='D:/PycharmProjects/recommender/models/model'
    )
    rmf._make_hash(
        signature_t='CosineHash', lsh_t='LSHDB',
        lsh_kwargs={'data_service':ds,'pref':'hullabalooza','num_bands': 5},
        signature_kwargs={'num_row': 20, 'num_hpp': 200}
    )
    rmf._update_hash(None)
    rmf._save_hash('D:/PycharmProjects/recommender/models/model/')