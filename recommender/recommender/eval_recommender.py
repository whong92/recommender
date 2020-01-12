from recommender.utils.eval_utils import compute_auc, compute_ap
# from recommender.recommender.recommenderMF import RecommenderMF
from recommender.recommender.recommenderALS import RecommenderALS
#from recommender.recommender.recommenderCFSimple import RecommenderCFSimple
import os
import numpy as np
from tqdm import tqdm
from recommender.utils.ItemMetadata import ExplicitDataFromCSV

if __name__=="__main__":

    data_folder = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small'
    model_folder = 'D:\\PycharmProjects\\recommender\\models'

    ratings_csv = os.path.join(data_folder, 'ratings_sanitized.csv')
    user_map_csv = os.path.join(data_folder, 'user_map.csv')
    item_map_csv = os.path.join(data_folder, 'item_map.csv')
    md_csv = os.path.join(data_folder, 'metadata.csv')
    stats_csv = os.path.join(data_folder, 'stats.csv')
    df_train = os.path.join(data_folder, 'ratings_train.csv')
    df_test = os.path.join(data_folder, 'ratings_test.csv')
    d = ExplicitDataFromCSV(True,
                            ratings_csv=ratings_csv,
                            user_map_csv=user_map_csv,
                            item_map_csv=item_map_csv,
                            md_csv=md_csv,
                            stats_csv=stats_csv,
                            ratings_train_csv=df_train,
                            ratings_test_csv=df_test
                            )

    # model_path = os.path.join(model_folder, 'model/07201914185949/')
    model_path = os.path.join(model_folder, 'ALS_01202018225751')
    # model_path = os.path.join(model_folder, 'CF_2019_11_02')
    # rmf = RecommenderMF(mode='predict', model_path=model_path)
    # rmf = RecommenderCFSimple(mode='predict', model_file=model_path)
    rmf = RecommenderALS(mode='predict', model_path=model_path)

    # explicit score prediction
    F = rmf.predict(np.array(d.df_test['user']), np.array(d.df_test['item']))
    print("bla: {:f}".format(np.sqrt(np.mean(np.power(F['rhat'] - d.df_test['rating'], 2)))))

    # implicit score prediction
    M = len(d.user_map)
    AUC = -np.ones(shape=(M,))
    MAP = -np.ones(shape=(M,))
    df_test_rel = d.df_test[d.df_test['rating'] > 3.0] # relevant positives are scores above 3.0
    for m in tqdm(range(100)):
        auc = compute_auc(rmf, m, df_test_rel, d.df_train)
        if auc < 0:
            continue
        AUC[m] = auc
    AUC = AUC[AUC>-1]
    mAUC = np.mean(AUC)

    print("mean AUC: {:f}".format(mAUC))