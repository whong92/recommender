from recommender.utils.eval_utils import compute_auc, compute_ap, eval_model
from recommender.recommender.recommenderMF import RecommenderMF
from recommender.recommender.recommenderALS import RecommenderALS
#from recommender.recommender.recommenderCFSimple import RecommenderCFSimple
import os, shutil
import numpy as np
from tqdm import tqdm
from recommender.utils.ItemMetadata import ExplicitDataFromCSV
import pandas as pd
import re

if __name__=="__main__":

    data_folder = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small'
    model_folder = 'D:\\PycharmProjects\\recommender\\models'

    d = ExplicitDataFromCSV(True, data_folder=data_folder)

    # model_path = os.path.join(model_folder, 'MF_2020-02-08.01-32-49')
    model_path = os.path.join(model_folder, 'ALS_2020-03-14.13-16-49')
    # model_path = os.path.join(model_folder, 'CF_2019_11_02')

    AUCs = []

    # recommender ALS
    match = re.compile('epoch-[0-9]{3}')
    src_dirs = list(map(
        lambda x: os.path.join(model_path, x),
        filter(lambda x: match.match(x) is not None, os.listdir(model_path))
    ))
    dst_dir = os.path.join(model_path, 'epoch-best')

    # recommender MF
    # m = re.compile('model.[0-9]{3}-[0-9.]{3}.h5')
    # src_dirs = map(
    #     lambda x: os.path.join(model_path, x),
    #     filter(lambda x: m.match(x) is not None, os.listdir(model_path))
    # )
    # dst_dir = os.path.join(model_path, 'model_best.h5')

    for src_dir in src_dirs:

        print('eval {}'.format(src_dir))

        # for recommender ALS
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

        rmf = RecommenderALS(mode='predict', model_path=model_path)
        # rmf = RecommenderCFSimple(mode='predict', model_file=model_path)
        # rmf = RecommenderALS(mode='predict', model_path=model_path)

        # explicit score prediction
        F = rmf.predict(np.array(d.df_test['user']), np.array(d.df_test['item']))
        print(F['rhat'].shape)
        print("bla: {:f}".format(np.sqrt(np.mean(np.power(F['rhat'] - d.df_test['rating'], 2)))))

        # implicit score prediction

        df_test_rel = np.array(
            d.df_test.loc[d.df_test['rating'] > 3.0, 'item']) # relevant positives are scores above 3.0

        AUC = eval_model(rmf, d)

        AUC = AUC[AUC>-1]
        mAUC = np.mean(AUC)
        AUCs.append(mAUC)

        print("mean AUC: {:f}".format(mAUC))

    pd.DataFrame({
        'epoch': list(range(len(src_dirs))), 'AUC': AUCs
    }).to_csv(os.path.join(model_path, 'AUC.csv'))