from recommender.utils.eval_utils import compute_auc, compute_ap, eval_model, AUCCallback
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
    model_path = os.path.join(model_folder, 'ALS_2020-04-12.12-18-49')

    d = ExplicitDataFromCSV(True, data_folder=data_folder)
    rmf = RecommenderALS(mode='predict', model_path=model_path)

    AUCC = AUCCallback(data=d, M=np.arange(0,d.N,d.N//300), batchsize=10)
    AUCC.set_model(rmf)
    AUCC.on_epoch_end(0)