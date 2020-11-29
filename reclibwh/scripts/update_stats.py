from reclibwh.utils.ItemMetadata import ExplicitDataFromCSV
import numpy as np


if __name__=="__main__":

    data_folder = '/home/ong/personal/recommender/data/ml-latest-small'

    print('loading data....')
    d = ExplicitDataFromCSV(True, data_folder=data_folder)

    # print('computing stats....')
    # d.update_stats()

    # print('saving....')
    # d.save(data_folder)

    print(d.get_item_mean_ratings(np.arange(d.M)))