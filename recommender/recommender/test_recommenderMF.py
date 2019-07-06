from ..utils.utils import csv2df
from .recommenderMF import RecommenderMF
import pandas as pd
import numpy as np

if __name__=="__main__":
    src_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\ratings.csv'
    df, user_map, item_map, N, M = csv2df(src_csv,
                                          'movieId', 'userId', 'rating', return_cat_mapping=True)

    # prediction
    df_mov = pd.read_csv('D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\movies.csv')
    rmf = RecommenderMF(mode='predict',
                        model_path='D:/PycharmProjects/recommender/models/model/1556408619',
                        lsh_path='D:/PycharmProjects/recommender/models/model')

    thing = df.groupby('item').count()['user']
    thing = thing.loc[thing > 10]

    rmf._make_hash()
    rmf._update_hash(items=thing.index)
    rmf._save_hash('./bla')

    user = 55
    user_df = df.loc[df.user_cat == user]

    pred = rmf.predict(np.array(user_df['user']), np.array(user_df['item']))
    user_df['rhat'] = pred['rhat']
    user_df.rename(columns={'item': 'item_code'}, inplace=True)
    user_df = user_df.merge(item_map, left_on='item_code', right_on='item_cat')

    sorted_items, sorted_score = rmf.recommend_lsh(user)
    sorted_items = pd.DataFrame(
        {
            'item_code': sorted_items,
            'rhat': sorted_score,
        }
    )
    sorted_items = sorted_items.merge(item_map, left_on='item_code', right_on='item_cat')

    print(sorted_items.merge(df_mov, left_on='item', right_on='movieId')[:10])
    print(user_df.sort_values('rating').merge(df_mov, left_on='item', right_on='movieId')[:10])

    print(sorted_items.merge(df_mov, left_on='item', right_on='movieId')[-10:])
    print(user_df.sort_values('rating').merge(df_mov, left_on='item', right_on='movieId')[-10:])


