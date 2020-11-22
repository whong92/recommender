from reclibwh.utils.ItemMetadata import ExplicitDataFromCSV, ExplicitDataFromSql3
import numpy as np

if __name__=="__main__":
    
    data_folder = '/home/ong/personal/recommender/data/ml-20m'

    dcsv = ExplicitDataFromCSV(True, data_folder=data_folder)
    dsql = ExplicitDataFromSql3(
        '/home/ong/personal/FiML/FiML/db.sqlite3',
        'backend_rating', 'user_id', 'film_id', 'rating', 'backend_film', 'dataset_id',
        it_item_mean_col='mean_rating',
        ut='auth_user',
        ut_id_col='id',
        user_offset=0, # who cares
    )    

    md_df = dcsv.fetch_md(list(range(dcsv.M)))
    mean_ratings = dcsv.get_item_mean_ratings(np.arange(dcsv.M))
    md_df = md_df.merge(mean_ratings, left_index=True, right_index=True)
    
    print(len(md_df.index.unique()))
    print(len(md_df))

    dsql.import_items(
        **{
            'item_ids': md_df.index,
            'names': md_df['title'],
            'desc': md_df['desc'],
            'poster_path': md_df['poster_path'],
            'mean_rating': md_df['rating_item_mean']
        }
    )

    print(dsql.get_item_mean_ratings())