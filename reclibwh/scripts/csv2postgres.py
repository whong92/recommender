from reclibwh.utils.ItemMetadata import ExplicitDataFromCSV, ExplicitDataFromPostgres
import numpy as np

if __name__=="__main__":
    
    data_folder = '/home/ong/personal/recommender/data/ml-20m-2'
    postgres_config='/home/ong/personal/recommender/reclibwh/apps/gcp.postgres.config'

    dcsv = ExplicitDataFromCSV(True, data_folder=data_folder)
    dsql = ExplicitDataFromPostgres(
        postgres_config,
        rt='backend_rating', rt_user_col='user_id', rt_item_fk_col='film_id', rt_rating_col='rating',
        it='backend_film', it_item_id_col='dataset_id', it_item_mean_col='mean_rating', ut='auth_user', ut_id_col='id',
        user_offset=0
    )    

    md_df = dcsv.fetch_md(list(range(dcsv.M)))
    mean_ratings = dcsv.get_item_mean_ratings(np.arange(dcsv.M))
    md_df = md_df.merge(mean_ratings, left_index=True, right_index=True)
    
    print(len(md_df.index.unique()))
    print(len(md_df))

    dsql.add_items(
        **{
            'item_ids': md_df.index,
            'names': md_df['title'],
            'desc': md_df['desc'],
            'poster_path': md_df['poster_path'],
            'mean_rating': md_df['rating_item_mean']
        }
    )

    print(dsql.get_item_mean_ratings())
        