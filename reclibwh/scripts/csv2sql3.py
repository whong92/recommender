from reclibwh.utils.ItemMetadata import ExplicitDataFromCSV, ExplicitDataFromSql3

if __name__=="__main__":
    
    data_folder = '/home/ong/personal/recommender/data/ml-latest-small-2'

    dcsv = ExplicitDataFromCSV(True, data_folder=data_folder)
    dsql = ExplicitDataFromSql3(
        '/home/ong/personal/FiML/FiML/db_exp.sqlite3',
        'backend_rating', 'user_id', 'film_id', 'rating', 'backend_film', 'dataset_id')    

    md_df = dcsv.fetch_md(list(range(dcsv.M)))

    print(len(md_df.index.unique()))
    print(len(md_df))

    dsql.add_items(
        **{
            'item_ids': md_df.index,
            'names': md_df['title'],
        }
    )