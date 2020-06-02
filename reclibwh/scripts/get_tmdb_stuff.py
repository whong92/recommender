import requests
from ..utils.ItemMetadata import ExplicitDataFromCSV
import os
import pandas as pd
import json
from tqdm import tqdm

data_folder = '/home/ong/personal/recommender/data/ml-20m-2'
dcsv = ExplicitDataFromCSV(True, data_folder=data_folder)

links = pd.read_csv(os.path.join(data_folder, 'links.csv'))
print(dcsv.item_map)
links = links.merge(dcsv.item_map, left_on='movieId', right_on='item')
links['item'] = links.index

data = {}
tmdb_url = "https://api.themoviedb.org/3/movie/{:d}?api_key=c2d254179dfca9990118a4b4371827c5&language=en-US"

for i, row in tqdm(links.iterrows()):
    try:
        url = tmdb_url.format(int(row['tmdbId']))
        res = requests.get(url)
        data[row['item']] = res.json()

        if i%1000 == 0:
            with open(os.path.join(data_folder, 'tmdb_data.json'), 'w') as fp:
                json.dump(data, fp, indent=4)
    except Exception as e:
        print('fucked up {}'.format(e))

with open(os.path.join(data_folder, 'tmdb_data.json'), 'w') as fp:
    json.dump(data, fp, indent=4)

################################################################### 

# with open(os.path.join(data_folder, 'tmdb_data_new.json'), 'r') as fp:
#     data = json.load(fp)

dcsv.md_df['desc'] = dcsv.md_df.index.map(lambda p: data[str(p)].get("overview", "No description available").replace("\r", "") if str(p) in data else '\"No description available\"')
dcsv.md_df['desc'].map(lambda p: '\"' + p + '\"' if not p.startswith('\"') else p)

dcsv.md_df['poster_path'] = dcsv.md_df.index.map(lambda p: data[str(p)].get("poster_path", "") if str(p) in data else "")
dcsv.md_df['poster_path'] = dcsv.md_df['poster_path'].map(lambda p: ("https://image.tmdb.org/t/p/w440_and_h660_face" + p) if (p is not None and len(p) > 0) else "")

dcsv.md_df.to_csv(os.path.join(data_folder, 'metadata_2.csv'))