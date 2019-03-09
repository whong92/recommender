from .utils import *

def test_csv2df():
    df,_,_ = csv2df('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv', 'userId', 'movieId', 'rating')
    print(df.head())

def test_csv2umCSR():
    df,_,_ = csv2df('D:/PycharmProjects/recommender/data/ml-latest-small/ratings.csv', 'userId', 'movieId', 'rating')
    um = df2umCSR(df)
    print(um[:30,:30])

if __name__ == "__main__":
    test_csv2df()
    test_csv2umCSR()