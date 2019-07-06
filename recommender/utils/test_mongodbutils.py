from .mongodbutils import DataService
import numpy as np

if __name__=="__main__":

    dbconn = ("localhost", 27017)
    db_name = "test"
    url = "mongodb://{:s}:{:d}/".format(*dbconn)

    data = DataService(url, db_name)
    data.insert_features(
        "bla",
        {
            0: [0.,0.,0.],
            1: [1.,1.,1.],
            2: [2.,2.,2.]
        }
    )

    data.update_band(
        "bla", 3,
        {
            3: {1, 2, 3, 4},
            4: {9, 10, 11, 12},
            9: {0, 1, 3, 8},
        }
    )