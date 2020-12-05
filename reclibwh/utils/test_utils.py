import shutil
import os

def refresh_dir(d):

    if not os.path.exists(d):
        os.makedirs(d)
    else:
        shutil.rmtree(d)
        os.makedirs(d)