from .minhash import MinHashSignature
from .cosineSimHash import CosineSimHash

def MakeSignature(name, *args, **kwargs):
    switcher = {
        'MinHash' : MinHashSignature,
        'CosineHash': CosineSimHash
    }
    assert name in switcher, "{0} not in {1}".format(name, switcher.keys())
    return switcher[name](*args, **kwargs)