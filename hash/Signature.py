from . import *

class Signature:
    def __init__(self):
        self.hashes = []
        return

    def generate_signature(self, x):
        raise NotImplementedError

