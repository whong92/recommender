class Recommender:
    def __init__(self, model_file=None):
        self.model_file = model_file
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        return

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, items, users):
        raise NotImplementedError

    def recommend(self, user):
        raise NotImplementedError
