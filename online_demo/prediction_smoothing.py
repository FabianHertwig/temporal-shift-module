class PredictionSmoothing:
    def __init__(self, size: int):
        self.size = size
        self.history = []

    def add_prediction(self, prediction: int):
        self.history.append(prediction)
        if len(self.history) > self.size:
            self.history = self.history[1:]
        assert len(self.history) <= self.size

    def get_most_common_prediction(self):
        return max(set(self.history), key=self.history.count)