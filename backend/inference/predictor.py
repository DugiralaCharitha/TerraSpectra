from inference.predictor import Predictor
class Predictor:
    def __init__(self):
        self.model = None

    def predict(self, image_path: str):
        return {
            "status": "success",
            "prediction": "placeholder",
            "image_path": image_path,
        }