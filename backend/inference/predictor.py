class Predictor:
    def __init__(self):
        self.model = None

    def load_model(self, model):
        """Attach a trained model to the predictor."""
        self.model = model

    def predict(self, image_path: str):
        """Run prediction using the loaded model."""
        if self.model is None:
            return {
                "status": "error",
                "message": "Model is not loaded",
                "image_path": image_path,
            }

        prediction = self.model.predict(image_path)

        return {
            "status": "success",
            "prediction": prediction,
            "image_path": image_path,
        }