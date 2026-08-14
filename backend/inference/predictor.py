from pathlib import Path


class Predictor:
    def __init__(self):
        self.model = None

    def predict(self, image_path: str):
        image = Path(image_path)

        if not image.exists():
            return {
                "status": "error",
                "message": "Image file does not exist",
                "image_path": image_path,
            }

        if not image.is_file():
            return {
                "status": "error",
                "message": "Image path is not a file",
                "image_path": image_path,
            }

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