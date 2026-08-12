from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"


def predict_image(image_path: str):
    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    image_path = image_path.resolve()

    if not image_path.is_file():
        return {
            "status": "error",
            "message": "Image file not found",
            "image_path": str(image_path),
        }

    extension = image_path.suffix.lower()

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    if extension not in ALLOWED_EXTENSIONS:
        return {
            "status": "error",
            "message": "Unsupported image format",
            "image_path": str(image_path),
        }

    return {
        "status": "success",
        "prediction": "placeholder",
        "image_path": str(image_path),
    }