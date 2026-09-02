import os
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.prediction_service import predict_image


router = APIRouter()


UPLOAD_DIR = "uploads"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".npy",
}


@router.get("/")
def root():
    return {
        "message": "TerraSpectra Backend Running"
    }


@router.get("/health")
def health():
    return {
        "status": "healthy"
    }


@router.post("/upload")
async def upload(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are allowed."
        )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    # Generate a unique filename
    # so two users/files don't overwrite each other.
    unique_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        unique_filename
    )

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        with open(
            file_path,
            "wb"
        ) as buffer:
            buffer.write(contents)

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to save image: {str(exc)}"
        )

    return {
        "filename": file.filename,
        "file_path": file_path,
        "message": "Image uploaded successfully."
    }


@router.post("/predict")
async def predict(image_path: str):

    if not image_path:
        raise HTTPException(
            status_code=400,
            detail="Image path is required."
        )

    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail="Image file not found."
        )

    try:

        result = predict_image(
            image_path
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )


@router.get("/results/{id}")
async def get_result(id: int):

    return {
        "id": id,
        "status": "success",
        "message": "Result endpoint working"
    }


@router.delete("/image/{id}")
def delete_image(id: int):

    return {
        "id": id,
        "status": "success",
        "message": "Image deleted successfully"
    }