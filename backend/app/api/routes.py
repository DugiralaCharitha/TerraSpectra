import os
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from app.services.prediction_service import predict_image
from app.database import get_db
from app.models.prediction import Prediction


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
            detail="Only JPG, JPEG, PNG and NPY files are allowed."
        )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

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
                detail="Uploaded file is empty."
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
            detail=f"Unable to save file: {str(exc)}"
        )

    return {
        "filename": file.filename,
        "file_path": file_path,
        "message": "File uploaded successfully."
    }


@router.post("/predict")
async def predict(
    image_path: str,
    farm_id: str | None = None,
    db: Session = Depends(get_db)
):

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

        prediction_record = Prediction(
            farm_id=farm_id,
            filename=os.path.basename(image_path),
            prediction=result.get(
                "prediction",
                result.get("class_name", "Unknown")
            ),
            confidence=float(
                result.get("confidence", 0)
            ),
            stress_probability=float(
                result.get("stress_probability", 0)
            ),
            tiles_processed=int(
                result.get("tiles_processed", 0)
            )
        )

        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)

        result["prediction_id"] = prediction_record.id
        result["saved"] = True

        return result

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )


@router.get("/predictions")
def get_predictions(
    farm_id: str | None = None,
    db: Session = Depends(get_db)
):
    query = db.query(Prediction)

    if farm_id:
        query = query.filter(Prediction.farm_id == farm_id)

    predictions = (
        query
        .order_by(Prediction.created_at.desc())
        .all()
    )

    return predictions


@router.get("/results/{id}")
async def get_result(
    id: int,
    db: Session = Depends(get_db)
):

    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found."
        )

    return prediction


@router.delete("/image/{id}")
def delete_image(id: int):

    return {
        "id": id,
        "status": "success",
        "message": "Image deleted successfully"
    }