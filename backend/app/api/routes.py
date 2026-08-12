import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.prediction_service import predict_image

router = APIRouter()


@router.get("/")
def root():
    return {"message": "TerraSpectra Backend Running"}


@router.get("/health")
def health():
    return {"status": "healthy"}

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {
        "filename": file.filename,
        "file_path": file_path,
        "message": "File uploaded and saved successfully"
    }
@router.post("/predict")
async def predict(image_path: str):
    return predict_image(image_path)

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