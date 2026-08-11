from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():
    return {"message": "TerraSpectra Backend Running"}


@router.get("/health")
def health():
    return {"status": "healthy"}