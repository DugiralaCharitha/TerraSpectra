from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="TerraSpectra API",
    version="1.0.0",
    description="Backend API for TerraSpectra"
)

app.include_router(router)