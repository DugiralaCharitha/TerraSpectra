from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    farm_id = Column(String, nullable=True)
    filename = Column(String, nullable=False)

    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    stress_probability = Column(Float, nullable=False)

    tiles_processed = Column(Integer, nullable=False)

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )