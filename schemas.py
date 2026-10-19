from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    features: list[float] = Field(
        min_length=30,
        max_length=30,
        description="The 30 diagnostic features in training order",
    )


class PredictionResponse(BaseModel):
    prediction: str
    message: str