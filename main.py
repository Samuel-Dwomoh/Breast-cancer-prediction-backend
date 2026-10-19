from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from schemas import PredictionInput, PredictionResponse
from model_service import load_model, predict

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for a trained machine learning model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API is running"}


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "model"),
    }


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(
    data: PredictionInput,
    request: Request,
):
    try:
        model = request.app.state.model

        # Pass the 30 numeric values to the prediction service
        prediction = predict(model, data.features)

        return PredictionResponse(
            prediction=prediction,
            message="Prediction generated successfully",
        )

    except Exception as e:
        print(f"Prediction error: {type(e).__name__}: {e}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from e