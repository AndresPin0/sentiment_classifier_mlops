import os
import logging
from typing import Dict, List
from datetime import datetime
import asyncio
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from app.model_loader import ONNXModelLoader
from app.prediction_logger import PredictionLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Classifier API",
    description="API for sentiment classification using ONNX model",
    version="1.0.0"
)

model_loader = None

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/model.onnx")
PREDICTIONS_DEV_FILE = os.getenv("PREDICTIONS_DEV_FILE", "./logs/predicciones_dev.txt")
PREDICTIONS_PROD_FILE = os.getenv("PREDICTIONS_PROD_FILE", "./logs/predicciones_prod.txt")

log_file = PREDICTIONS_DEV_FILE if ENVIRONMENT == "dev" else PREDICTIONS_PROD_FILE
prediction_logger = PredictionLogger(environment=ENVIRONMENT, log_file_path=log_file)


class TextInput(BaseModel):
    text: str = Field(..., description="Text to classify", min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    text: str
    label: str
    score: float
    timestamp: str


@app.on_event("startup")
async def startup_event():
    global model_loader
    try:
        logger.info(f"Initializing model - Path: {MODEL_PATH}, URL: {MODEL_URL}")
        model_loader = ONNXModelLoader(model_path=MODEL_PATH, model_url=MODEL_URL)
        await model_loader.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/")
async def root():
    return {
        "message": "Sentiment Classifier API",
        "environment": ENVIRONMENT,
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    if model_loader is None or not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded(),
        "environment": ENVIRONMENT
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    if model_loader is None or not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        label, score = model_loader.predict(input_data.text)
        
        response = PredictionResponse(
            text=input_data.text,
            label=label,
            score=float(score),
            timestamp=datetime.utcnow().isoformat()
        )
        
        asyncio.create_task(
            prediction_logger.log_prediction(
                text=input_data.text,
                label=label,
                score=score,
                timestamp=response.timestamp
            )
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(texts: List[str]):
    if model_loader is None or not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for text in texts:
            label, score = model_loader.predict(text)
            result = {
                "text": text,
                "label": label,
                "score": float(score),
                "timestamp": datetime.utcnow().isoformat()
            }
            results.append(result)
            
            asyncio.create_task(
                prediction_logger.log_prediction(
                    text=text,
                    label=label,
                    score=score,
                    timestamp=result["timestamp"]
                )
            )
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/logs/descargar")
async def download_logs(environment: str = ENVIRONMENT):
    if environment not in ("dev", "prod"):
        raise HTTPException(status_code=400, detail="Invalid environment. Use 'dev' or 'prod'.")

    file_path = PREDICTIONS_DEV_FILE if environment == "dev" else PREDICTIONS_PROD_FILE

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    return FileResponse(
        path=file_path,
        media_type="text/plain",
        filename=os.path.basename(file_path)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

