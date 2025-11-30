import os
import logging
from datetime import datetime
from typing import Optional
import json

logger = logging.getLogger(__name__)


class PredictionLogger:
    def __init__(self, environment: str = "dev", log_file_path: str = ""):
        self.environment = environment
        self.log_file_path = log_file_path or f"./logs/predicciones_{environment}.txt"
        
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
    
    async def log_prediction(
        self,
        text: str,
        label: str,
        score: float,
        timestamp: str
    ):
        try:
            log_entry = {
                "timestamp": timestamp,
                "text": text[:200],
                "label": label,
                "score": float(score),
                "environment": self.environment
            }
            
            log_line = json.dumps(log_entry, ensure_ascii=False) + "\n"
            
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(log_line)
            
            logger.debug(f"Logged prediction: {label} (score: {score:.4f})")
            
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
    
