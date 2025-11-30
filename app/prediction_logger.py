"""
Prediction logger for storing predictions in cloud storage.
"""
import os
import logging
from datetime import datetime
from typing import Optional
import json

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Logs predictions to cloud storage (Google Drive or other)."""
    
    def __init__(self, environment: str = "dev", log_file_path: str = ""):
        """
        Initialize prediction logger.
        
        Args:
            environment: Environment name (dev or prod)
            log_file_path: Local path for storing prediction logs
        """
        self.environment = environment
        self.log_file_path = log_file_path or f"./logs/predicciones_{environment}.txt"
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
    
    async def log_prediction(
        self,
        text: str,
        label: str,
        score: float,
        timestamp: str
    ):
        """
        Log prediction to file (local and cloud).
        
        Args:
            text: Input text
            label: Predicted label
            score: Prediction score
            timestamp: Timestamp of prediction
        """
        try:
            # Format log entry
            log_entry = {
                "timestamp": timestamp,
                "text": text[:200],  # Truncate for logging
                "label": label,
                "score": float(score),
                "environment": self.environment
            }
            
            # Write to local file
            log_line = json.dumps(log_entry, ensure_ascii=False) + "\n"
            
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(log_line)
            
            logger.debug(f"Logged prediction: {label} (score: {score:.4f})")
            
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
            # Don't raise - logging failures shouldn't break predictions
    
