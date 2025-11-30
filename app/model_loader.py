"""
ONNX Model Loader for sentiment classification.
"""
import os
import sys
import logging
import numpy as np
from typing import Tuple, Optional
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class ONNXModelLoader:
    """Loads and manages ONNX model for inference."""
    
    def __init__(self, model_path: str, model_url: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            model_path: Local path to ONNX model file
            model_url: URL to download model from if not found locally (optional)
        """
        self.model_path = model_path
        self.model_url = model_url
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer = None
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    def _download_model_if_needed(self):
        """Download model from URL if it doesn't exist locally."""
        if os.path.exists(self.model_path):
            logger.info(f"Model file exists at: {self.model_path}")
            return
        
        if not self.model_url:
            raise FileNotFoundError(
                f"Model file not found at {self.model_path} and no MODEL_URL provided"
            )
        
        logger.info(f"Model not found locally, downloading from: {self.model_url}")
        
        try:
            # Import download utilities
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from scripts.download_model import download_model
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Download model
            download_model(self.model_url, self.model_path)
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model download failed: {self.model_path}")
            
            logger.info(f"Model downloaded successfully to: {self.model_path}")
            
        except ImportError:
            # Fallback to manual download using requests
            logger.warning("Could not import download script, trying manual download with requests...")
            import requests
            import re
            
            # Try to extract file ID from Google Drive URL
            if "drive.google.com" in self.model_url:
                file_id = None
                if "/d/" in self.model_url:
                    file_id = self.model_url.split("/d/")[1].split("/")[0]
                elif "id=" in self.model_url:
                    file_id = self.model_url.split("id=")[1].split("&")[0]
                
                if file_id:
                    # Use the download script logic
                    session = requests.Session()
                    URL = "https://drive.google.com/uc?export=download"
                    response = session.get(URL, params={"id": file_id}, stream=True)
                    
                    # Check for confirmation token
                    token = None
                    for pattern in [r"confirm=([0-9A-Za-z-_]+)", r'name="confirm" value="([^"]+)"']:
                        match = re.search(pattern, response.text)
                        if match:
                            token = match.group(1)
                            break
                    
                    if token:
                        logger.info("Large file detected, applying confirmation token...")
                        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
                    
                    # Validate it's not HTML
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" in content_type.lower():
                        raise ValueError("Received HTML instead of file (Drive blocked it)")
                else:
                    # Direct download
                    response = requests.get(self.model_url, stream=True)
                    response.raise_for_status()
            else:
                # Direct download (not Google Drive)
                response = requests.get(self.model_url, stream=True)
                response.raise_for_status()
            
            # Download the file
            total_size = 0
            with open(self.model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            if total_size < 100_000:
                raise ValueError("Downloaded file is too small to be a valid ONNX model")
            
            logger.info(f"Model downloaded successfully: {self.model_path} ({total_size / 1024 / 1024:.2f} MB)")
        
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise
        
    async def load_model(self):
        """Load ONNX model and tokenizer."""
        try:
            # Download model if needed
            self._download_model_if_needed()
            
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Loading ONNX model from: {self.model_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options,
                providers=['CPUExecutionProvider']
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.session is not None and self.tokenizer is not None
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (label, score)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)
            
            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            )
            
            logits = outputs[0]
            
            probabilities = self._softmax(logits[0])
            
            predicted_class = np.argmax(probabilities)
            score = float(probabilities[predicted_class])
            
            label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
            
            return label, score
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

