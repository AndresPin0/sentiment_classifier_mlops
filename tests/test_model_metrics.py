"""
Tests for model quality metrics.
"""
import pytest
import os
import sys
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.model_loader import ONNXModelLoader


@pytest.fixture
def model_path():
    """Get model path from environment or default."""
    model_path = os.getenv("MODEL_PATH", "./models/model.onnx")
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")
    return model_path


@pytest.fixture
def test_data_path():
    """Get test data path from environment or default."""
    test_data_path = os.getenv("TEST_DATA_PATH", "./test_data/test_data.parquet")
    if not os.path.exists(test_data_path):
        pytest.skip(f"Test data file not found: {test_data_path}")
    return test_data_path


@pytest.fixture
def test_data(test_data_path):
    file_ext = Path(test_data_path).suffix.lower()
    
    if file_ext == '.parquet' or not file_ext:
        try:
            df = pd.read_parquet(test_data_path)
            
            text_col = None
            label_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if text_col is None and ('text' in col_lower or 'sentence' in col_lower or 'review' in col_lower):
                    text_col = col
                if label_col is None and ('label' in col_lower or 'sentiment' in col_lower or 'target' in col_lower):
                    label_col = col
            
            if text_col is None:
                text_col = df.columns[0]
            if label_col is None:
                label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            labels = df[label_col].tolist()
            labels_binary = []
            for label in labels:
                if isinstance(label, str):
                    if label.upper() == "POSITIVE":
                        labels_binary.append(1)
                    elif label.upper() == "NEGATIVE":
                        labels_binary.append(0)
                    else:
                        labels_binary.append(1 if label.upper().startswith("POS") else 0)
                else:
                    labels_binary.append(1 if int(label) > 0 else 0)
            
            return {
                "texts": df[text_col].tolist(),
                "labels": labels_binary
            }
        except Exception as e:
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(test_data_path)
                df = table.to_pandas()
                
                text_col = df.columns[0]
                label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                
                labels = df[label_col].tolist()
                labels_binary = []
                for label in labels:
                    if isinstance(label, (int, float)):
                        labels_binary.append(1 if int(label) > 0 else 0)
                    elif isinstance(label, str):
                        labels_binary.append(1 if label.upper() == "POSITIVE" else 0)
                    else:
                        labels_binary.append(1 if int(label) > 0 else 0)
                
                return {
                    "texts": df[text_col].tolist(),
                    "labels": labels_binary
                }
            except Exception as e2:
                raise Exception(f"Could not load parquet file from {test_data_path}: {str(e)}, {str(e2)}")
    else:
        try:
            with open(test_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise Exception(f"Could not load JSON file from {test_data_path}: {str(e)}")


@pytest.fixture
async def model_loader(model_path):
    """Create and load model loader."""
    loader = ONNXModelLoader(model_path=model_path)
    await loader.load_model()
    return loader


@pytest.mark.asyncio
async def test_model_accuracy(model_loader, test_data):
    """
    Test that model accuracy meets minimum threshold.
    Default threshold: 80%
    """
    min_accuracy = float(os.getenv("MIN_ACCURACY", "0.80"))
    
    texts = test_data.get("texts", [])
    true_labels = test_data.get("labels", [])
    
    if not texts or not true_labels:
        pytest.skip("Test data is empty or missing required fields")
    
    if len(texts) != len(true_labels):
        pytest.skip("Test data texts and labels have different lengths")
    
    predicted_labels = []
    for text in texts:
        label, _ = model_loader.predict(text)
        predicted_labels.append(1 if label == "POSITIVE" else 0)
    
    true_labels_binary = []
    for label in true_labels:
        if isinstance(label, (int, float)):
            true_labels_binary.append(1 if int(label) > 0 else 0)
        elif isinstance(label, str):
            true_labels_binary.append(1 if label.upper() == "POSITIVE" else 0)
        else:
            true_labels_binary.append(int(label))
    
    accuracy = accuracy_score(true_labels_binary, predicted_labels)
    
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Minimum Required: {min_accuracy:.4f} ({min_accuracy*100:.2f}%)")
    print(f"Test samples: {len(texts)}")
    
    assert accuracy >= min_accuracy, \
        f"Model accuracy {accuracy:.4f} is below minimum threshold {min_accuracy:.4f}"


@pytest.mark.asyncio
async def test_model_f1_score(model_loader, test_data):
    """
    Test that model F1 score meets minimum threshold.
    Default threshold: 0.80
    """
    min_f1 = float(os.getenv("MIN_F1", "0.80"))
    
    texts = test_data.get("texts", [])
    true_labels = test_data.get("labels", [])
    
    if not texts or not true_labels:
        pytest.skip("Test data is empty or missing required fields")
    
    if len(texts) != len(true_labels):
        pytest.skip("Test data texts and labels have different lengths")
    
    predicted_labels = []
    for text in texts:
        label, _ = model_loader.predict(text)
        predicted_labels.append(1 if label == "POSITIVE" else 0)
    
    true_labels_binary = []
    for label in true_labels:
        if isinstance(label, (int, float)):
            true_labels_binary.append(1 if int(label) > 0 else 0)
        elif isinstance(label, str):
            true_labels_binary.append(1 if label.upper() == "POSITIVE" else 0)
        else:
            true_labels_binary.append(int(label))
    
    f1 = f1_score(true_labels_binary, predicted_labels, average="weighted")
    
    print(f"\nModel F1 Score: {f1:.4f}")
    print(f"Minimum Required: {min_f1:.4f}")
    print(f"Test samples: {len(texts)}")
    
    assert f1 >= min_f1, \
        f"Model F1 score {f1:.4f} is below minimum threshold {min_f1:.4f}"


@pytest.mark.asyncio
async def test_model_outputs_valid(model_loader, test_data):
    """Test that all model outputs are valid."""
    texts = test_data.get("texts", [])
    
    if not texts:
        pytest.skip("Test data is empty")
    
    for text in texts:
        label, score = model_loader.predict(text)
        
        # Check label is valid
        assert label in ["POSITIVE", "NEGATIVE"], \
            f"Invalid label: {label}"
        
        # Check score is valid
        assert isinstance(score, (int, float)), \
            f"Score should be numeric, got {type(score)}"
        assert 0.0 <= score <= 1.0, \
            f"Score should be between 0 and 1, got {score}"

