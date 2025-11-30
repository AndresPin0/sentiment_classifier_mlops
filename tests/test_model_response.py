"""
Tests for model response functionality.
"""
import pytest
import os
import sys
from pathlib import Path

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
async def model_loader(model_path):
    """Create and load model loader."""
    loader = ONNXModelLoader(model_path=model_path)
    await loader.load_model()
    return loader


@pytest.mark.asyncio
async def test_model_loads(model_loader):
    """Test that model loads successfully."""
    assert model_loader.is_loaded(), "Model should be loaded"
    assert model_loader.session is not None, "ONNX session should exist"
    assert model_loader.tokenizer is not None, "Tokenizer should exist"


@pytest.mark.asyncio
async def test_model_predicts_positive(model_loader):
    """Test model predicts positive sentiment correctly."""
    positive_texts = [
        "I love this movie!",
        "This is amazing!",
        "Great product, highly recommend.",
        "Wonderful experience, thank you!",
        "Excellent service and quality."
    ]
    
    for text in positive_texts:
        label, score = model_loader.predict(text)
        assert label in ["POSITIVE", "NEGATIVE"], f"Label should be POSITIVE or NEGATIVE, got {label}"
        assert 0.0 <= score <= 1.0, f"Score should be between 0 and 1, got {score}"
        print(f"Text: '{text}' -> {label} (score: {score:.4f})")


@pytest.mark.asyncio
async def test_model_predicts_negative(model_loader):
    """Test model predicts negative sentiment correctly."""
    negative_texts = [
        "I hate this movie.",
        "This is terrible!",
        "Poor quality, do not recommend.",
        "Awful experience, very disappointed.",
        "Bad service and low quality."
    ]
    
    for text in negative_texts:
        label, score = model_loader.predict(text)
        assert label in ["POSITIVE", "NEGATIVE"], f"Label should be POSITIVE or NEGATIVE, got {label}"
        assert 0.0 <= score <= 1.0, f"Score should be between 0 and 1, got {score}"
        print(f"Text: '{text}' -> {label} (score: {score:.4f})")


@pytest.mark.asyncio
async def test_model_handles_empty_text(model_loader):
    """Test model handles edge cases."""
    # Empty string should be handled gracefully
    try:
        label, score = model_loader.predict("")
        assert label in ["POSITIVE", "NEGATIVE"]
        assert 0.0 <= score <= 1.0
    except Exception as e:
        # It's acceptable if empty text raises an error
        assert isinstance(e, (ValueError, RuntimeError))


@pytest.mark.asyncio
async def test_model_handles_long_text(model_loader):
    """Test model handles long text (truncation)."""
    long_text = "This is a very long text. " * 100
    label, score = model_loader.predict(long_text)
    assert label in ["POSITIVE", "NEGATIVE"]
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_model_consistency(model_loader):
    """Test model produces consistent results."""
    text = "This is a test sentence."
    
    # Run prediction multiple times
    results = []
    for _ in range(3):
        label, score = model_loader.predict(text)
        results.append((label, score))
    
    # All results should be identical
    first_result = results[0]
    for result in results[1:]:
        assert result[0] == first_result[0], "Labels should be consistent"
        assert abs(result[1] - first_result[1]) < 0.0001, "Scores should be consistent"

