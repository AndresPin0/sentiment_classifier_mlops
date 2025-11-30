"""
Script to run tests with environment variables from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    print("Warning: .env file not found")

# Set environment variables for pytest
os.environ["MODEL_PATH"] = os.getenv("MODEL_PATH", "./models/model.onnx")
os.environ["TEST_DATA_PATH"] = os.getenv("TEST_DATA_PATH", "./test_data/test_data.parquet")
os.environ["MIN_ACCURACY"] = os.getenv("MIN_ACCURACY", "0.80")
os.environ["MIN_F1"] = os.getenv("MIN_F1", "0.80")

if __name__ == "__main__":
    import pytest
    import sys
    
    # Run pytest
    sys.exit(pytest.main(["-v", "tests/"]))

