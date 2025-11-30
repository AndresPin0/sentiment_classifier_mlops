import os
import re
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


MODEL_URL = os.getenv("MODEL_URL")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/model.onnx")


if not MODEL_URL:
    print("MODEL_URL is not set")
    sys.exit(1)


def get_confirm_token(response_text):
    patterns = [
        r"confirm=([0-9A-Za-z-_]+)",
        r'name="confirm" value="([^"]+)"'
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1)

    return None


def download_file_from_google_drive(file_id, destination):
    session = requests.Session()

    URL = "https://drive.google.com/uc?export=download"
    response = session.get(URL, params={"id": file_id}, stream=True)

    token = get_confirm_token(response.text)

    if token:
        print("Large file detected, applying confirmation token...")
        response = session.get(
            URL,
            params={"id": file_id, "confirm": token},
            stream=True
        )

    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type.lower():
        raise ValueError("Received HTML instead of file (Drive blocked it).")

    save_response_content(response, destination)


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    os.makedirs(Path(destination).parent, exist_ok=True)

    total = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    if total < 100_000:
        raise ValueError("Downloaded file is too small to be a valid ONNX model.")

    print(f"Model downloaded successfully: {destination} ({total / 1024 / 1024:.2f} MB)")


def extract_file_id(url):
    if "id=" in url:
        return url.split("id=")[1].split("&")[0]
    if "/d/" in url:
        return url.split("/d/")[1].split("/")[0]
    return None


def main():
    print(f"MODEL_URL = {MODEL_URL[:60]}...")
    print(f"MODEL_PATH = {MODEL_PATH}")

    if "drive.google.com" in MODEL_URL:
        file_id = extract_file_id(MODEL_URL)
        if not file_id:
            raise ValueError("❌ Could not extract Google Drive file ID")

        print(f"Downloading from Drive ID: {file_id}")
        download_file_from_google_drive(file_id, MODEL_PATH)

    else:
        print("Direct download...")
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        save_response_content(r, MODEL_PATH)


if __name__ == "__main__":
    main()
