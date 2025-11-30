import os
import sys
import subprocess
import requests
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


def download_data(data_url: str, output_path: str):
    output_path_obj = Path(output_path)
    if output_path_obj.suffix.lower() not in ['.parquet', '.json']:
        output_path = str(output_path_obj.with_suffix('.parquet'))
    elif output_path_obj.suffix.lower() == '.json':
        print("Warning: Output format is JSON. Consider using .parquet for better performance.")
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading test data from: {data_url}")
    print(f"Output path: {output_path}")
    
    try:
        if "drive.google.com" in data_url or "drive" in data_url.lower():
            file_id = None
            if "/d/" in data_url:
                file_id = data_url.split("/d/")[1].split("/")[0]
            elif "id=" in data_url:
                file_id = data_url.split("id=")[1].split("&")[0]
            
            if file_id:
                print(f"Downloading from Google Drive with file ID: {file_id}")
                print("Using curl to handle large files and avoid anti-bot blocks...")
                
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                
                curl_available = False
                try:
                    subprocess.run(["curl", "--version"], 
                                  capture_output=True, 
                                  check=True, 
                                  timeout=5)
                    curl_available = True
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    pass
                
                if curl_available:
                    print("Using curl for download...")
                    temp_file = str(Path(output_path).parent / f"temp_download_{file_id}")
                    cmd = [
                        "curl",
                        "-L",
                        "-C", "-",
                        "-o", temp_file,
                        download_url
                    ]
                    result = subprocess.run(cmd, check=True)
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        Path(temp_file).rename(output_path)
                        print("✓ Download successful with curl")
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            return
                else:
                    print("curl not available, using requests with session...")
                    session = requests.Session()
                    
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': '*/*',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Referer': 'https://drive.google.com/',
                    }
                    
                    response = session.get(download_url, headers=headers, stream=True, allow_redirects=True)
                    
                    if 'virus scan warning' in response.text.lower() or 'download quota exceeded' in response.text.lower():
                        print("Large file detected, handling confirmation page...")
                        import re
                        confirm_match = re.search(r'confirm=([^&]+)', response.text)
                        if confirm_match:
                            confirm_token = confirm_match.group(1)
                            download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                            response = session.get(download_url, headers=headers, stream=True, allow_redirects=True)
                    
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        raise Exception("Received HTML page - file may require manual download or different permissions")
                    
                    total_size = int(response.headers.get('Content-Length', 0))
                    downloaded = 0
                    
                    temp_file = str(Path(output_path).parent / f"temp_download_{file_id}")
                    with open(temp_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    percent = (downloaded / total_size) * 100
                                    print(f"\rProgress: {percent:.1f}% ({downloaded / 1024:.1f} KB / {total_size / 1024:.1f} KB)", end="")
                    
                    print()
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        Path(temp_file).rename(output_path)
                        print("✓ Download successful with requests")
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            return
            else:
                print("Could not extract file ID, trying direct download...")
                curl_available = False
                try:
                    subprocess.run(["curl", "--version"], 
                                  capture_output=True, 
                                  check=True, 
                                  timeout=5)
                    curl_available = True
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    pass
                
                if curl_available:
                    temp_file = str(Path(output_path).parent / "temp_download_direct")
                    subprocess.run(["curl", "-L", "-o", temp_file, data_url], check=True)
                    if os.path.exists(temp_file):
                        Path(temp_file).rename(output_path)
                else:
                    temp_file = str(Path(output_path).parent / "temp_download_direct")
                    response = requests.get(data_url, stream=True)
                    response.raise_for_status()
                    with open(temp_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    if os.path.exists(temp_file):
                        Path(temp_file).rename(output_path)
        else:
            print("Downloading directly from URL...")
            curl_available = False
            try:
                subprocess.run(["curl", "--version"], 
                              capture_output=True, 
                              check=True, 
                              timeout=5)
                curl_available = True
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            if curl_available:
                temp_file = str(Path(output_path).parent / "temp_download_direct")
                subprocess.run(["curl", "-L", "-o", temp_file, data_url], check=True)
                if os.path.exists(temp_file):
                    Path(temp_file).rename(output_path)
            else:
                temp_file = str(Path(output_path).parent / "temp_download_direct")
                response = requests.get(data_url, stream=True)
                response.raise_for_status()
                with open(temp_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                if os.path.exists(temp_file):
                    Path(temp_file).rename(output_path)
        
        downloaded_path = Path(output_path)
        if not downloaded_path.exists():
            parent_dir = downloaded_path.parent
            base_name = downloaded_path.stem
            for ext in ['.parquet', '.json', '.csv']:
                alt_path = parent_dir / f"{base_name}{ext}"
                if alt_path.exists():
                    print(f"Found downloaded file with extension {ext}, renaming to {downloaded_path.name}...")
                    alt_path.rename(downloaded_path)
                    break
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Download failed: {output_path} not found")
        
        final_path = Path(output_path)
        expected_ext = final_path.suffix.lower()
        
        if expected_ext == '.parquet':
            actual_files = list(final_path.parent.glob(f"{final_path.stem}.*"))
            for actual_file in actual_files:
                if actual_file.suffix.lower() != '.parquet' and actual_file != final_path:
                    print(f"Renaming {actual_file.name} to {final_path.name}...")
                    actual_file.rename(final_path)
                    break
        
        file_size = os.path.getsize(output_path)
        file_format = Path(output_path).suffix.upper() or "unknown"
        size_unit = "MB" if file_size > 1024 * 1024 else "KB"
        size_value = file_size / (1024 * 1024) if file_size > 1024 * 1024 else file_size / 1024
        print(f"Test data downloaded successfully: {output_path}")
        print(f"Format: {file_format}, Size: {size_value:.2f} {size_unit}")
        
    except Exception as e:
        print(f"Error downloading test data: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    data_url = os.getenv("DATA_URL", "")
    output_path = os.getenv("TEST_DATA_PATH", os.getenv("OUTPUT_PATH", "./test_data/test_data.parquet"))
    
    if not data_url:
        print("Error: DATA_URL environment variable is not set")
        print("Make sure you have a .env file with DATA_URL configured")
        sys.exit(1)
    
    download_data(data_url, output_path)

